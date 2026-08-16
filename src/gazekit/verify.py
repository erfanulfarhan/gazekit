"""Decide whether gaze correction is actually running.

The app's log looks identical whether correction works or silently does
nothing, so it cannot answer the only question that matters. The backend's
input and output frames are memory-mapped files on disk, so they can be
compared directly, with no camera permission and no guessing.

A working corrector copies the frame and replaces only the eye regions:

    active   -> a few hundred large differences inside one eye-sized box
    inactive -> frames identical, or differences smeared across the frame

Three things make this harder than a naive diff, each learned by getting it
wrong first:

1. The two mmaps are not synchronised. A single read compares frame N against
   corrected frame M, and any head movement swamps the warp. So sample many
   pairs and keep the most aligned one.
2. Alignment cannot be scored by mean or median whole-frame difference: both
   are dominated by sensor noise, which says nothing about whether the frames
   depict the same instant. Counting differences too large to be noise
   measures misalignment directly.
3. A quit app leaves both files frozen, which otherwise reads as a valid
   measurement of a stale snapshot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .app import INPUT_MMAP, OUTPUT_MMAP

__all__ = ["FrameSpec", "VerifyResult", "verify", "load_frame"]

HEADER_BYTES = 64


@dataclass(frozen=True)
class FrameSpec:
    width: int = 1920
    height: int = 1080
    channels: int = 4  # BGRA

    @property
    def nbytes(self) -> int:
        return self.width * self.height * self.channels


@dataclass
class VerifyResult:
    active: bool
    live: bool
    reason: str
    changed_pixels: int
    total_pixels: int
    max_difference: int
    strong_pixels: int
    region: tuple[int, int, int, int] | None  # x0, x1, y0, y1
    coverage: float
    frames: tuple = ()  # (input, output) arrays for the chosen pair

    @property
    def changed_fraction(self) -> float:
        return self.changed_pixels / self.total_pixels if self.total_pixels else 0.0


def load_frame(path: Path, spec: FrameSpec):
    import numpy as np

    raw = path.read_bytes()
    need = HEADER_BYTES + spec.nbytes
    if len(raw) < need:
        raise RuntimeError(
            f"{path.name} is {len(raw)} bytes, expected at least {need}. "
            "Do width/height match the app's resolution?"
        )
    body = raw[HEADER_BYTES:need]
    return np.frombuffer(body, dtype=np.uint8).reshape(
        spec.height, spec.width, spec.channels
    )


def _best_aligned_pair(spec: FrameSpec, tries: int, strong: int,
                       input_path: Path, output_path: Path):
    """Sample pairs, keep the one with fewest large differences.

    Fewest large differences means best temporal alignment, since an aligned
    pair differs strongly only where the eyes were rewritten.
    """
    import numpy as np

    best = None
    best_score = None
    first = None
    live = False

    for _ in range(max(1, tries)):
        a = load_frame(input_path, spec)
        b = load_frame(output_path, spec)
        if first is None:
            first = a.copy()
        elif not live and not np.array_equal(a, first):
            live = True
        d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
        score = int((d > strong).sum())
        if best_score is None or score < best_score:
            best_score, best = score, (a, b)
        time.sleep(0.04)

    return best, live


def verify(spec: FrameSpec = FrameSpec(), tries: int = 40, threshold: int = 8,
           strong: int = 60, max_coverage: float = 0.25,
           input_path: Path = INPUT_MMAP,
           output_path: Path = OUTPUT_MMAP) -> VerifyResult:
    """Compare the backend's input and output frames and judge the result."""
    import numpy as np

    (inp, out), live = _best_aligned_pair(spec, tries, strong, input_path, output_path)
    total = spec.width * spec.height

    diff = np.abs(inp.astype(np.int16) - out.astype(np.int16)).max(axis=2)
    changed = diff > threshold
    strong_mask = diff > strong

    base = dict(live=live, changed_pixels=int(changed.sum()), total_pixels=total,
                max_difference=int(diff.max()), strong_pixels=int(strong_mask.sum()),
                frames=(inp, out))

    if not changed.any():
        return VerifyResult(active=False, reason="frames are byte-identical",
                            region=None, coverage=0.0, **base)

    if not strong_mask.any():
        return VerifyResult(
            active=False,
            reason=f"no differences above {strong}; looks like sensor noise",
            region=None, coverage=0.0, **base,
        )

    # Percentiles, not min/max: a few stray noise pixels above the strong
    # threshold would otherwise stretch the box across the whole frame.
    ys, xs = np.where(strong_mask)
    x0, x1 = int(np.percentile(xs, 2)), int(np.percentile(xs, 98))
    y0, y1 = int(np.percentile(ys, 2)), int(np.percentile(ys, 98))
    coverage = ((x1 - x0) * (y1 - y0)) / total

    if coverage > max_coverage:
        return VerifyResult(
            active=False,
            reason="strong differences are spread across the frame, so this is "
                   "probably motion between snapshots rather than a warp",
            region=(x0, x1, y0, y1), coverage=coverage, **base,
        )

    return VerifyResult(active=True, reason="localised eye-region warp",
                        region=(x0, x1, y0, y1), coverage=coverage, **base)


def save_comparison(result: VerifyResult, path: Path, pad: int = 45,
                    scale: int = 2) -> None:
    """Write a before/after crop of the region the warp touched."""
    import cv2
    import numpy as np

    if not result.region or not result.frames:
        raise ValueError("no region to crop; verification did not locate a warp")

    inp, out = result.frames
    x0, x1, y0, y1 = result.region
    cx0, cx1 = max(0, x0 - pad), min(inp.shape[1], x1 + pad)
    cy0, cy1 = max(0, y0 - pad), min(inp.shape[0], y1 + pad)

    panels = []
    for frame, label in ((inp, "BEFORE (raw camera)"), (out, "AFTER (gaze corrected)")):
        crop = cv2.resize(frame[cy0:cy1, cx0:cx1, :3], None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)
        for colour, thickness in (((0, 0, 0), 4), ((255, 255, 255), 2)):
            cv2.putText(crop, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, colour, thickness)
        panels.append(crop)

    gap = np.full((6, panels[0].shape[1], 3), 255, np.uint8)
    cv2.imwrite(str(path), np.vstack([panels[0], gap, panels[1]]))
