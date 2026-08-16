"""Camera and display discovery, read from the system rather than assumed."""

from __future__ import annotations

import math
import re
import subprocess

__all__ = ["list_cameras", "find_camera", "main_display_geometry", "DISPLAY_SIZES"]

# Built-in Mac panels, keyed by native resolution. Apple does not report a
# diagonal anywhere queryable, so it has to be inferred.
DISPLAY_SIZES = {
    (2560, 1664): 13.6,   # MacBook Air 13"
    (2880, 1864): 15.3,   # MacBook Air 15"
    (3024, 1964): 14.2,   # MacBook Pro 14"
    (3456, 2234): 16.2,   # MacBook Pro 16"
    (2560, 1600): 13.3,   # older Air / Pro 13"
    (2880, 1800): 15.4,   # older Pro 15"
}


def list_cameras() -> list[str]:
    """Camera names in AVFoundation order, which is OpenCV's index order."""
    try:
        import AVFoundation  # type: ignore

        session = AVFoundation.AVCaptureDeviceDiscoverySession.\
            discoverySessionWithDeviceTypes_mediaType_position_(
                [
                    "AVCaptureDeviceTypeBuiltInWideAngleCamera",
                    "AVCaptureDeviceTypeExternalUnknown",
                    "AVCaptureDeviceTypeContinuityCamera",
                ],
                "vide",
                0,
            )
        return [str(d.localizedName()) for d in session.devices()]
    except Exception:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        return [m.group(1).strip() for m in re.finditer(r"^\s{4}(\S.*):\s*$", out, re.M)]


def find_camera(name_fragment: str) -> tuple[int, str]:
    """Resolve a camera by substring to (index, full name).

    Raises rather than picking one, because silently grabbing the wrong camera
    produces a calibration that is subtly wrong instead of obviously broken.
    """
    cameras = list_cameras()
    matches = [(i, n) for i, n in enumerate(cameras)
               if name_fragment.lower() in n.lower()]
    if not matches:
        raise LookupError(f"No camera matching {name_fragment!r}. Available: {cameras}")
    if len(matches) > 1:
        raise LookupError(f"{name_fragment!r} is ambiguous: {[n for _, n in matches]}")
    return matches[0]


def main_display_geometry() -> tuple[float, int, int]:
    """(diagonal_inches, width_px, height_px) for the built-in display."""
    out = subprocess.run(
        ["system_profiler", "SPDisplaysDataType"],
        capture_output=True, text=True, timeout=30,
    ).stdout
    for m in re.finditer(r"Resolution:\s*(\d+)\s*x\s*(\d+)", out):
        res = (int(m.group(1)), int(m.group(2)))
        if res in DISPLAY_SIZES:
            return DISPLAY_SIZES[res], res[0], res[1]
    return 13.6, 2560, 1664


def screen_height_cm(diagonal_inches: float, width_px: int, height_px: int) -> float:
    aspect = width_px / height_px
    return (diagonal_inches / math.sqrt(1 + aspect**2)) * 2.54
