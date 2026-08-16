"""Solve camera geometry from a measurement instead of guessing at it.

GazeAt's own calibration is arrow-key nudging with no ground truth, which is
how a camera offset ends up at 60 cm horizontally and the warp network gets
asked for a 45 degree sideways redirect. Here the numbers are solved:

- `focal_length` from the observed iris separation at a known distance
- `camera_offset` from the display's real physical dimensions
"""

from __future__ import annotations

import statistics
from pathlib import Path

from .cameras import find_camera, main_display_geometry
from .geometry import camera_offset_from_display, solve_focal_length

__all__ = ["measure_ipd_pixels", "solve_profile"]

# Iris centres provided by MediaPipe FaceMesh when refine_landmarks is on.
LEFT_IRIS, RIGHT_IRIS = 468, 473


def measure_ipd_pixels(camera_index: int, width: int, height: int,
                       samples: int = 15) -> float:
    """Median iris-to-iris distance in pixels, over several frames."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. On macOS your "
            "terminal needs camera access: System Settings > Privacy & "
            "Security > Camera."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    )

    readings: list[float] = []
    attempts = 0
    try:
        for _ in range(10):  # let auto-exposure settle
            cap.read()
        while len(readings) < samples and attempts < samples * 8:
            attempts += 1
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            lx, ly = lm[LEFT_IRIS].x * w, lm[LEFT_IRIS].y * h
            rx, ry = lm[RIGHT_IRIS].x * w, lm[RIGHT_IRIS].y * h
            readings.append(((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5)
    finally:
        cap.release()
        mesh.close()

    if len(readings) < 3:
        raise RuntimeError(
            f"Only {len(readings)} face detections. Be in frame, lit from the "
            "front, and looking at the screen."
        )
    return statistics.median(readings)


def solve_profile(camera: str = "MacBook", distance_cm: float = 55.0,
                  ipd_cm: float = 6.3, width: int = 1920, height: int = 1080,
                  samples: int = 15, measure: bool = True,
                  focal_fallback: float = 650.0) -> tuple[str, dict, float | None]:
    """Return (camera name, profile dict, measured ipd_pixels or None).

    With `measure` off, the camera is never opened: the offset still comes
    from real display geometry, and the focal length keeps its default. That
    path is useful when the terminal has no camera permission, and it still
    fixes the error that matters most in practice, a wrong camera offset.
    """
    index, name = find_camera(camera)

    ipd_pixels = None
    focal = focal_fallback
    if measure:
        ipd_pixels = measure_ipd_pixels(index, width, height, samples)
        focal = solve_focal_length(ipd_pixels, distance_cm, ipd_cm, width)

    diagonal, dw, dh = main_display_geometry()
    offset = camera_offset_from_display(diagonal, dw, dh)

    profile = {
        "focal_length": round(focal, 1),
        "ipd": ipd_cm,
        "camera_offset": [round(v, 2) for v in offset],
    }
    return name, profile, ipd_pixels
