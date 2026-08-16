"""Gaze redirection geometry.

Deliberately free of TensorFlow, OpenCV and numpy, so the angle math can be
tested in milliseconds and reasoned about without a model in the loop.

Coordinates are camera-centred and in centimetres. `camera_offset` is the
camera's position relative to the centre of the screen, so a camera mounted
above the screen has a negative Y.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CameraGeometry",
    "estimate_gaze_angle",
    "solve_focal_length",
    "camera_offset_from_display",
]


@dataclass(frozen=True)
class CameraGeometry:
    """A physical camera-and-screen setup.

    `focal_length` is in pixels *at reference_width*, and is rescaled to the
    live frame width before use. Skipping that rescale is a classic defect:
    depth is estimated from apparent inter-pupillary distance, so a focal
    length calibrated at 640px underestimates distance threefold at 1920px,
    and the redirection angle inflates with it.
    """

    focal_length: float = 650.0
    ipd: float = 6.3
    camera_offset: tuple[float, float, float] = (0.0, -10.0, -1.0)
    reference_width: int = 640
    clamp_deg: float | None = 25.0

    def effective_focal(self, frame_width: int) -> float:
        return self.focal_length * (frame_width / self.reference_width)


def _clamp(value: float, limit: float | None) -> float:
    if limit is None:
        return value
    return math.copysign(min(abs(value), limit), value)


def estimate_gaze_angle(
    le_center: tuple[float, float],
    re_center: tuple[float, float],
    video_size: tuple[int, int],
    geometry: CameraGeometry,
    _skip_scaling: bool = False,
) -> tuple[list[float], list[float]]:
    """Return ([vertical, horizontal] degrees, [x, y, z] eye position in cm).

    `_skip_scaling` reproduces the unscaled-focal defect so the regression
    test can assert the difference. Never set it in production.
    """
    width, height = video_size
    focal = geometry.focal_length if _skip_scaling else geometry.effective_focal(width)

    ipd_pixels = math.hypot(le_center[0] - re_center[0], le_center[1] - re_center[1])
    if ipd_pixels <= 0:
        return [0.0, 0.0], [0.0, 0.0, 0.0]

    # Depth from apparent inter-pupillary distance. Negative is in front.
    eye_z = -(focal * geometry.ipd) / ipd_pixels
    depth = abs(eye_z)

    off_x, off_y, off_z = geometry.camera_offset
    eye_x = -depth * (le_center[0] + re_center[0] - width) / (2 * focal) + off_x
    eye_y = depth * (le_center[1] + re_center[1] - height) / (2 * focal) + off_y
    eye_position = [eye_x, eye_y, eye_z]

    # Angle from the eye to screen centre, plus the angle back to the camera.
    # The sum is how far the gaze has to be redirected to meet the lens.
    a_v = math.degrees(math.atan((0.0 - eye_y) / (0.0 - eye_z)))
    a_h = math.degrees(math.atan((0.0 - eye_x) / (0.0 - eye_z)))
    a_v += math.degrees(math.atan((eye_y - off_y) / (off_z - eye_z)))
    a_h += math.degrees(math.atan((eye_x - off_x) / (off_z - eye_z)))

    # Warping networks are trained over a limited angular range. Past it the
    # spatial transformer degenerates and the correction silently vanishes,
    # which is indistinguishable from the feature being switched off. Clamping
    # converts that into visibly reduced correction, which can be debugged.
    return [_clamp(a_v, geometry.clamp_deg), _clamp(a_h, geometry.clamp_deg)], eye_position


def solve_focal_length(
    ipd_pixels: float,
    distance_cm: float,
    ipd_cm: float = 6.3,
    frame_width: int = 1920,
    reference_width: int = 640,
) -> float:
    """Solve focal length in pixels at `reference_width` from one measurement.

    Sit at a measured distance, detect the irises, and the optics follow from
    `f = ipd_pixels * distance / ipd_cm`. This replaces nudging arrow keys
    until it subjectively looks right.
    """
    if ipd_pixels <= 0 or distance_cm <= 0:
        raise ValueError("ipd_pixels and distance_cm must be positive")
    return (ipd_pixels * distance_cm / ipd_cm) * (reference_width / frame_width)


def camera_offset_from_display(
    diagonal_inches: float,
    width_px: int,
    height_px: int,
    camera_above_top_cm: float = 0.6,
) -> tuple[float, float, float]:
    """Compute camera offset from display geometry instead of guessing it.

    Assumes a camera centred horizontally just above the panel, which holds
    for every built-in Mac display. Returns centimetres from screen centre.
    """
    if height_px <= 0 or width_px <= 0 or diagonal_inches <= 0:
        raise ValueError("display dimensions must be positive")
    aspect = width_px / height_px
    height_cm = (diagonal_inches / math.sqrt(1 + aspect**2)) * 2.54
    return (0.0, -(height_cm / 2 + camera_above_top_cm), -1.0)
