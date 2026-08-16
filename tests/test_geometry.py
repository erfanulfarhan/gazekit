"""Golden-value tests for gaze geometry.

These lock down the resolution-scaling defect: the same physical scene must
produce the same redirection angle regardless of capture resolution.
"""

import math
import pytest

from gazekit.geometry import (
    CameraGeometry,
    camera_offset_from_display,
    estimate_gaze_angle,
    solve_focal_length,
)

IPD_CM = 6.3
REF_W = 640


def synth_eyes(width, height, f_ref, distance_cm, eye_y_frac=0.44):
    """Synthesise eye centres for a person centred at a given distance."""
    f_eff = f_ref * (width / REF_W)
    ipd_px = f_eff * IPD_CM / distance_cm
    cx, cy = width / 2, height * eye_y_frac
    return (cx - ipd_px / 2, cy), (cx + ipd_px / 2, cy)


def geom(offset=(0.0, -10.0, -1.0), focal=650.0, clamp=25.0):
    return CameraGeometry(
        focal_length=focal, ipd=IPD_CM, camera_offset=offset,
        reference_width=REF_W, clamp_deg=clamp,
    )


class TestResolutionInvariance:
    """The defect this suite exists to prevent."""

    @pytest.mark.parametrize("width,height", [(640, 480), (1280, 720), (1920, 1080)])
    def test_angle_is_invariant_to_resolution(self, width, height):
        le, re = synth_eyes(width, height, 650.0, 60.0)
        alpha, _ = estimate_gaze_angle(le, re, (width, height), geom())
        # Same physical scene at 640 is the reference.
        le0, re0 = synth_eyes(640, 480, 650.0, 60.0)
        expected, _ = estimate_gaze_angle(le0, re0, (640, 480), geom())
        assert alpha[0] == pytest.approx(expected[0], abs=0.15)
        assert alpha[1] == pytest.approx(expected[1], abs=0.15)

    def test_unscaled_focal_would_have_failed(self):
        """Guard the regression directly: ignoring reference_width inflates the angle."""
        le, re = synth_eyes(1920, 1080, 650.0, 60.0)
        correct, _ = estimate_gaze_angle(le, re, (1920, 1080), geom())
        broken, _ = estimate_gaze_angle(le, re, (1920, 1080), geom(), _skip_scaling=True)
        assert broken[0] > correct[0] * 1.8


class TestDepthEstimation:
    @pytest.mark.parametrize("distance", [40.0, 55.0, 60.0, 80.0])
    def test_recovers_true_distance(self, distance):
        le, re = synth_eyes(1920, 1080, 650.0, distance)
        _, eye_pos = estimate_gaze_angle(le, re, (1920, 1080), geom())
        assert abs(eye_pos[2]) == pytest.approx(distance, rel=0.02)


class TestAngleMagnitude:
    def test_matches_hand_computed_geometry(self):
        """Camera 10cm above screen centre, viewer at 60cm -> about 9.5 degrees."""
        le, re = synth_eyes(1920, 1080, 650.0, 60.0)
        alpha, _ = estimate_gaze_angle(le, re, (1920, 1080), geom())
        assert alpha[0] == pytest.approx(math.degrees(math.atan(10.0 / 60.0)), abs=1.0)

    def test_closer_viewer_needs_more_correction(self):
        a_near, _ = estimate_gaze_angle(
            *synth_eyes(1920, 1080, 650.0, 40.0), (1920, 1080), geom()
        )
        a_far, _ = estimate_gaze_angle(
            *synth_eyes(1920, 1080, 650.0, 80.0), (1920, 1080), geom()
        )
        assert a_near[0] > a_far[0]

    def test_centred_viewer_has_no_horizontal_correction(self):
        le, re = synth_eyes(1920, 1080, 650.0, 60.0)
        alpha, _ = estimate_gaze_angle(le, re, (1920, 1080), geom())
        assert alpha[1] == pytest.approx(0.0, abs=0.5)


class TestClamp:
    """Bad geometry must degrade to less correction, never a dead warp field."""

    def test_absurd_offset_is_clamped(self):
        le, re = synth_eyes(1920, 1080, 650.0, 25.0)
        alpha, _ = estimate_gaze_angle(le, re, (1920, 1080), geom(offset=(0, -60, -1)))
        assert abs(alpha[0]) <= 25.0

    def test_clamp_preserves_sign(self):
        le, re = synth_eyes(1920, 1080, 650.0, 25.0)
        alpha, _ = estimate_gaze_angle(le, re, (1920, 1080), geom(offset=(0, 60, -1)))
        assert alpha[0] <= 0 and abs(alpha[0]) <= 25.0

    def test_normal_geometry_is_untouched(self):
        le, re = synth_eyes(1920, 1080, 650.0, 60.0)
        a_clamped, _ = estimate_gaze_angle(le, re, (1920, 1080), geom(clamp=25.0))
        a_free, _ = estimate_gaze_angle(le, re, (1920, 1080), geom(clamp=None))
        assert a_clamped[0] == pytest.approx(a_free[0], abs=1e-9)


class TestReturnsFloats:
    def test_angles_are_not_truncated(self):
        le, re = synth_eyes(1920, 1080, 650.0, 63.0)
        alpha, _ = estimate_gaze_angle(le, re, (1920, 1080), geom())
        assert isinstance(alpha[0], float)
        assert alpha[0] != int(alpha[0]), "sub-degree precision must survive"


class TestSolvers:
    def test_solve_focal_length_round_trips(self):
        le, re = synth_eyes(1920, 1080, 650.0, 60.0)
        ipd_px = abs(re[0] - le[0])
        f = solve_focal_length(ipd_px, 60.0, IPD_CM, 1920, REF_W)
        assert f == pytest.approx(650.0, rel=0.01)

    def test_camera_offset_for_13_6_inch_macbook_air(self):
        off = camera_offset_from_display(13.6, 2560, 1664)
        assert off[0] == 0.0
        assert off[1] == pytest.approx(-10.0, abs=0.8)

    def test_larger_display_pushes_camera_further_from_centre(self):
        small = camera_offset_from_display(13.6, 2560, 1664)
        large = camera_offset_from_display(27.0, 2560, 1440)
        assert abs(large[1]) > abs(small[1])
