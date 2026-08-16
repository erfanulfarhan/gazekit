"""Tests for backend process selection and settings IO.

GazeAt leaks a backend on every quit, and several were observed running at
once, each burning most of a core and writing to the same output mmap. So
reaping is routine, and "which ones do we kill" is worth testing: getting it
wrong either leaves the leak in place or kills the user's live session.
"""

import json

import pytest

from gazekit.app import read_settings, select_stale_pids, write_settings


class TestReapAll:
    def test_kills_everything_when_not_keeping(self):
        assert select_stale_pids([10, 20, 30], keep_newest=False) == [10, 20, 30]

    def test_empty_stays_empty(self):
        assert select_stale_pids([], keep_newest=False) == []


class TestKeepNewest:
    def test_spares_the_highest_pid(self):
        assert select_stale_pids([10, 20, 30], keep_newest=True) == [10, 20]

    def test_single_backend_is_never_killed(self):
        """The regression: a lone backend is the live session, not a leak."""
        assert select_stale_pids([67691], keep_newest=True) == []

    def test_empty_stays_empty(self):
        assert select_stale_pids([], keep_newest=True) == []

    def test_unsorted_input_spares_the_max_not_the_last(self):
        assert select_stale_pids([30, 10, 20], keep_newest=True) == [10, 20]

    def test_real_observed_leak(self):
        """Five backends seen at once; only the newest was live."""
        assert select_stale_pids(
            [67008, 67402, 67633, 67655, 67691], keep_newest=True
        ) == [67008, 67402, 67633, 67655]

    @pytest.mark.parametrize("pids", [[1], [1, 2], [5, 3, 9], [100, 2, 50, 7]])
    def test_never_returns_the_survivor(self, pids):
        assert max(pids) not in select_stale_pids(pids, keep_newest=True)


class TestSettingsIO:
    def test_merge_preserves_unrelated_keys(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({
            "camera_offset": [0, -21, -1], "eye_scale": 0.85,
            "resolution": {"width": 1920, "height": 1080},
        }))
        write_settings({"camera_offset": [0, -10, -1]}, path=path, backup=False)
        result = read_settings(path)
        assert result["camera_offset"] == [0, -10, -1]
        assert result["eye_scale"] == 0.85
        assert result["resolution"] == {"width": 1920, "height": 1080}

    def test_backup_is_written_and_holds_the_original(self, tmp_path):
        path = tmp_path / "settings.json"
        original = {"camera_offset": [0, -21, -1]}
        path.write_text(json.dumps(original))
        backup = write_settings({"camera_offset": [0, -10, -1]}, path=path)
        assert backup is not None and backup.exists()
        assert json.loads(backup.read_text()) == original
