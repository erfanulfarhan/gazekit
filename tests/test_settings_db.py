"""Tests for the camera profile store."""

import pytest

from gazekit.settings_db import SettingsDB

PROFILE = {"focal_length": 650.0, "ipd": 6.3, "camera_offset": [0.0, -10.0, -1.0]}


@pytest.fixture
def db(tmp_path):
    return SettingsDB(tmp_path / "profiles.db")


class TestRoundTrip:
    def test_put_then_get(self, db):
        db.put("MacBook Air Camera", PROFILE)
        assert db.get("MacBook Air Camera") == PROFILE

    def test_missing_returns_none(self, db):
        assert db.get("nope") is None

    def test_put_overwrites_in_place(self, db):
        db.put("cam", PROFILE)
        db.put("cam", {**PROFILE, "focal_length": 720.0})
        assert db.get("cam")["focal_length"] == 720.0
        assert db.names() == ["cam"]

    def test_creates_parent_directory(self, tmp_path):
        nested = SettingsDB(tmp_path / "a" / "b" / "profiles.db")
        nested.put("cam", PROFILE)
        assert nested.get("cam") == PROFILE


class TestMultipleProfiles:
    def test_profiles_are_independent(self, db):
        iphone = {**PROFILE, "focal_length": 380.0}
        db.put("MacBook Air Camera", PROFILE)
        db.put("iPhone Camera", iphone)
        assert db.get("MacBook Air Camera") == PROFILE
        assert db.get("iPhone Camera") == iphone

    def test_names_sorted(self, db):
        for name in ("zeta", "alpha", "mid"):
            db.put(name, PROFILE)
        assert db.names() == ["alpha", "mid", "zeta"]

    def test_find_is_case_insensitive_substring(self, db):
        db.put("MacBook Air Camera", PROFILE)
        db.put("iPhone Camera", PROFILE)
        assert db.find("macbook") == ["MacBook Air Camera"]
        assert len(db.find("camera")) == 2
        assert db.find("nothing") == []


class TestDelete:
    def test_delete_removes(self, db):
        db.put("cam", PROFILE)
        assert db.delete("cam") is True
        assert db.get("cam") is None

    def test_delete_missing_is_false(self, db):
        assert db.delete("cam") is False


class TestPersistence:
    def test_survives_reopen(self, tmp_path):
        path = tmp_path / "profiles.db"
        SettingsDB(path).put("cam", PROFILE)
        assert SettingsDB(path).get("cam") == PROFILE
