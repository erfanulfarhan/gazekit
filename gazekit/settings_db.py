"""Minimal SQLite store for named camera profiles.

Mirrors the on-disk schema the GazeAt backend already reads, so profiles
written here can be applied to it directly. Kept deliberately small: a single
table of JSON blobs keyed by name is all a handful of camera profiles needs.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["SettingsDB"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS UserSetting (
    name       TEXT PRIMARY KEY,
    cfg        TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SettingsDB:
    """Named camera profiles, stored as JSON in SQLite."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(_SCHEMA)

    def get(self, name: str) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT cfg FROM UserSetting WHERE name = ?", (name,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, name: str, cfg: dict) -> None:
        """Insert or update a profile, preserving its original created_at."""
        now = _now()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO UserSetting (name, cfg, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET cfg = excluded.cfg,
                                                updated_at = excluded.updated_at
                """,
                (name, json.dumps(cfg), now, now),
            )

    def names(self) -> list[str]:
        with sqlite3.connect(self.path) as conn:
            return [r[0] for r in conn.execute(
                "SELECT name FROM UserSetting ORDER BY name"
            )]

    def find(self, fragment: str) -> list[str]:
        """Profile names containing `fragment`, case-insensitively."""
        return [n for n in self.names() if fragment.lower() in n.lower()]

    def delete(self, name: str) -> bool:
        with sqlite3.connect(self.path) as conn:
            return conn.execute(
                "DELETE FROM UserSetting WHERE name = ?", (name,)
            ).rowcount > 0
