"""Talking to the GazeAt app from outside it.

GazeAt ships as a signed, notarised binary with hardened runtime, so its
bundle cannot be modified. Everything here works through the two files it
leaves in its container: a settings JSON and a SQLite database.

Worth knowing, because it is not obvious and cost real debugging time:

- The Swift UI reads the JSON, but the Python backend reads its geometry from
  the SQLite database. Writing only the JSON changes less than you expect.
- The app rewrites the JSON from its in-memory UI state when it quits, so
  edits made while it is running get clobbered. Write while it is closed.
- The app leaks a backend process on every quit. Several accumulate, each
  burning most of a core, and they all write to the same output mmap, so
  stale writers corrupt corrected frames with uncorrected ones.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

__all__ = [
    "CONTAINER", "SETTINGS_JSON", "BACKEND_DB", "INPUT_MMAP", "OUTPUT_MMAP",
    "backend_pids", "select_stale_pids", "reap_backends",
    "app_is_running", "read_settings", "write_settings",
]

CONTAINER = Path.home() / "Library/Containers/com.willywangkaa.gazeatcamera/Data/tmp"
SETTINGS_JSON = CONTAINER / "gaze_correction_settings.json"
BACKEND_DB = CONTAINER / "gaze_user_settings.db"
INPUT_MMAP = CONTAINER / "gaze_input_frame.mmap"
OUTPUT_MMAP = CONTAINER / "gaze_output_frame.mmap"

BACKEND_PATTERN = "gaze_backend"
APP_PATTERN = "GazeAt.app/Contents/MacOS"


def _pgrep(pattern: str) -> list[int]:
    try:
        out = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=10
        ).stdout.split()
    except Exception:
        return []
    return [int(p) for p in out if p.isdigit()]


def backend_pids() -> list[int]:
    return _pgrep(BACKEND_PATTERN)


def app_is_running() -> bool:
    return bool(_pgrep(APP_PATTERN))


def select_stale_pids(pids: list[int], keep_newest: bool) -> list[int]:
    """Which backend PIDs to kill.

    With `keep_newest` the highest PID is always spared, *including when it is
    the only one*. Gating that on there being more than one process inverts
    the meaning in exactly the case that matters: a lone backend is the live
    session, not a leak.
    """
    if not keep_newest or not pids:
        return list(pids)
    newest = max(pids)
    return [p for p in pids if p != newest]


def reap_backends(keep_newest: bool = False) -> list[int]:
    """Terminate stray backends. Returns the PIDs actually signalled."""
    targets = select_stale_pids(backend_pids(), keep_newest)
    if not targets:
        return []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        alive = []
        for pid in targets:
            try:
                os.kill(pid, sig)
                alive.append(pid)
            except ProcessLookupError:
                pass
        if not alive:
            break
        time.sleep(1.0)
    return targets


def read_settings(path: Path = SETTINGS_JSON) -> dict:
    return json.loads(path.read_text())


def write_settings(values: dict, path: Path = SETTINGS_JSON,
                   backup: bool = True) -> Path | None:
    """Merge `values` into the settings JSON, backing the original up first."""
    current = read_settings(path)
    backup_path = None
    if backup:
        backup_path = path.with_suffix(f".json.bak.{int(time.time())}")
        shutil.copy2(path, backup_path)
    current.update(values)
    path.write_text(json.dumps(current))
    return backup_path
