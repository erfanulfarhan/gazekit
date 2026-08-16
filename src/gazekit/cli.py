"""gazekit command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import app
from .calibrate import solve_profile
from .settings_db import SettingsDB
from .verify import FrameSpec, save_comparison, verify

DEFAULT_DB = Path.home() / ".gazekit" / "profiles.db"


def _print_profile(name: str, profile: dict, ipd_pixels: float | None,
                   width: int) -> None:
    if ipd_pixels is not None:
        print(f"measured IPD  : {ipd_pixels:.1f} px at {width}px wide")
    print(f"focal_length  : {profile['focal_length']}  (at reference width 640)")
    off = profile["camera_offset"]
    print(f"camera_offset : [{off[0]}, {off[1]}, {off[2]}] cm from screen centre")


def cmd_calibrate(args) -> int:
    name, profile, ipd_px = solve_profile(
        camera=args.camera, distance_cm=args.distance, ipd_cm=args.ipd,
        width=args.width, height=args.height, samples=args.samples,
        measure=not args.no_measure, focal_fallback=args.focal,
    )
    print(f"camera        : {name}")
    if args.no_measure:
        print("measurement   : skipped, keeping default focal length")
    _print_profile(name, profile, ipd_px, args.width)

    SettingsDB(args.db).put(name, profile)
    print(f"\nsaved profile {name!r} to {args.db}")
    print("apply it with:  gazekit apply")
    return 0


def cmd_apply(args) -> int:
    if not app.SETTINGS_JSON.exists():
        print(f"Settings file not found: {app.SETTINGS_JSON}", file=sys.stderr)
        print("Launch GazeAt once so it creates the file.", file=sys.stderr)
        return 1

    db = SettingsDB(args.db)
    matches = db.find(args.camera)
    if not matches:
        print(f"No profile matching {args.camera!r}. Have: {db.names() or '(none)'}",
              file=sys.stderr)
        print("Run: gazekit calibrate --no-measure", file=sys.stderr)
        return 1
    profile = db.get(matches[0])
    assert profile is not None

    if app.app_is_running():
        print("WARNING: GazeAt is running. It rewrites the settings file from its")
        print("         own UI state on quit, so this change will be lost. Quit it")
        print("         first:  osascript -e 'tell application \"GazeAt\" to quit'")
        if not args.force:
            print("\nRefusing to write. Pass --force to do it anyway.", file=sys.stderr)
            return 1

    offset = list(profile["camera_offset"])
    offset[1] = round(offset[1] * args.strength, 2)

    if not args.no_reap:
        reaped = app.reap_backends(keep_newest=args.keep_newest)
        if reaped:
            print(f"reaped stray backends: {reaped}")

    backup = app.write_settings({
        "focal_length": profile["focal_length"],
        "camera_offset": offset,
        "gaze_enabled": True,
    })

    # The backend reads geometry from its own database, not from the JSON.
    if app.BACKEND_DB.exists():
        SettingsDB(app.BACKEND_DB).put("camera_default", {
            "focal_length": profile["focal_length"],
            "ipd": profile.get("ipd", 6.3),
            "camera_offset": offset,
        })
        print(f"updated backend database: {app.BACKEND_DB.name}")

    print(f"profile       : {matches[0]}   strength {args.strength}x")
    print(f"camera_offset : {offset}")
    if backup:
        print(f"backup        : {backup.name}")
    print("\nStart GazeAt, then check with:  gazekit verify")
    return 0


def cmd_verify(args) -> int:
    for path in (app.INPUT_MMAP, app.OUTPUT_MMAP):
        if not path.exists():
            print(f"Missing {path}.", file=sys.stderr)
            print("Is GazeAt running with a camera selected?", file=sys.stderr)
            return 2

    result = verify(
        spec=FrameSpec(args.width, args.height, args.channels),
        tries=args.best_of, threshold=args.threshold, strong=args.strong,
    )

    if not result.live:
        print("NOTE: frames never changed while sampling, so these are the last")
        print("      frames of a previous session, not live video. Start GazeAt")
        print("      and select a camera for a live reading.\n")

    print(f"differing pixels : {result.changed_pixels:,} of {result.total_pixels:,} "
          f"({100 * result.changed_fraction:.2f}%)")
    print(f"max difference   : {result.max_difference}")
    print(f"strong diffs     : {result.strong_pixels:,} (> {args.strong})")
    if result.region:
        x0, x1, y0, y1 = result.region
        print(f"warp region      : x {x0}-{x1}, y {y0}-{y1} "
              f"({x1 - x0}x{y1 - y0} px, {100 * result.coverage:.1f}% of frame)")

    if result.active:
        print(f"\nVERDICT: correction IS being applied ({result.reason}).")
    else:
        print(f"\nVERDICT: correction does NOT look active ({result.reason}).")

    if args.save and result.region:
        save_comparison(result, args.save)
        print(f"wrote {args.save}")

    return 0 if result.active else 1


def cmd_reap(args) -> int:
    reaped = app.reap_backends(keep_newest=args.keep_newest)
    if reaped:
        print(f"reaped {len(reaped)} backend(s): {reaped}")
    else:
        print("nothing to reap.")
    return 0


def cmd_status(args) -> int:
    pids = app.backend_pids()
    print(f"GazeAt running : {app.app_is_running()}")
    print(f"backends       : {len(pids)} {pids if pids else ''}")
    if len(pids) > 1:
        print("  WARNING: multiple backends share one output mmap and corrupt")
        print("           each other's frames. Run: gazekit reap --keep-newest")
    if app.BACKEND_DB.exists():
        cfg = SettingsDB(app.BACKEND_DB).get("camera_default")
        print(f"backend geometry: {json.dumps(cfg) if cfg else '(none)'}")
    if app.SETTINGS_JSON.exists():
        print(f"app settings    : {app.SETTINGS_JSON.read_text()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gazekit",
        description="Calibrate and verify gaze correction on macOS.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("calibrate", help="solve camera geometry and save a profile")
    c.add_argument("--camera", default="MacBook")
    c.add_argument("--distance", type=float, default=55.0,
                   help="measured eye-to-screen distance in cm")
    c.add_argument("--ipd", type=float, default=6.3,
                   help="your inter-pupillary distance in cm")
    c.add_argument("--width", type=int, default=1920)
    c.add_argument("--height", type=int, default=1080)
    c.add_argument("--samples", type=int, default=15)
    c.add_argument("--no-measure", action="store_true",
                   help="skip the camera; offset from display, default focal")
    c.add_argument("--focal", type=float, default=650.0)
    c.add_argument("--db", type=Path, default=DEFAULT_DB)
    c.set_defaults(func=cmd_calibrate)

    a = sub.add_parser("apply", help="push a profile into GazeAt")
    a.add_argument("--camera", default="MacBook")
    a.add_argument("--strength", type=float, default=1.0,
                   help="scale the correction; 1.0 is geometrically correct")
    a.add_argument("--db", type=Path, default=DEFAULT_DB)
    a.add_argument("--no-reap", action="store_true")
    a.add_argument("--keep-newest", action="store_true")
    a.add_argument("--force", action="store_true",
                   help="write even though GazeAt is running and will overwrite")
    a.set_defaults(func=cmd_apply)

    v = sub.add_parser("verify", help="check whether correction is really running")
    v.add_argument("--width", type=int, default=1920)
    v.add_argument("--height", type=int, default=1080)
    v.add_argument("--channels", type=int, default=4)
    v.add_argument("--threshold", type=int, default=8)
    v.add_argument("--strong", type=int, default=60)
    v.add_argument("--best-of", type=int, default=40)
    v.add_argument("--save", type=Path, help="write a before/after PNG")
    v.set_defaults(func=cmd_verify)

    r = sub.add_parser("reap", help="kill leaked backend processes")
    r.add_argument("--keep-newest", action="store_true",
                   help="spare the live session's backend")
    r.set_defaults(func=cmd_reap)

    s = sub.add_parser("status", help="show app, backend and geometry state")
    s.set_defaults(func=cmd_status)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
