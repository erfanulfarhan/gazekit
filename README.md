# gazekit

Calibrate and **verify** gaze correction on macOS.

Gaze-correction cameras warp your eyes so you appear to look at the lens
during video calls. When one stops working, it does so silently: the logs look
identical whether the warp is running or doing nothing at all. `gazekit`
answers the question the logs cannot, by measuring the video pipeline directly.

```console
$ gazekit verify
differing pixels : 11,002 of 2,073,600 (0.53%)
strong diffs     : 989 (> 60)
warp region      : x 557-978, y 398-514 (421x116 px, 2.4% of frame)

VERDICT: correction IS being applied (localised eye-region warp).
```

## What it does

| Command | Purpose |
| --- | --- |
| `gazekit verify` | Prove whether correction is actually running |
| `gazekit calibrate` | Solve camera geometry from a measurement |
| `gazekit apply` | Push a solved profile into the app |
| `gazekit status` | App, backend and geometry state at a glance |
| `gazekit reap` | Kill leaked backend processes |

## How verification works

The backend's input and output frames are memory-mapped files on disk, so they
can be compared without touching a camera or asking for permission. A working
corrector copies the frame and replaces **only** the eye regions:

- **active** means a few hundred large differences inside one eye-sized box
- **inactive** means identical frames, or differences smeared everywhere

Three things make this harder than a naive diff, and each one produced a false
result before it was handled:

1. **The two mmaps are not synchronised.** A single read compares frame N
   against corrected frame M, and any head movement swamps the warp. `gazekit`
   samples many pairs and keeps the most aligned one.
2. **Alignment cannot be scored by mean or median difference.** Both are
   dominated by sensor noise, which says nothing about whether two frames show
   the same instant. Counting differences too large to be noise measures
   misalignment directly. On identical input this changed the selected pair
   from 70,751 strong differences to 221.
3. **A quit app leaves both files frozen**, which otherwise reads as a
   perfectly valid measurement of a stale snapshot.

## Calibration, solved rather than nudged

Most gaze-correction calibration is arrow-key nudging with no ground truth,
which is how a camera offset ends up 60 cm horizontally and the network is
asked for a 45 degree sideways redirect. `gazekit` solves the numbers instead:

- **focal length** from observed iris separation at a measured distance,
  `f = ipd_pixels * distance / ipd_cm`
- **camera offset** from the display's real physical dimensions

```console
$ gazekit calibrate --distance 55      # measured eye-to-screen distance
$ gazekit apply                        # geometrically correct
$ gazekit apply --strength 1.8         # exaggerated, easier to see
```

`--distance` is the one number that cannot be inferred, and its accuracy sets
everything else. If your terminal lacks camera permission, `--no-measure`
still derives the camera offset from display geometry, which in practice is
the error that matters most.

### Why geometry matters this much

The redirection angle is `atan(camera_offset_y / viewing_distance)`. For a
13.6 inch laptop at 60 cm that is about **9.5 degrees**. Feed the same code a
default meant for a much larger screen (`-21` cm) and it asks for 19 degrees;
forget to scale focal length from a 640px calibration to a 1920px frame and it
compounds to 42. Warping networks are trained over roughly plus or minus 15 to
25 degrees, and outside that range the spatial transformer degenerates and the
correction silently vanishes.

`gazekit` therefore clamps the angle. Too little correction is visible and
debuggable; none at all is indistinguishable from the feature being broken.

## Install

```bash
git clone https://github.com/erfanulfarhan/gazekit
cd gazekit
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[camera,dev]"
.venv/bin/python -m pytest -q
```

The `camera` extra (OpenCV, MediaPipe, AVFoundation bindings) is needed only
for `calibrate` and for `verify --save`. Plain `verify` needs just numpy.

> **Avoid putting the virtualenv in an iCloud-synced folder** such as a synced
> Desktop or Documents. Sync can duplicate and rename files underneath it: a
> `.pth` finder module renamed to `finder 2.py` silently breaks the install,
> and the import failure that follows points nowhere near the real cause.

## Notes on the GazeAt app

`gazekit` currently targets [GazeAt](https://github.com/WangWilly/gaze-correction-cam),
reading and writing the files it leaves in its container. Three behaviours are
worth knowing, since none are documented and each cost real debugging time:

- **The backend reads geometry from SQLite, not from the settings JSON.** The
  Swift UI reads the JSON; the Python backend reads `gaze_user_settings.db`.
  Writing only the JSON changes less than you expect. `gazekit apply` writes
  both.
- **The app rewrites the JSON from its UI state on quit**, so edits made while
  it runs are lost. `gazekit apply` refuses to write while it is running.
- **It leaks a backend process on every quit.** Five were once observed at
  once, each burning most of a core, all writing to the same output mmap so
  stale writers corrupt corrected frames with uncorrected ones. This alone
  reproduces "correction stopped working".

Also: if the virtual camera never appears in Zoom, the camera system extension
is probably awaiting approval. Check with `systemextensionsctl list` and look
for `activated waiting for user` rather than `activated enabled`.

## Credits

`gazekit` is my own tooling, and it is not a gaze-correction model. It stands
on work by others:

- **Chih-Fan Hsu, Yu-Shuen Wang, Chin-Laung Lei, Kuan-Ta Chen.** *Look at Me!
  Correcting Eye Gaze in Live Video Communication.* ACM TOMM 15(2), 2019.
  [doi:10.1145/3311784](https://doi.org/10.1145/3311784) — the warping network
  and the geometric model this calibration follows.
- **[WangWilly/gaze-correction-cam](https://github.com/WangWilly/gaze-correction-cam)**
  — the macOS app that `gazekit` calibrates and verifies.

Those projects carry their own licences and are not redistributed here.

## Licence

MIT, see [LICENSE](LICENSE).
