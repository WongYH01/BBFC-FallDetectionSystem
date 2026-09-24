# demo-cam v2 — ONNX pose + calibrated fall detection

A Flask app on the MediaMTX RTSP feed: live MJPEG with skeletons, manual
recording, automatic pre-buffered fall clips, and a measurement mode. v2 keeps
v1's camera plumbing and replaces the fall decision.

| | v1 | v2 |
|---|---|---|
| pose | `yolo26n-pose.pt` (torch) | `yolo26n-pose.onnx` (ONNX Runtime), `.pt` fallback |
| decision | box width/height rule | transformer + room-fitted head, rolling mean, latch, descent gate |
| box ratio | the alarm | debug overlay only (`FALL_DEBUG=1`) |

## How the alarm is decided

`fallcore.stream.EnsembleStream` owns the arithmetic:

1. The **largest person's** keypoints fill a 4 s ring buffer (60 rows).
2. Once a second that window is scored by `augnone_ms_coords_hn_s99.pt` and its
   pooled embedding read by a logistic head fitted on **this room's** clips
   (`probe_augnone_ms_coords_hn_s99.npz`, `scripts/fit_calibration_head.py`).
3. A rolling mean over the last `BUFFER_LEN=4` windows latches the alarm above
   `FALL_THRESHOLD`, and clears after `FALL_CLEAR_WINDOWS` below
   `FALL_CLEAR_BELOW`.
4. A window may only **raise** the alarm if the hips descended at least
   `FALL_MIN_DESCENT` body lengths per second somewhere inside it. Latching and
   clearing are untouched, so a fall alarms on its descent and stays alarmed.

Step 4 exists because a motionless body carries no information about how it got
there: without it the verdict is re-made every second on windows of someone
simply lying still, and the head settles that tie on height in frame — which is
furniture, and is read differently the moment the camera moves.

Measured on two camera positions in the same room, 60 and 52 clips. The first is
leave-one-subject-out on the position the head was fitted at; the second is that
head applied as-is after the camera was moved to another wall and a lower angle:

| | clip F1 | FP | FN |
|---|---|---|---|
| fitted position, with the gate | **0.967** | 1 | 1 |
| fitted position, no gate | 0.967 | 1 | 1 |
| camera re-aimed, with the gate | **0.909** | 2 | 3 |
| camera re-aimed, no gate | 0.877 | 4 | 3 |
| camera re-aimed, no head at all | 0.724 | 9 | 7 |

So the gate is free where the head was fitted and halves the false alarms where
it was not, and the head itself carries most of its value across the move — the
last row is what you get without one. The four-checkpoint ensemble v2 used to
ship scored 0.853 at the fitted position, and the best single checkpoint 0.836 —
at four forward passes per window instead of one.

Transfer to a *different room* is untested: every clip here comes from one.

## Deploy on a new machine

The venv is uv-managed (no pip). From the repo root:

```
uv venv demo-cam/.venv
uv pip install --python demo-cam/.venv/Scripts/python.exe -r demo-cam/requirements.txt
uv pip install --python demo-cam/.venv/Scripts/python.exe -e .
python scripts/fetch_weights.py              # ONNX pose + the calibrated pair
copy demo-cam\.env.example demo-cam\.env     # fill in MediaMTX host/credentials
cd demo-cam && .venv\Scripts\python.exe v2\app.py
```

`app.py` reads `.env` from the working directory, so run it from `demo-cam/`; the
page is on port 5005. The pose pane drives inference — if nobody is watching the
MJPEG stream, no frames are scored and the UI reports "No feed".

Offline check before pointing it at the camera, same decision code as the live
path:

```
.venv\Scripts\python.exe v2\replay.py path\to\clip.mp4
```

## Settings

Defaults shown; full list with commentary in `demo-cam/.env.example`.

| Var | Default | Meaning |
|---|---|---|
| `POSE_SCALE` | `yolo26n` | backbone size; selects `{scale}-pose.onnx`. **Do not change** — the head was fitted on yolo26n at 640 |
| `POSE_IMGSZ` | `640` | backbone input size; 480/320 are cheaper but lose falls |
| `POSE_THREADS` | `2` | ONNX Runtime pool. ORT's default spins on every core: 8.4 ms wall / 70 ms CPU per frame, against 16.4 / 33 at two threads |
| `CALIBRATION_HEAD` | `runs/checkpoints/probe_…_s99.npz` | the room head; empty falls back to the checkpoint's own classifier. A head whose backbone or feature variant disagrees with its checkpoint is refused |
| `ENSEMBLE_CKPTS` | `runs/checkpoints/augnone_ms_coords_hn_s99.pt` | one file = the calibrated model; several = their mean |
| `BUFFER_LEN` / `FALL_THRESHOLD` | `4` / `0.5` | rolling mean length and the level that engages the alarm |
| `FALL_MIN_DESCENT` | `0.10` | the descent gate, body lengths/s; `0` restores pre-gate behaviour |
| `FALL_CLEAR_BELOW` / `FALL_CLEAR_WINDOWS` | `0.2` / `2` | hysteresis for clearing |
| `VID_STRIDE` / `STREAM_FRAME_STRIDE` | `1` / `2` | decode-and-score one frame in N, then rows per window sample. Their product must keep rows ~67 ms apart (see below) |
| `STREAM_STALL_SEC` / `STREAM_CONNECT_SEC` | `8` / `24` | force a reconnect when frames stop or never start |
| `IDLE_STOP_SEC` | `30` | drop the RTSP connection this long after the last viewer (ignored while clips or a recording run) |
| `FALL_DEBUG` | `0` | draw each box's w/h ratio (the v1 rule, debug only) |

**Row spacing is the setting that matters.** Training rows are 2 frames apart at
30 fps ≈ 67 ms, and a 60-row window is 4 s. Match it:

| camera | `VID_STRIDE` | `STREAM_FRAME_STRIDE` |
|---|---|---|
| 30 fps | 2 | 1 (halves the pose cost; measured bit-identical to full rate) |
| 30 fps | 1 | 2 |
| 15 fps | 1 | 1 |

Get it wrong and the window's time span changes, not just its cost: 15 fps with
`VID_STRIDE=2` stretches a window to ~8 s, which the model has never seen.

## What to expect

- The alarm trails the fall by **1–2 s** (two high windows must lift the 4-window
  mean over the threshold), plus the 1 s scoring cadence and a 4 s warm-up at
  stream start.
- Remaining false alarms are controlled descents to the floor — `sit floor`,
  `tie shoelace`. The gate removes the lying-still ones, not these.
- Multi-person scenes are the known weakness: the pipeline watches the largest
  box, as training did, so it can follow a bystander instead of the faller.
- CPU: ~23 ms/frame for pose through `track()` (n-640, 2 ORT threads, 9800X3D);
  the classifier adds ~0.7 ms once a second. One CPU serves one camera at 30 fps
  with `VID_STRIDE=2`.

## One reader, many viewers

`detector.py` runs **one capture+inference thread per process**, started on the
first viewer, publishing the newest annotated JPEG into a single slot that every
viewer subscribes to. Two tabs cost one RTSP session and one pose pass. The
earlier per-request design opened a connection and an inference loop per tab, and
a loop busy with inference is a loop not draining its socket — MediaMTX then
discards frames for a slow reader ("reader is too slow, discarding 469 frames"),
which is the frozen picture.

`/stream/status` reports what that one reader is doing: `fps`, `viewers`,
`reconnects`, `age_sec`, `last_error` and the tuning in force. Climbing
`reconnects` with nobody opening tabs means the Pi is dropping the path; check
`journalctl -u mediamtx.service` there, and see `../pi/` for the MediaMTX config
and the watchdog that restarts it when the hardware encoder wedges.

## Files

```
app.py            Flask routes: /, /video_feed, /stream/status, /record/*,
                  /fall/status, /fall_clips/status; reloader off by design
detector.py       RTSP -> ONNX pose -> calibrated model -> MJPEG / status / recorder
metrics.py        measurement mode, model columns and provenance
fall_recorder.py  pre-buffered fall-clip capture (unchanged from v1)
replay.py         offline replay of one video through the same decision stack
METRICS.md        what the measurement numbers do and do not mean
../pi/            Pi-side MediaMTX config + encoder watchdog (deploy there)
```
