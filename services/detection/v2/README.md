# v2 — ONNX pose + calibrated fall detection

A Flask app on the MediaMTX RTSP feed: live MJPEG with skeletons, manual
recording, automatic pre-buffered fall clips, and a measurement mode. v2 keeps
v1's camera plumbing and replaces the fall decision.

| | v1 | v2 |
|---|---|---|
| pose | `yolo26n-pose.pt` (torch) | `yolo26m-pose-384x640-int8.onnx` (ONNX Runtime), yolo26n as the backup |
| decision | box width/height rule | transformer + room-fitted head, rolling mean, latch, descent gate |
| box ratio | the alarm | debug overlay only (`FALL_DEBUG=1`) |

## What is recorded

- The **automatic fall clip** (`v2/fall_clips/fall_<ts>.mp4`) is **skeleton
  only**: the pose drawn on a blank canvas, no camera pixels. It is the file
  the detection service uploads to the notification service.
- Beside it, the **keypoint file** (`v2/fall_clips/fall_<ts>.npz`): the same
  frames as numbers -- every tracked person's keypoints, and the alarm and
  score -- which the detection service uploads after the clip as training data
  (format: `fallcore/keypoint_file.py`; `FALL_KEYPOINTS_ENABLED=0` stops it).
- The **manual Record button** is the one path that writes raw camera footage
  (`v2/videos/pose_<ts>.mp4` is a side-by-side of the camera image and the
  annotated frame). Set `ALLOW_MANUAL_RECORDING=0` to refuse it with a 403;
  the detection service ships with it off.

## How the alarm is decided

`fallcore.stream.EnsembleStream` owns the arithmetic, and
`fallcore.stream.SubjectStreams` runs one of them per tracked person:

1. **Each tracked person's** keypoints fill their own 4 s ring buffer (60 rows).
2. Once a second that window is scored by `augnone_ms_coords_hn_s99.pt` and its
   pooled embedding read by a logistic head fitted on **this room's** clips, as
   seen by the pose backbone that is running
   (`probe_augnone_ms_coords_hn_s99_yolo26m-int8.npz` for yolo26m,
   `probe_augnone_ms_coords_hn_s99.npz` for yolo26n;
   `scripts/fit_calibration_head.py`).
3. A rolling mean over the last `BUFFER_LEN=4` windows latches the alarm above
   `FALL_THRESHOLD`, and clears after `FALL_CLEAR_WINDOWS` below
   `FALL_CLEAR_BELOW`.
4. A window may only **raise** the alarm if the hips descended at least
   `FALL_MIN_DESCENT` body lengths per second somewhere inside it. Latching and
   clearing are untouched, so a fall alarms on its descent and stays alarmed.

5. The room's alarm is up while **anyone's** is, so one fall is one alarm.

Step 4 exists because a motionless body carries no information about how it got
there: without it the verdict is re-made every second on windows of someone
simply lying still, and the head settles that tie on height in frame — which is
furniture, and is read differently the moment the camera moves. With every
person decided, it is also what keeps someone lying still on a bench, who
scores ~0.97 on every window, from raising the alarm.

**Why per person.** Following the largest box alone follows whoever is closest
to the camera, so a fall behind a seated or lying bystander is decided on the
bystander's keypoints. On the four picam-2 recordings with a second person in
shot, replayed through this script's decision, the largest box catches 1 fall
and per-person decisions catch all 4.

**A person is not a tracker id.** ByteTrack gives the faller a new id in 34 of
82 picam falls, usually 0.5–1 s into the descent, when the box turns from tall
to wide. If a new id started a new stream, neither stream would hold the whole
fall, and that loses 20 of 75 falls. So a lost person's stream keeps running on
"nobody seen" rows for one window, and a new id that appears within
`FALL_RECONNECT_SEC` and `FALL_RECONNECT_DIST` of a just-lost person continues
that person's stream. With both, the single-person picam sets catch 75 of 82
falls, the same as the largest box, with 9 false alarms in 82 everyday clips
against its 7. Those 2 extra are lost streams running on empty rows, which the
model partly reads as a fall. `FALL_SUBJECTS=largest` goes back to the largest
box.

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

## Pose backbone

yolo26m runs, as a static INT8 ONNX export at 384x640 (1280x720 frames
letterbox to 384x640, so a square 640 input is 44% padding). Its weights are
7-bit (`reduce_range`), so a CPU without VNNI cannot saturate them and scores
the same as one with it. yolo26n is the backup, in the same image with the
head fitted on its own keypoints (`fallcore.pose_backbone`).

Both heads are fitted on picam-1 only. On the two sessions neither saw --
picam-2 (camera re-aimed) and picam-3 (another room, 15 fps) -- under the live
rule (buffer 2, clear after 1, descent gate):

| | picam-2 TP/FP/FN | picam-3 TP/FP/FN | falls caught | false alarms |
|---|---|---|---|---|
| **yolo26m INT8** + its head | 24/1/0 | 26/3/2 | 50 / 52 | 4 / 52 |
| yolo26n + its head | 24/2/0 | 26/5/2 | 50 / 52 | 7 / 52 |

The difference is within the ±2-3 clips these sets resolve. The yolo26n head on
yolo26m keypoints is worse than either (picam-3 24/3/4), which is why the head
follows the backbone. The m head is fitted with segment labels
(`fit_calibration_head.py --labels segment`): a fall clip's windows before the
fall count as No-Fall rather than Fall.

Speed, whole loop (track, draw, skeleton, JPEG) on a Ryzen 7 9800X3D: yolo26m
INT8 18.2 fps on 2 ORT threads and 24.8 on 4, against ~43 for yolo26n on 2.
A 15 fps camera needs ~20 with the 1.3x margin, hence `POSE_THREADS=4`. No other
CPU has been timed: run `scripts/bench_pose.py` on the target first, and set
`POSE_SCALE=yolo26n` where yolo26m does not keep up.

## Deploy on a new machine

The venv is uv-managed (no pip) and shared with demo-cam v1. From the repo
root:

```
uv venv demo-cam/.venv
uv pip install --python demo-cam/.venv/Scripts/python.exe -r demo-cam/requirements.txt
uv pip install --python demo-cam/.venv/Scripts/python.exe -e services/detection
cd services/detection
copy .env.example .env                       # MediaMTX host/credentials, FALL_WEIGHTS_TOKEN
..\..\demo-cam\.venv\Scripts\python.exe scripts\fetch_weights.py
..\..\demo-cam\.venv\Scripts\python.exe v2\app.py
```

`app.py` finds `services/detection/.env` by searching up from `v2/`; the page is
on port 5005. The pose pane drives inference — if nobody is watching the
MJPEG stream, no frames are scored and the UI reports "No feed".

Offline check before pointing it at the camera. Replay builds the same Fall
Decision as the live path (`fallcore.fall_decision`), reads the same settings
below from the environment, and prints the decision it ran under; a flag such
as `--min-descent 0` or `--threshold 0.6` overrides one setting on top:

```
..\..\demo-cam\.venv\Scripts\python.exe v2\replay.py path\to\clip.mp4
```

## Settings

Defaults shown; full list with commentary in `services/detection/.env.example`.

| Var | Default | Meaning |
|---|---|---|
| `POSE_SCALE` | `yolo26m` | pose backbone and the head fitted on its keypoints: `yolo26m` (INT8, 384x640) or `yolo26n` (640). yolo26n also takes over by itself when the yolo26m files are missing or it will not load |
| `POSE_MODEL` | — | replaces the backbone's file (a `.pt` runs on torch); the head still follows `POSE_SCALE`, and nothing falls back |
| `POSE_IMGSZ` | the backbone's | input size, `640` or `384,640`; an ONNX export only runs at the size it was exported at |
| `POSE_THREADS` | `4` | ONNX Runtime pool. ORT's default spins on every core: 8.4 ms wall / 70 ms CPU per frame, against 16.4 / 33 at two threads. Four lets a yolo26m INT8 model hold a 15 fps camera (whole loop 24.8 fps, against 18.2 on two) |
| `CALIBRATION_HEAD` | the pose backbone's | the room head; empty falls back to the checkpoint's own classifier, and setting it turns the yolo26n fallback off. A head whose backbone or feature variant disagrees with its checkpoint is refused |
| `ENSEMBLE_CKPTS` | `runs/checkpoints/augnone_ms_coords_hn_s99.pt` | one file = the calibrated model; several = their mean |
| `BUFFER_LEN` / `FALL_THRESHOLD` | `4` / `0.5` | rolling mean length and the level that engages the alarm |
| `FALL_MIN_DESCENT` | `0.10` | the descent gate, body lengths/s; `0` restores pre-gate behaviour |
| `FALL_CLEAR_BELOW` / `FALL_CLEAR_WINDOWS` | `0.2` / `2` | hysteresis for clearing |
| `FALL_SUBJECTS` | `tracked` | `tracked` decides every tracked person separately; `largest` the largest box alone |
| `FALL_RECONNECT_SEC` / `FALL_RECONNECT_DIST` | `2.0` / `0.35` | a new tracker id continues a person lost at most this long ago and this far from its box centre (fractions of the frame); `0` never joins |
| `POSE_TRACKER` | `bytetrack.yaml` | the Ultralytics tracker that supplies the ids |
| `ALLOW_MANUAL_RECORDING` | `1` | `0` refuses the raw-footage Record button (403); automatic clips stay skeleton-only |
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
- CPU: the yolo26m INT8 loop runs 24.8 fps on 4 ORT threads (9800X3D), enough
  for a 15 fps camera; yolo26n's ~23 ms/frame on 2 threads serves 30 fps with
  `VID_STRIDE=2`. The classifier adds ~0.7 ms once a second.

## Switching pose models

Time the candidates offline, on the machine that will run the detector, before
changing `POSE_SCALE` or `POSE_MODEL`:

```
python scripts/bench_pose.py                  # every .pt/.onnx in models/
python scripts/bench_pose.py --camera-fps 15 --imgsz 480 --threads 4
```

It reads `POSE_IMGSZ` / `POSE_THREADS` / `POSE_DEVICE` from the environment,
then `services/detection/.env`,
and prints ms per frame, fps, the `VID_STRIDE` / `STREAM_FRAME_STRIDE` pair that
keeps rows ~67 ms apart for that camera, and a verdict: `fits` (≥1.3× the rate
those strides demand, leaving room for tracking, drawing and encoding),
`marginal` or `too slow`. It runs without the camera. After the switch, the
`fps` field of `/stream/status` is the live check.

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
fall_recorder.py  pre-buffered skeleton-only fall clips and their keypoint files;
                  publishes clip_started / clip_ready events (pop_events) for
                  the detection service, and only ever names a file
                  `fall_<ts>.mp4` / `fall_<ts>.npz` once it is complete
replay.py         offline replay of one video through the same decision stack
../pi/            Pi-side MediaMTX config + encoder watchdog (deploy there)
```
