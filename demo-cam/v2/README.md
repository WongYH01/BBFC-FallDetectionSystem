# demo-cam v2 — ONNX pose + rolling calibrated-model fall detection

v2 keeps v1's camera plumbing (MediaMTX RTSP, ByteTrack, MJPEG, manual
recording, auto fall-clip capture, metrics mode) and replaces the fall decision:

| | v1 | v2 |
|---|---|---|
| pose backend | `yolo26n-pose.pt` (torch) | `yolo26n-pose.onnx` (ONNX Runtime), `.pt` fallback |
| fall decision | box width/height rule + frame latch | calibrated transformer: one backbone + a head fitted on this room, rolling mean of 4 windows, latched |
| extra signal | — | per-window `p_fall`, rolling mean, per-model probabilities |
| box ratio | the alarm | debug overlay only (`FALL_DEBUG=1`) |

The decision arithmetic lives in `fallcore.stream` (`EnsembleStream`): a 4 s
keypoint ring buffer per largest person, one window scored per second by
`augnone_ms_coords_hn_s99.pt` and read by a logistic head fitted on this room's
own clips (`probe_augnone_ms_coords_hn_s99.npz`, `scripts/fit_calibration_head.py`),
then a rolling mean over the last `BUFFER_LEN=4` windows with a hysteresis latch.

Held out on the camera's five subjects (fit on four, score the fifth): window AUC
0.992 and rolling F1 **0.967** (1 false alarm, 1 miss), against 0.836 for the
best single checkpoint and 0.853 for the four-checkpoint ensemble v2 used to
ship — at one forward pass per window instead of four. `replay.py` below is the
offline twin of the live path; keep the two model lists in step.

The head is a **commissioning artefact**: it needs that room's own falls and
ADLs, so a new room needs one fitted there. `CALIBRATION_HEAD=` (empty) falls
back to the checkpoint's own classifier, and `ENSEMBLE_CKPTS` taking several
files restores the four-model mean.
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

`fetch_weights.py` pulls the five weight files from
`junyuu/fall-detection-bbfc` — pinned to a revision, each verified against its
SHA-256 — and writes them to the paths the detector resolves by default:
`demo-cam/models/yolo26n-pose.onnx` and `runs/checkpoints/*.pt`.

The Hub repo is private and the fetch script authenticates **only** through
`demo-cam/.env`. Add a read token from
https://huggingface.co/settings/tokens to the file you already created from
`.env.example`:

```
# demo-cam/.env
FALL_WEIGHTS_TOKEN=hf_xxx
```

There is no CLI flag, process-environment or stored-login fallback — one place
to put the secret. `--check` verifies the local files without a token;
`--force` re-downloads.

Regenerating instead of fetching: notebook 18 exports the ONNX and notebooks
02/10/14 train the checkpoints. Weights under `models/` are gitignored.

## Run

```
cd demo-cam
.venv\Scripts\python.exe v2\app.py
```

`app.py` loads `.env` from the working directory (as v1 did), so run it from
`demo-cam/`; the page is on port 5005. The pose pane drives inference — if the
MJPEG stream is not being watched, no frames are scored and the UI reports
"No feed" rather than a stale state.

### Offline check before the camera

```
.venv\Scripts\python.exe v2\replay.py path\to\clip.mp4
```

prints one line per scored window and a summary (`alarm`, `first_alarm_s`).
Runs the same decision code as the live path.

## Env vars (see `demo-cam/.env.example`)

| Var | Default | Meaning |
|---|---|---|
| `POSE_SCALE` | `yolo26m` | backbone size (n/s/m/…); selects `{scale}-pose.onnx` |
| `POSE_MODEL` | first existing `{POSE_SCALE}-pose.onnx` in `demo-cam/models/` or `runs/onnx/` | explicit pose file, overrides `POSE_SCALE`; `.pt` switches back to torch |
| `POSE_IMGSZ` | `640` | backbone input size |
| `STREAM_FRAME_STRIDE` | `2` | rows between window samples; 2 matches ~30 fps training data, use 1 when the pose rate is ~15 fps |
| `VID_STRIDE` | `1` | decode-and-score one frame in every N; the socket is still drained at full rate, so this lowers CPU without making us a slow reader. 2 pairs with `STREAM_FRAME_STRIDE=1` — measured to reproduce the full-rate rows bit-for-bit |
| `POSE_THREADS` | `2` | ONNX Runtime thread pool for the pose graph; 0 leaves ORT's default (all cores, spinning between operators). Measured at 640: default 8.4 ms wall / 70 ms CPU per frame vs 16.4 / 33 at two threads |
| `CALIBRATION_HEAD` | `runs/checkpoints/probe_augnone_ms_coords_hn_s99.npz` | logistic head fitted on this room's clips; empty = the checkpoint's own classifier. A head whose backbone or feature variant does not match its checkpoint is refused |
| `POSE_DEVICE` | *(empty)* | device for the pose model; empty picks from the model (ONNX → CPU). Setting `cuda` needs an `onnxruntime-gpu` build — a CUDA request against a CPU session dies on the first frame |
| `JPEG_QUALITY` | `75` | MJPEG quality for the browser feed (OpenCV's own default is 95) |
| `STREAM_STALL_SEC` | `8` | no decoded frame for this long → the reader force-reconnects; the guard against the Pi's encoder going silent |
| `STREAM_CONNECT_SEC` | `24` | the same guard for a connection that never delivers a first frame |
| `IDLE_STOP_SEC` | `30` | seconds after the last viewer leaves before the RTSP connection is dropped (ignored while fall clips or a recording are running) |
| `FLASK_DEBUG` | `0` | `1` turns on the Flask debugger; the auto-reloader stays off either way — it loads the models twice and can leave two RTSP readers on the Pi |
| `ENSEMBLE_CKPTS` | the four `runs/checkpoints/*.pt` above | `os.pathsep`-separated checkpoints |
| `BUFFER_LEN` | `4` | rolling-mean window count |
| `FALL_THRESHOLD` | `0.5` | rolling mean that engages the alarm |
| `FALL_CLEAR_BELOW` / `FALL_CLEAR_WINDOWS` | `0.2` / `2` | hysteresis: consecutive updates below this clear the alarm |
| `FALL_DEBUG` | `0` | draw each box's w/h ratio (v1 rule, debug only) |
| `FALL_CLIPS_ENABLED`, `RECORD_FPS`, `SKELETON_BG`, `METRICS_*` | as v1 | recording, clip capture, metrics |

**Pose scale:** v2 defaults to **yolo26n**, the scale the keypoint caches were
extracted at and the one the fitted head was trained against — at another scale
the head reads an input distribution it never saw (the pipeline runs, the
probabilities are off-distribution). `POSE_IMGSZ=640` is the exported size.
Measured on the 9800X3D (`runs/metrics/pose_onnx_*.csv`): n at 640 is ~23 ms per
frame through `track()` against ~83 ms for m; ONNX at 480/320 is cheaper still
but loses falls (2 and 5 clips of the 60), and dynamically quantized INT8 is
*slower* than fp32 here, so do not reach for it.

**yolo26m on CPU is ~73 ms/frame (14 FPS)**, against ~10 ms/frame for n (raw
ONNX Runtime, 640). At 14 FPS the loader is effectively decimating the stream,
so each buffered row is ~73 ms apart — almost exactly training's 66.7 ms — and
`STREAM_FRAME_STRIDE=1` is the correct pairing: a 60-row window then still
covers about 4 s of wall clock. With the default stride 2 the window would
stretch to ~9 s and no longer match the trained time scale. Even so, one CPU
serves one camera at that rate; for 30 fps live use the ONNX Runtime GPU
provider (`onnxruntime-gpu`) or scale n.

## What to expect (measured on the 60-clip picam set, notebook 17)

- The calibrated model at buffer 4 scores **rolling F1 0.967** held out on the
  camera's five subjects (1 ADL false alarm, 1 fall missed). The four-checkpoint
  ensemble it replaced scored 0.853 pooled on the same clips, `coords + hard neg`
  alone 0.836.
- The alarm trails the fall by roughly **1–2 s** (two high windows must lift the
  4-window mean over 0.5), plus the 1 s scoring cadence and the 4 s warm-up at
  stream start.
- Recurring false alarms are `lie down`, `sit floor` and `tie shoelace`: the
  zero-shot model still reads controlled descent / lying as falling. This is the
  domain-adaptation problem (Tier 1), not a wiring bug.
- The classifier watches the **largest person only**, as training did.
- CPU: the ONNX pose path measured ~23 ms/frame through `track()` (n-640, 2 GB
  ONNX Runtime threads) on the 9800X3D; the calibrated model adds ~0.7 ms once a
  second, against ~3.2 ms for the four-checkpoint mean. Set `VID_STRIDE=2` with
  `STREAM_FRAME_STRIDE=1` on a 30 fps camera to halve the pose cost — measured to
  reproduce the full-rate rows bit-for-bit.

## One reader, many viewers

`/video_feed` used to call `gen_frames()` once per HTTP request, and each call
opened its own RTSP connection **and** its own inference loop — so every browser
tab was a second reader on the Pi and a second pose loop here competing for the
same CPU. A loop busy doing inference is a loop not draining its socket, and
MediaMTX never blocks the camera for a slow reader: its queue fills and it drops
the oldest frames ("reader is too slow, discarding 469 frames"), which is the
frozen picture. Closing a tab did not reliably end the session either, so
orphaned readers accumulated across reloads.

`detector.py` now runs **one capture+inference thread per process**, started on
the first viewer and publishing the newest annotated JPEG into a single slot;
every viewer subscribes to that slot. Two tabs cost one RTSP session and one
pose pass. The port came from v1's `fix stream getting frozen issue`
keeping v2's own decisions: ONNX/torch pose via `model.track`, the calibrated
model, the recorder, the metrics tap and the fall-clip buffer.

`/stream/status` reports what that one reader is doing: `fps`, `viewers`,
`reconnects`, `age_sec`, `last_error`, and the tuning currently in force. Watch
`reconnects`: a number that climbs while nobody opens or closes a tab is the Pi
dropping the path (check `journalctl -u mediamtx.service` there). `age_sec`
climbing past `stall_sec` means the watchdog is about to force a reconnect.

The **Pi side** has its own folder: `demo-cam/pi/` — `README.md` for what the
two journal errors mean, `mediamtx.recommended.yml` (writeQueueSize 2048, 720p15
@ 2 Mbps so the hardware encoder is never starved) and a
`mediamtx-watchdog` script + systemd unit that restarts MediaMTX when the
encoder wedges on `ioctl(VIDIOC_QBUF)`, which is the other half of the freeze.

For CPU, set `VID_STRIDE=2` with `STREAM_FRAME_STRIDE=1`: pose every 2nd frame
halves the only expensive stage and reproduces the trained row spacing exactly
(the caches are every 2nd frame of 30 fps), while the socket is still drained at
full rate.

## Weights

`scripts/fetch_weights.py` pulls the three files this version needs — the ONNX
pose export, `runs/checkpoints/augnone_ms_coords_hn_s99.pt` and
`runs/checkpoints/probe_augnone_ms_coords_hn_s99.npz` — from the Hugging Face
release, pinned to the revision their SHA-256s were taken from. The four
single-corpus checkpoints the ensemble used to average were removed from the
release when the calibrated model replaced them; they are still downloadable
from the previous revision, which the fetch script names in its docstring, so
the old ensemble can be rebuilt if it is ever wanted back.

The head is room-specific: it was fitted on this camera's clips, so another room
needs its own (`scripts/fit_calibration_head.py`) rather than a copy of this
one.

## Files

```
v2/app.py          Flask routes: /, /video_feed, /stream/status, /record/*,
                   /fall/status, /fall_clips/status; reloader off by design
v2/detector.py     RTSP -> ONNX pose -> calibrated model -> MJPEG / status /
                   recorder, with the single shared capture thread
v2/metrics.py      measurement mode + ensemble columns and provenance
v2/fall_recorder.py  pre-buffered fall-clip capture (unchanged from v1)
v2/replay.py       offline replay of one video through the same decision stack
v2/METRICS.md      what the measurement numbers do and do not mean
../pi/             Pi-side MediaMTX config + encoder watchdog (deploy there)
```
