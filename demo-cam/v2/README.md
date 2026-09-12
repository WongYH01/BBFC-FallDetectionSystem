# demo-cam v2 — ONNX pose + rolling ensemble fall detection

v2 keeps v1's camera plumbing (MediaMTX RTSP, ByteTrack, MJPEG, manual
recording, auto fall-clip capture, metrics mode) and replaces the fall decision:

| | v1 | v2 |
|---|---|---|
| pose backend | `yolo26n-pose.pt` (torch) | `yolo26n-pose.onnx` (ONNX Runtime), `.pt` fallback |
| fall decision | box width/height rule + frame latch | four-checkpoint transformer ensemble, rolling mean of 4 windows, latched |
| extra signal | — | per-window `p_fall`, rolling mean, per-model probabilities |
| box ratio | the alarm | debug overlay only (`FALL_DEBUG=1`) |

The decision arithmetic lives in `fallcore.stream` (`EnsembleStream`): a 4 s
keypoint ring buffer per largest person, one window scored per second, the mean
of `final_yolo26n`, `hn10_full_hn_s99`, `hn10_coords_hn_s99` and
`omnifall_cs_full`, then a rolling mean over the last `BUFFER_LEN=4` windows
with a hysteresis latch. Notebook 17 measured this rule; `replay.py` below is
the offline twin of the live path.

## Deploy on a new machine

The venv is uv-managed (no pip). From the repo root:

```
uv venv demo-cam/.venv
uv pip install --python demo-cam/.venv/Scripts/python.exe -r demo-cam/requirements.txt
uv pip install --python demo-cam/.venv/Scripts/python.exe -e .
python scripts/fetch_weights.py              # ONNX pose + 4 classifiers
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
| `POSE_MODEL` | first existing of `demo-cam/models/yolo26n-pose.onnx`, `runs/onnx/yolo26n-pose-imgsz640.onnx` | pose weights; `.pt` switches back to torch |
| `POSE_IMGSZ` | `640` | backbone input size |
| `ENSEMBLE_CKPTS` | the four `runs/checkpoints/*.pt` above | `os.pathsep`-separated checkpoints |
| `BUFFER_LEN` | `4` | rolling-mean window count |
| `FALL_THRESHOLD` | `0.5` | rolling mean that engages the alarm |
| `FALL_CLEAR_BELOW` / `FALL_CLEAR_WINDOWS` | `0.2` / `2` | hysteresis: consecutive updates below this clear the alarm |
| `FALL_DEBUG` | `0` | draw each box's w/h ratio (v1 rule, debug only) |
| `FALL_CLIPS_ENABLED`, `RECORD_FPS`, `SKELETON_BG`, `METRICS_*` | as v1 | recording, clip capture, metrics |

## What to expect (measured on the 60-clip picam set, notebook 17)

- The four-model ensemble at buffer 4 scores **F1 ≈ 0.89** (5 ADL false alarms,
  2 falls missed) — higher buffer lengths trade more false alarms away for a
  little recall.
- The alarm trails the fall by roughly **1–2 s** (two high windows must lift the
  4-window mean over 0.5), plus the 1 s scoring cadence and the 4 s warm-up at
  stream start.
- Recurring false alarms are `lie down`, `sit floor` and `tie shoelace`: the
  zero-shot model still reads controlled descent / lying as falling. This is the
  domain-adaptation problem (Tier 1), not a wiring bug.
- The classifier watches the **largest person only**, as training did.
- CPU: the ONNX pose path measured ~24 ms/frame through `track()` (41 fps) on
  the 9800X3D; the ensemble adds ~3 ms once per second.

## Files

```
v2/app.py          Flask routes (unchanged from v1 plus /fall_clips/status)
v2/detector.py     RTSP -> ONNX pose -> ensemble -> MJPEG / status / recorder
v2/metrics.py      measurement mode + ensemble columns and provenance
v2/fall_recorder.py  pre-buffered fall-clip capture (unchanged from v1)
v2/replay.py       offline replay of one video through the same decision stack
v2/METRICS.md      what the measurement numbers do and do not mean
```
