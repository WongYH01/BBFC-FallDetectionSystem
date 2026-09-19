# v2 measurement mode

Tick **Collect metrics (no video)** in the sidebar, then press Record. On Stop
you get two files in `v2/metrics/`:

- `metrics_<ts>.json` — the summary
- `metrics_<ts>_frames.csv` — one row per frame

A measurement run writes **no video at all**. That is deliberate: encoding a
2560x720 mp4 every frame costs more than the detector does, so any figure
gathered while recording measures the encoder as much as the model.

The pose pane must be open in the browser — inference is driven by the MJPEG
request, so nothing is measured if nobody is watching the feed.

---

## v2 additions: the calibrated-model decision

Each CSV row now carries the fall decision as it stood on that frame:

| Column | Meaning |
|---|---|
| `p_fall` | P(fall) of the **last scored window** (empty until the 4 s buffer fills) |
| `p_fall_rolling` | mean of the last `BUFFER_LEN` window probabilities — the decision score |
| `n_windows` | windows scored so far in the run |
| `alarm` | the latched decision on that frame (1 = alarm engaged) |

These repeat between window scores, which is correct: the decision only changes
once per second. The summary gains a `fall_detection` block (model members -
now one backbone plus its room-fitted head,
buffer length, threshold, windows scored, mean probabilities, frames alarmed)
and the provenance block now names the **pose backend** (onnx or torch) and the
exact checkpoint paths — a pt run and an ONNX run are not comparable, and two
ensemble compositions are not either. `alarm_pct` is the share of scored frames
that were latched, not an accuracy figure; these runs carry no ground truth.

---

## What each number actually measures

### `rates.stream_fps` — frames the camera delivered

Counted by tapping Ultralytics' stream loader. Its reader thread pulls frames
off the RTSP socket at camera rate and stores each one; the consumer then takes
only the newest and discards whatever queued up behind it. Those discarded
frames are invisible to every other measurement, which is why this is counted
separately.

**Measures frames arriving at this laptop from MediaMTX.** Loss upstream of
MediaMTX — a struggling Pi, a bad link between Pi and MediaMTX — does not appear
here.

If the JSON says `"stream_fps_source": "unavailable"`, the tap did not install
or counted nothing, and `stream_fps` and `dropped_frame_pct` will be `null`.
That is intentional: a broken tap reports nothing rather than a plausible-looking
zero.

### `rates.inference_fps` — frames actually processed

Frames that completed the full loop. The difference from `stream_fps` is
`dropped_frame_pct`, and on a slow device that is usually the majority of them.

### `rates.model_inference_ms_mean` — the model alone

From Ultralytics' own per-frame timing. Note this is **not** `1000 /
inference_fps`. The gap between the two is everything that is not the model:
JPEG decode, `plot()`, the fall overlay, JPEG encode, MJPEG write. If your fps
is poor while this number is small, the model is not your bottleneck.

### `latency_ms` — arrival to encoded

**Scope: from the frame arriving at this host to the annotated JPEG being ready
to send.** It does **not** include:

- the camera sensor and the Pi's H.264 encode
- the network between Pi, MediaMTX, and this laptop
- the browser's decode and paint

Do not quote this as "end-to-end latency" without saying so. The real
sensor-to-display figure is larger — measure it with the procedure below.

`unmatched` counts frames whose arrival timestamp could not be tied to that exact
frame. A few is normal; a large number means the figure is unreliable.

### `keypoint_confidence` — visible joints only

`mean_visible` averages every keypoint scoring above `0.25` (the same threshold
Ultralytics uses to decide whether to draw a joint), pooled across the whole run
rather than averaged per frame, so busier frames weigh more.

**Always read it together with `visible_keypoint_rate`.** Restricted to visible
joints, a model that finds two confident keypoints and misses the other fifteen
still scores ~0.9. The rate is what tells you whether a high mean means good
pose estimation or just a very selective one.

### `presence_continuity` — did we keep seeing anyone

Percentage of frames with at least one person, plus the longest unbroken stretch
detecting nobody. The gap length is the number that matters for fall detection:
a detector blind for two seconds is a detector that can miss a fall.

### `track_stability` — proxies, not MOTA

- `frames_with_ids_pct` — of frames that had detections, how many carried track
  IDs at all. On this camera a large fraction typically do not.
- `fragmentation` — distinct IDs divided by the most people ever on screen at
  once. Two people and thirty IDs means the tracker kept losing and re-acquiring
  them.

**These are proxies.** Real ID-switch and identity-preservation counts (MOTA,
IDF1) require frame-by-frame ground-truth annotation, which these runs do not
have. High fragmentation is *evidence* of instability, not a measurement of it.
Say so if you put these in a report.

---

## Glass-to-glass latency (manual)

The automated figure misses the sensor, the Pi encode, the network, and the
browser. This procedure gets the real number.

1. Open a millisecond stopwatch full-screen on the laptop
   (`https://www.google.com/search?q=stopwatch`, or any clock showing
   milliseconds).
2. Point the camera at that screen, with the v1 page's pose pane visible
   alongside the stopwatch.
3. Photograph the screen with a phone, capturing the live stopwatch and the pose
   pane — which shows the stopwatch as it was when that frame was captured — in
   one shot.
4. Latency = live reading − reading visible in the pose pane.
5. **Repeat about ten times and take the median.** A single reading is dominated
   by shutter timing and tells you very little.

Record the result here, with the conditions:

| Date | Venv / device | imgsz | Median glass-to-glass | Measured `latency_ms` p50 | Difference |
|---|---|---|---|---|---|
| | | | | | |

That difference is the constant the automated figure is missing. Once you have
it, the per-run `latency_ms` becomes a useful proxy you can track cheaply.

---

## Comparing runs

Only compare runs from the **same venv, device, and thermal state**. The WSL
venv uses CUDA and the Windows venv is CPU-only, so their figures are not
comparable at all, and the laptop drifts measurably across consecutive runs as
it heats up. Every summary carries a `provenance` block naming the device,
model, and `imgsz` for exactly this reason — check it before putting two numbers
side by side.

For anything you intend to publish, run three times and report the spread, not a
single number.

## Tunables

| Env var | Default | Effect |
|---|---|---|
| `METRICS_WARMUP_FRAMES` | `30` | Frames excluded from the summary. They stay in the CSV with `warmup=1`. |
| `KPT_CONF_THRESH` | `0.25` | What counts as a visible keypoint. |

Warm-up frames are dropped because the first inference of a run pays for CUDA
kernel compilation and cuDNN autotuning and can be an order of magnitude slower
than steady state. If a run is shorter than the warm-up count, the JSON says
`"insufficient_data": true` rather than reporting figures from nothing.
