# detection service

The v2 fall detector (`v2/`) as a container, wired to the Java notification
service. One container watches one room: it pulls the MediaMTX RTSP feed, runs
the calibrated fall decision, and on a latch sends an alert and a skeleton-only
clip to `services/notification`, which owns Telegram, escalation and storage.

```
camera -> MediaMTX (Pi) --RTSP--> detection service --POST /events-----------> notification service
                                       |             --PUT  /events/:id/clip------>  |-> Telegram
                                       |             --PUT  /events/:id/keypoints->  |-> Postgres
                                  skeleton-only mp4                                  |-> Supabase bucket
                                  + keypoints .npz
```

## What is sent, and what is stored

- **Alert** (`POST /events`): `{eventId, roomId, roomName, confidence}`, sent
  the moment the alarm latches. `eventId` is a fresh UUID per incident;
  the notification service dedups on it, so retries are safe.
- **Clip** (`PUT /events/{eventId}/clip`): the automatic fall clip, uploaded
  when it is complete. It is **skeleton only** -- pose drawn on a black canvas,
  no camera pixels ever reach `fall_recorder` -- which is also the point of the
  notification service's `skeleton-clips` bucket.
- **Keypoint file** (`PUT /events/{eventId}/keypoints`): `fall_<ts>.npz`, the
  same frames as the clip as numbers -- every tracked person's keypoints in the
  cache convention `(y_norm, x_norm, conf)`, their tracker and subject ids
  (one subject spans the new id the tracker gives a faller mid-fall), their
  boxes, the alarm and score per frame, and the frame the alarm latched on.
  Uploaded right
  after the clip. The notification service files it under the alert's triage
  outcome (`keypoints/labelled/genuine|false_alarm/`), so every alert becomes a
  labelled training example. Format: `fallcore/keypoint_file.py`;
  `FALL_KEYPOINTS_ENABLED=0` turns it off.
- The service does **not** keep footage: the local clip and keypoint file are
  deleted after the upload (`KEEP_UPLOADED_CLIPS=0`), and the manual Record
  button -- the one path that writes raw camera video -- is refused with
  `ALLOW_MANUAL_RECORDING=0`.

## Run

The image fetches its own weights during the build (`scripts/fetch_weights.py`
in a build stage), so nothing has to be on the host. The build reads
`FALL_WEIGHTS_TOKEN` from `.env`, which compose passes in as a build secret: the
token is not in any layer of the image, and the running container gets it
blanked.

```
copy .env.example .env                                # fill in ROOM_*, MEDIAMTX_*, FALL_WEIGHTS_TOKEN
copy ..\..\.env.example.local ..\..\.env              # notification settings
docker compose up -d --build                          # repo root docker-compose.yml
```

The weights stage is cached, so rebuilding for a code change does not download
again; `docker compose build --no-cache detection` fetches them afresh.

The image holds one pose backbone, yolo26m (INT8, 384x640), and the camera
head fitted on its keypoints. They are one choice, not settings: the detector
refuses a head fitted on another backbone, stops when either file is missing,
and refuses the variables that used to swap them (`POSE_SCALE`, `POSE_MODEL`,
`POSE_IMGSZ`, `ENSEMBLE_CKPTS`, `CALIBRATION_HEAD`). `/stream/status` reports
it as `pose`.

Detection and notification share the root project's network, so
`http://notification:8081/api/v1/notification` resolves. The UI is published on
`127.0.0.1:5005` only (it has no auth). Logs: `docker compose logs -f detection`.

Local run without Docker (from this directory):

```
..\..\demo-cam\.venv\Scripts\python.exe -m service.main
```

## Configuration

All settings come from the environment. `.env.example` holds only what a
deployment must set; every other setting has a working default, in
`service/config.py` for the service and in `v2/README.md` for the stream, pose
and fall-decision variables. The required three are `ROOM_ID`, `ROOM_NAME` and
`MEDIAMTX_HOST`.

The startup guard refuses to run with `FALL_CLIPS_ENABLED=0`: with no viewer
attached the capture watchdog would drop the RTSP connection after
`IDLE_STOP_SEC`, and the service would go blind.

## Reliability

- The alert is spooled to an append-only JSONL journal (`SPOOL_DIR`) before the
  HTTP call, so a restart between the latch and the upload loses nothing.
- Transient failures (timeout, connection refused, 5xx, 404 because the event
  has not landed yet) retry with exponential backoff up to
  `NOTIFY_MAX_BACKOFF_SEC`, forever.
- Permanent failures (400, clip over `MAX_CLIP_BYTES`, 413/415 on the keypoint
  file) are logged and dropped; the file is kept for inspection.
- The clip is uploaded before the keypoint file, and the journal records when
  it has gone, so a keypoint upload that has to retry never delays or re-sends
  the clip.
- A clip that finished while the service was restarting is re-attached at boot
  (`RECONCILE_CLIPS=1`), with its keypoint file. This is safe because
  `fall_recorder` only renames each file to its final name (`fall_*.mp4`,
  `fall_*.npz`) once it is complete; a half-written one stays `.raw.mp4` or
  `.npz.tmp` and is never uploaded. If the video writer fails to open, the
  keypoint file is still sent; if there is no keypoint file either, the event is
  closed out -- the alert stands, nothing empty is uploaded.

## Tests

```
python -m unittest discover -s tests -v
```

The notifier tests run against a stub HTTP server and exercise the whole status
matrix; the watcher and outbox tests use a temporary spool and clip directory;
the recorder tests drive `fall_recorder.push_frame` through a whole clip and
read back the keypoint file it writes.

## Files

```
Dockerfile             python:3.12-slim + ffmpeg; torch from the CPU wheel index
                       (build context is this directory)
pyproject.toml         the `fallcore` package (`pip install -e .` from here)
fallcore/              the pose + fall-decision pipeline v2 imports
v2/                    the Flask app, detector and fall_recorder the service runs
scripts/               fetch_weights, bench_pose, fit_calibration_head
models/ runs/          weights and checkpoints (gitignored; fetch_weights.py)
requirements.txt       runtime pins (opencv headless; torch installed separately)
service/config.py      Settings.from_env() -- ROOM_ID/ROOM_NAME required
service/main.py        entry point: env, always-on capture, watcher, outbox, Flask UI
service/watcher.py     fall_recorder events -> outbox; restart reconciliation
service/spool.py       append-only JSONL outbox with compaction
service/notifier.py    HTTP client for POST /events and PUT /events/:id/{clip,keypoints}
service/outbox.py      delivery worker with backoff and clip/keypoint-file cleanup
tests/                 unittest suite (service + scripts/bench_pose.py)
```
