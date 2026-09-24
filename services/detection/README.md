# detection service

The demo-cam v2 fall detector as a container, wired to the Java notification
service. One container watches one room: it pulls the MediaMTX RTSP feed, runs
the calibrated fall decision, and on a latch sends an alert and a skeleton-only
clip to `services/notification`, which owns Telegram, escalation and storage.

```
camera -> MediaMTX (Pi) --RTSP--> detection service --POST /events-----> notification service
                                       |             --PUT  /events/:id/clip-->   |-> Telegram
                                       |                                       |-> Postgres
                                  skeleton-only mp4                            |-> Supabase bucket
```

## What is sent, and what is stored

- **Alert** (`POST /events`): `{eventId, roomId, roomName, confidence}`, sent
  the moment the alarm latches. `eventId` is a fresh UUID per incident;
  the notification service dedups on it, so retries are safe.
- **Clip** (`PUT /events/{eventId}/clip`): the automatic fall clip, uploaded
  when it is complete. It is **skeleton only** -- pose drawn on a black canvas,
  no camera pixels ever reach `fall_recorder` -- which is also the point of the
  notification service's `skeleton-clips` bucket.
- The service does **not** keep footage: the local clip is deleted after the
  upload (`KEEP_UPLOADED_CLIPS=0`), and the manual Record button -- the one
  path that writes raw camera video -- is refused with `ALLOW_MANUAL_RECORDING=0`.
  A metrics run writes CSV/JSON, no video, and stays allowed.

## Run

Weights are artifacts and are bind-mounted, not baked into the image. Fetch
them once on the host:

```
python scripts/fetch_weights.py          # demo-cam/models + runs/checkpoints
```

Then:

```
cd services/notification && docker compose up -d      # db + storage + notification
cd services/detection
copy .env.example .env                                # fill in ROOM_*, MEDIAMTX_*
docker compose up -d --build
```

The compose file joins the notification compose's default network, so
`http://notification:8081/api/v1/notification` resolves. The UI is published on
`127.0.0.1:5005` only (it has no auth). Logs: `docker compose logs -f detection`.

Local run without Docker (from this directory):

```
..\..\.venv-train\Scripts\python.exe -m service.main
```

## Configuration

All settings come from the environment; `.env.example` documents every one.
The required three are `ROOM_ID`, `ROOM_NAME` and `MEDIAMTX_HOST`; the stream,
pose and fall-decision variables are the ones from `demo-cam/.env.example`.

The startup guard refuses to run with `FALL_CLIPS_ENABLED=0`: with no viewer
attached the capture watchdog would drop the RTSP connection after
`IDLE_STOP_SEC`, and the service would go blind.

## Reliability

- The alert is spooled to an append-only JSONL journal (`SPOOL_DIR`) before the
  HTTP call, so a restart between the latch and the upload loses nothing.
- Transient failures (timeout, connection refused, 5xx, 404 because the event
  has not landed yet) retry with exponential backoff up to
  `NOTIFY_MAX_BACKOFF_SEC`, forever.
- Permanent failures (400, clip over `MAX_CLIP_BYTES`) are logged and dropped;
  the clip file is kept for inspection.
- A clip that finished while the service was restarting is re-attached at boot
  (`RECONCILE_CLIPS=1`). This is safe because `fall_recorder` only renames a
  clip to its final `fall_*.mp4` name once it is complete; a half-written file
  stays `.raw.mp4` and is never uploaded. If the writer fails to open, the
  `clip_ready` event reports the clip lost and the event is closed out -- the
  alert stands, nothing empty is uploaded.

## Tests

```
python -m unittest discover -s tests -v
```

The notifier tests run against a stub HTTP server and exercise the whole status
matrix; the watcher and outbox tests use a temporary spool and clip directory.

## Files

```
Dockerfile             python:3.12-slim + ffmpeg; torch from the CPU wheel index
docker-compose.yml     joins notification_default; bind-mounts weights
requirements.txt       runtime pins (opencv headless; torch installed separately)
service/config.py      Settings.from_env() -- ROOM_ID/ROOM_NAME required
service/main.py        entry point: env, always-on capture, watcher, outbox, Flask UI
service/watcher.py     fall_recorder events -> outbox; restart reconciliation
service/spool.py       append-only JSONL outbox with compaction
service/notifier.py    HTTP client for POST /events and PUT /events/:id/clip
service/outbox.py      delivery worker with backoff and clip cleanup
tests/                 unittest suite
```
