"""Server-side YOLO11 pose tracking on the camera stream.

Pulls frames from the MediaMTX RTSP feed, runs Ultralytics pose tracking
(keypoints + persistent track IDs), draws the skeleton, and yields JPEG
frames as an MJPEG stream for the Flask `/video_feed` route.

Quick-and-dirty: model loads once at import, single shared generator.
"""
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

# Force RTSP over TCP before OpenCV's FFmpeg backend loads — UDP drops packets
# and stalls frame delivery ("Waiting for stream"). Must be set before cv2 opens
# the capture.
#
# The rest are latency options, and they matter for the "reader is too slow"
# warnings MediaMTX logs about us: nobuffer/low_delay stop FFmpeg holding frames
# back for smoothness we do not want, and reorder_queue_size;0 disables the
# packet-reordering buffer, which over a TCP transport can only ever add delay.
# Together they keep this client's own backlog near zero, so whatever lag shows
# up is real network or real CPU rather than buffering we asked for.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|reorder_queue_size;0",
)

import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

import metrics as _metrics
import fall_recorder

# ---------------------------------------------------------------------------
# Config — reuse the same env vars the Flask app already reads
# ---------------------------------------------------------------------------
STREAM_USERNAME = os.environ.get("STREAM_USERNAME", "")
STREAM_PASSWORD = os.environ.get("STREAM_PASSWORD", "")
MEDIAMTX_HOST   = os.environ["MEDIAMTX_HOST"]
RTSP_PORT       = os.environ.get("MEDIAMTX_RTSP_PORT", "8554")
STREAM_PATH     = os.environ.get("STREAM_PATH", "cam")

# Weights live in the shared models/ folder at the repo root (one level up).
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_NAME = os.environ.get("POSE_MODEL", str(MODELS_DIR / "yolo26n-pose.pt"))
# MODEL_NAME = hf_hub_download(repo_id="melihuzunoglu/human-fall-detection", filename="best.pt")

IMGSZ      = int(os.environ.get("POSE_IMGSZ", "640"))  # drop to 480 if slow

# Decode-and-score one frame in every VID_STRIDE. The RTSP socket is still
# drained at full rate — Ultralytics' loader grabs every frame and only fully
# retrieves every Nth — so this buys CPU back WITHOUT making MediaMTX call us a
# slow reader. At 2 on a 15 fps stream the fall debounce still gets ~7 samples a
# second, which is well inside FALL_MIN_FRAMES territory.
VID_STRIDE = max(1, int(os.environ.get("VID_STRIDE", "1")))

# JPEG quality for the MJPEG fan-out. OpenCV defaults to 95, which is a lot of
# bytes and a lot of encode time for a monitoring feed nobody is pixel-peeping.
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "75"))

# No decoded frame for this long means the stream is wedged rather than slow.
# The Pi's hardware encoder dying mid-stream (the ioctl(VIDIOC_QBUF) failures in
# the MediaMTX journal) looks exactly like this from here: the TCP connection
# stays up and nothing ever arrives again. Tear it down and reconnect instead of
# waiting forever on a socket that will never speak.
STREAM_STALL_SEC = float(os.environ.get("STREAM_STALL_SEC", "8"))

# The same guard for a connection that comes up but never delivers a first
# frame — a path that exists on MediaMTX with a dead publisher behind it looks
# exactly like that. Longer than STREAM_STALL_SEC because this window also has
# to cover Ultralytics building its predictor and the RTSP handshake itself.
STREAM_CONNECT_SEC = float(os.environ.get("STREAM_CONNECT_SEC",
                                          str(STREAM_STALL_SEC * 3)))

# Seconds after the last viewer leaves before the RTSP connection is dropped.
# Ignored while automatic fall clips are enabled or a recording is running —
# detection is the point, a browser being open is not.
IDLE_STOP_SEC = float(os.environ.get("IDLE_STOP_SEC", "30"))

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "videos")
RECORD_FPS = int(os.environ.get("RECORD_FPS", "10"))  # nominal, not measured

# Background of the skeleton-only companion recording. Black by default because
# the COCO keypoint palette is bright and reads best on a dark field; white is
# there for slides and printed reports, which usually want the opposite.
SKELETON_BG = os.environ.get("SKELETON_BG", "black")  # "black" | "white"

# ---------------------------------------------------------------------------
# Fall heuristic
#
# The whole rule: a person's bounding box is wider than it is tall. Standing
# people are tall boxes, people on the floor are wide ones.
#
# Be clear-eyed about what this is. It detects LYING DOWN, not FALLING — there
# is no velocity term, so someone already on the floor when the stream starts
# reads identically to someone who just went down. It MISSES a fall where the
# person lands head- or feet-toward the camera, because foreshortening keeps the
# box tall. It FIRES on sitting on the floor, crouching, and bending over.
# Occlusion by furniture clips the box and corrupts the ratio outright. Camera
# height and angle dominate all of it; from a ceiling mount the rule is close to
# meaningless. FALL_RATIO lets you trade these off per camera, but no threshold
# fixes them — they're limits of the feature, not of the tuning.
# ---------------------------------------------------------------------------
# Threshold on width/height. 1.0 is exactly "height less than width"; it's an
# env var because the right value depends entirely on how the camera is mounted.
FALL_RATIO        = float(os.environ.get("FALL_RATIO", "1.0"))
FALL_MIN_FRAMES   = int(os.environ.get("FALL_MIN_FRAMES", "3"))   # frames to latch
FALL_CLEAR_FRAMES = int(os.environ.get("FALL_CLEAR_FRAMES", "5")) # frames to clear
FALL_DEBUG        = os.environ.get("FALL_DEBUG", "") not in ("", "0")

# ---------------------------------------------------------------------------
# Recording state (side-by-side raw + annotated), shared with gen_frames()
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_recording = False
# Actual cv2.VideoWriter.write() calls happen on background threads, fed
# through these queues -- see _writer_worker(). A slow encode must never
# block gen_frames()'s frame-reading loop: that loop is also what's
# draining the RTSP source, so stalling it can back up the connection all
# the way to the camera/Pi. (This project briefly used H.264 here, which
# was heavy enough in software to cause exactly that -- back to mp4v; see
# the fourcc comment below.)
_write_queue: "queue.Queue | None" = None
_filename = None
_filepath = None
_start_time = None

# Skeleton-only companion file: the same frames with the pose drawn on a blank
# canvas instead of the camera image. Written alongside the side-by-side, never
# in place of it.
_write_queue_skel: "queue.Queue | None" = None
_filename_skel = None
_filepath_skel = None


def _writer_worker(writer: "cv2.VideoWriter", q: "queue.Queue") -> None:
    """Runs on its own thread for the life of one recording.

    Pulls frames off `q` and writes them -- this is where the (possibly
    slow) encode actually happens, off the frame-reading thread. A `None`
    on the queue is the signal that recording stopped: drain, then release.
    """
    while True:
        frame = q.get()
        if frame is None:
            break
        writer.write(frame)
    writer.release()

# Measurement mode. A run with this set writes NO video at all — see metrics.py
# for why that is the point rather than a shortcut.
_metrics_mode = False
_run: "object | None" = None


def start_recording(metrics: bool = False) -> dict:
    """Start a recording, or a measurement run when `metrics` is true.

    The two are mutually exclusive: a measurement run produces metrics/*.json
    and *.csv and no video, a normal recording produces the two mp4s and no
    metrics.
    """
    global _recording, _write_queue, _filename, _filepath, _start_time
    global _write_queue_skel, _filename_skel, _filepath_skel, _metrics_mode, _run
    with _lock:
        if _recording:
            return {"recording": True, "metrics": _metrics_mode,
                    "filename": _filename, "filename_skeleton": _filename_skel}

        if metrics:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            _run = _metrics.Run(ts, _metrics.provenance_of(model, IMGSZ))
            _metrics_mode = True
            _filename = _filename_skel = _filepath = _filepath_skel = None
            _write_queue = _write_queue_skel = None
            _start_time = time.time()
            _recording = True
            start_worker()   # a metrics run with no browser open still needs frames
            return {"recording": True, "metrics": True,
                    "filename": None, "filename_skeleton": None}

        _metrics_mode = False
        _run = None
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        # One now() for both names. Two calls could straddle a second boundary
        # and give the pair different timestamps, which is the one thing that
        # makes them not obviously a pair.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _filename = f"pose_{ts}.mp4"
        _filepath = os.path.join(VIDEOS_DIR, _filename)
        _filename_skel = f"pose_{ts}_skeleton.mp4"
        _filepath_skel = os.path.join(VIDEOS_DIR, _filename_skel)
        # Both opened lazily in the capture loop once frame sizes are known — they
        # differ (2560x720 side-by-side vs 1280x720 skeleton).
        _write_queue = None
        _write_queue_skel = None
        _start_time = time.time()
        _recording = True
        start_worker()   # ditto: recording does not require a live viewer
        return {"recording": True, "metrics": False, "filename": _filename,
                "filename_skeleton": _filename_skel}


def stop_recording() -> dict:
    """Stop the run and finalize whichever artefacts it was producing."""
    global _recording, _write_queue, _filename, _start_time, _write_queue_skel, _filename_skel
    global _metrics_mode, _run
    with _lock:
        duration = round(time.time() - _start_time, 1) if _start_time else 0
        was_metrics, run = _metrics_mode, _run
        # Signal the worker threads rather than releasing here directly: the
        # queues may still hold unwritten frames, and encoding those (then
        # calling release()) can take a moment -- doing it inline here would
        # block this Flask request thread, not the frame-reading loop, but
        # it's still better kept off any hot path. Finishes in the
        # background; the files are complete once each worker exits.
        if _write_queue is not None:
            _write_queue.put(None)
            _write_queue = None
        if _write_queue_skel is not None:
            _write_queue_skel.put(None)
            _write_queue_skel = None
        _recording = False
        _metrics_mode = False
        _run = None
        filename, filename_skel = _filename, _filename_skel
        _start_time = None

    # Outside the lock: writing the CSV touches the disk, and gen_frames needs
    # the lock every frame.
    if was_metrics and run is not None:
        out = run.finish()
        return {"recording": False, "metrics": True, "duration_sec": duration,
                "filename": None, "filename_skeleton": None,
                "filename_metrics": out["filename_metrics"],
                "filename_frames": out["filename_frames"],
                "summary": out["summary"]}

    return {"recording": False, "metrics": False, "filename": filename,
            "filename_skeleton": filename_skel, "duration_sec": duration}


def get_status() -> dict:
    with _lock:
        elapsed = round(time.time() - _start_time, 1) if _recording and _start_time else 0
        return {"recording": _recording, "metrics": _metrics_mode,
                "filename": _filename, "filename_skeleton": _filename_skel,
                "elapsed_sec": elapsed}


# ---------------------------------------------------------------------------
# Fall state, shared between gen_frames() (writer) and /fall/status (reader).
# Its own lock, not the recording one — different lifetime, and the Flask status
# thread reads this while the inference thread writes it.
# ---------------------------------------------------------------------------
_fall_lock = threading.Lock()
_tracks: dict = {}          # track_id -> {hits, misses, fallen, ratio, last_seen}
_frame_no = 0
_fall_since: "float | None" = None
_last_untracked: list = []  # this frame's ID-less boxes (no latch, see below)
_last_update = 0.0          # wall clock of the last frame scored

_STALE_FRAMES = 60          # forget a track's identity after this long unseen
_FRESH_FRAMES = 15          # ...but it only counts as "present" this recently
_STALE_SECONDS = 1.5        # no frames for this long -> report nothing, not "clear"

_RED = (0, 0, 255)


def _update_falls(result) -> list:
    """Score every box in this frame and update the per-person latch.

    Returns this frame's people as
    {"track_id", "bbox", "ratio", "fallen", "tracked"} — the drawing input.
    """
    global _frame_no, _fall_since, _last_untracked, _last_update

    boxes = getattr(result, "boxes", None)
    xyxy = []
    ids = None
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        # boxes.id is None whenever the tracker hasn't confirmed anything.
        # Measured on this camera: 72 of 117 frames that had detections came
        # back with no IDs at all. Never index it without this check.
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else None

    people = []
    untracked = []
    with _fall_lock:
        _frame_no += 1

        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            w, h = float(x2 - x1), float(y2 - y1)
            if h <= 0:                      # guards the division
                continue
            ratio = w / h
            bbox = (int(x1), int(y1), int(x2), int(y2))

            if ids is None:
                # No identity means no latch. Keying the debounce on box
                # position instead would be worse than not debouncing at all:
                # box order isn't stable, so "box 0" is fed by a different
                # person from frame to frame and the latch it builds up is
                # about nobody. Report the bare rule and move on — an ID-less
                # box lives exactly as long as it is on screen.
                p = {"track_id": None, "bbox": bbox, "ratio": ratio,
                     "fallen": ratio > FALL_RATIO, "tracked": False}
                untracked.append(p)
                people.append(p)
                continue

            key = ids[i]
            st = _tracks.get(key)
            if st is None:
                st = {"hits": 0, "misses": 0, "fallen": False,
                      "ratio": ratio, "last_seen": _frame_no}
                _tracks[key] = st

            if ratio > FALL_RATIO:
                st["hits"] += 1
                st["misses"] = 0
            else:
                st["misses"] += 1
                st["hits"] = 0

            # Asymmetric thresholds on purpose: a single latch-and-clear count
            # would strobe right at the boundary, which is the exact flicker
            # this debounce exists to kill.
            if not st["fallen"] and st["hits"] >= FALL_MIN_FRAMES:
                st["fallen"] = True
            elif st["fallen"] and st["misses"] >= FALL_CLEAR_FRAMES:
                st["fallen"] = False

            st["ratio"] = ratio
            st["last_seen"] = _frame_no
            people.append({"track_id": key, "bbox": bbox, "ratio": ratio,
                           "fallen": st["fallen"], "tracked": True})

        # Age tracks out by last_seen rather than rebuilding the dict each
        # frame, so one missed detection doesn't reset somebody's latch.
        for key in [k for k, v in _tracks.items()
                    if _frame_no - v["last_seen"] > _STALE_FRAMES]:
            del _tracks[key]

        _last_untracked = untracked
        _last_update = time.time()

        # A latched track counts toward the alarm for _FRESH_FRAMES after it
        # was last seen, so a brief detection dropout doesn't blink the alert
        # off. Beyond that it stops counting even though the identity is kept
        # — otherwise somebody who fell and then walked away holds the alarm
        # on for the full _STALE_FRAMES.
        if any(p["fallen"] for p in people) or _any_latched_locked():
            if _fall_since is None:
                _fall_since = time.time()
        else:
            _fall_since = None

    return people


def _live_tracks_locked() -> list:
    """Latched tracks recent enough to count as present. Caller holds the lock."""
    return [(k, v) for k, v in sorted(_tracks.items())
            if _frame_no - v["last_seen"] <= _FRESH_FRAMES]


def _any_latched_locked() -> bool:
    return any(v["fallen"] for _, v in _live_tracks_locked())


def _draw_falls(canvas, people) -> None:
    """Overlay the alert on top of result.plot()'s own annotations."""
    for p in people:
        x1, y1, x2, y2 = p["bbox"]
        if p["fallen"]:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), _RED, 3)
            # An un-ID'd box has nothing to call it; "ID None" reads as a bug.
            label = ("FALL" if p["track_id"] is None
                     else f"FALL  ID {p['track_id']}")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # Anchored to the BOTTOM of the box. Ultralytics puts its own
            # "person 0.90" chip at the top-left corner, and two chips in the
            # same place is unreadable.
            bot = min(y2, canvas.shape[0])
            top = max(bot - th - 8, 0)
            cv2.rectangle(canvas, (x1, top), (x1 + tw + 8, top + th + 8), _RED, -1)
            cv2.putText(canvas, label, (x1 + 4, top + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
        if FALL_DEBUG:
            # Right-aligned at the box top — the one corner nothing else uses.
            txt = f"{p['ratio']:.2f}"
            (tw, _th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(canvas, txt, (max(x2 - tw - 4, 0), max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        _RED if p["fallen"] else (0, 255, 255), 2, cv2.LINE_AA)

    if any(p["fallen"] for p in people):
        h, w = canvas.shape[:2]
        cv2.rectangle(canvas, (0, 0), (w, 40), _RED, -1)
        # ASCII only. putText draws Hershey fonts, which have no glyph for an
        # emoji or an em dash and silently substitute '?'.
        cv2.putText(canvas, f"FALL DETECTED ({sum(p['fallen'] for p in people)})",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                    cv2.LINE_AA)


def get_fall_status() -> dict:
    with _fall_lock:
        # No frames recently means the feed died or nobody is watching it.
        # Report an empty scene rather than a stale one — the page shows "no
        # feed" for this, which is honest; showing the last known state would
        # let a dead detector read as a safe room.
        live = time.time() - _last_update < _STALE_SECONDS

        people = []
        if live:
            people = [
                {"track_id": k, "ratio": round(v["ratio"], 2),
                 "fallen": v["fallen"], "tracked": True}
                for k, v in _live_tracks_locked()
            ]
            people += [
                {"track_id": None, "ratio": round(p["ratio"], 2),
                 "fallen": p["fallen"], "tracked": False}
                for p in _last_untracked
            ]

        n_fallen = sum(p["fallen"] for p in people)
        return {
            "live": live,
            "any_fallen": bool(n_fallen),
            "count": n_fallen,
            "tracked": len(people),
            "since_sec": (round(time.time() - _fall_since, 1)
                          if _fall_since and n_fallen else 0),
            "max_ratio": round(max((p["ratio"] for p in people), default=0.0), 2),
            "threshold": FALL_RATIO,
            "people": people,
        }


def get_fall_clip_status() -> dict:
    """Status of the automatic pre-buffered fall-clip recorder (fall_recorder.py)."""
    return fall_recorder.get_status()


def _rtsp_url() -> str:
    """Build rtsp://user:pass@host:port/path from env, URL-encoding creds."""
    if STREAM_USERNAME:
        auth = f"{quote(STREAM_USERNAME, safe='')}:{quote(STREAM_PASSWORD, safe='')}@"
    else:
        auth = ""
    return f"rtsp://{auth}{MEDIAMTX_HOST}:{RTSP_PORT}/{STREAM_PATH}"


# Load the model once at import (shared across requests)
model = YOLO(MODEL_NAME)


# ---------------------------------------------------------------------------
# One reader, many viewers
#
# This used to be a plain generator, and Flask called it once per request. Every
# browser that opened /video_feed therefore got its OWN model.track(), which
# opened its OWN RTSP connection to MediaMTX and ran its OWN YOLO loop. Two tabs
# meant two readers on the Pi and two inference loops here fighting over the same
# CPU — and a loop that is busy doing inference is a loop that is not draining
# its socket, which is precisely what MediaMTX reports as
#
#     [RTSP] [session 1497e783] reader is too slow, discarding 469 frames
#
# Worse, closing the tab did not reliably end the session. Ultralytics' stream
# loader keeps a background grab thread and a cv2.VideoCapture alive per source,
# and dropping the generator does not stop either, so orphaned readers piled up
# on the Pi across page reloads.
#
# So: exactly one capture+inference thread per process, started on first use,
# publishing the newest annotated JPEG into a single slot. Viewers subscribe to
# that slot. N viewers now cost one RTSP session and one inference pass, and a
# viewer on a slow link simply misses frames instead of applying back-pressure
# all the way to the camera.
# ---------------------------------------------------------------------------
_frame_cv = threading.Condition()
_latest_jpeg: "bytes | None" = None
_latest_seq = 0

_worker_lock = threading.Lock()
_worker: "threading.Thread | None" = None
_worker_stop = threading.Event()

_viewer_lock = threading.Lock()
_viewers = 0
_viewers_zero_since: "float | None" = None

_stream_lock = threading.Lock()
_stream_stats = {"connected": False, "frames": 0, "reconnects": 0,
                 "last_frame": 0.0, "started": 0.0, "fps": 0.0,
                 "last_error": None}

_BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"


def get_stream_status() -> dict:
    """What the single reader is doing — surfaced at /stream/status.

    Worth watching `reconnects` specifically: a number that climbs while you are
    not touching anything is the Pi dropping the path, not this end losing
    interest.
    """
    with _stream_lock:
        s = dict(_stream_stats)
    last = s.pop("last_frame")
    s.pop("started")
    s["fps"] = round(s["fps"], 1)
    s["age_sec"] = round(time.time() - last, 1) if last else None
    s["viewers"] = _viewers
    s["running"] = _worker is not None and _worker.is_alive()
    s["vid_stride"] = VID_STRIDE
    return s


def _close_stream(stream) -> None:
    """End one track() run and, more importantly, its RTSP session.

    This is the part that leaked. Ultralytics holds the VideoCapture inside
    predictor.dataset and runs a grab thread beside it; garbage-collecting the
    generator stops neither, so MediaMTX kept counting a closed tab as a live
    reader. close() on the dataset is what actually releases the capture and
    ends the session on the Pi.

    Safe to call from another thread while the capture loop is mid-iteration:
    the loader's threads stop, __next__ raises StopIteration, and the loop falls
    through to its reconnect. That is exactly how the stall watchdog below
    interrupts a wedged stream.
    """
    dataset = getattr(getattr(model, "predictor", None), "dataset", None)
    for obj, what in ((dataset, "dataset"), (stream, "generator")):
        close = getattr(obj, "close", None)
        if close is None:
            continue
        try:
            close()
        except Exception as exc:  # noqa: BLE001 - teardown must never raise
            print(f"[detector] closing the {what} failed: {exc!r}")


def _publish(jpeg: bytes) -> None:
    """Hand a finished frame to every waiting viewer and record the tick."""
    global _latest_jpeg, _latest_seq
    now = time.time()
    with _frame_cv:
        _latest_jpeg = jpeg
        _latest_seq += 1
        _frame_cv.notify_all()
    with _stream_lock:
        prev = _stream_stats["last_frame"]
        _stream_stats["connected"] = True
        _stream_stats["frames"] += 1
        _stream_stats["last_frame"] = now
        _stream_stats["last_error"] = None
        dt = now - prev if prev else 0
        if dt > 0:
            inst = 1.0 / dt
            # Smoothed, because the raw number jitters too much to read.
            _stream_stats["fps"] = (0.9 * _stream_stats["fps"] + 0.1 * inst
                                    if _stream_stats["fps"] else inst)


def _capture_loop() -> None:
    """The one thread that talks to the camera. Reconnects on anything."""
    # Assigned below, in the recording branch lifted out of the old gen_frames.
    global _write_queue, _write_queue_skel
    rtsp_url = _rtsp_url()
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    first = True

    while not _worker_stop.is_set():
        if not first:
            with _stream_lock:
                _stream_stats["reconnects"] += 1
        first = False
        stream = None
        with _stream_lock:
            _stream_stats["started"] = time.time()
        try:
            stream = model.track(
                source=rtsp_url,
                stream=True,
                imgsz=IMGSZ,
                vid_stride=VID_STRIDE,
                verbose=False,
            )
            for result in stream:
                if _worker_stop.is_set():
                    break

                # Cheap identity check after the first frame. model.track()
                # builds a fresh predictor and dataset on every reconnect and
                # the tap does not survive one, so re-installing has to be
                # automatic rather than remembered.
                _metrics.install_tap(getattr(model.predictor, "dataset", None))

                frame = result.plot()  # BGR ndarray with skeleton + track IDs

                # EXACTLY once per frame. _update_falls advances _frame_no and
                # every track's hits/misses, so scoring a second time to build
                # the skeleton canvas below would double-count the debounce and
                # latch the alert in ~1.5 frames instead of 3. Hold the result.
                people = _update_falls(result)

                # Before the recording branch, so a saved clip captures the
                # alert. plot() deep-copies orig_img, so drawing here cannot
                # reach the raw half of the hstack below.
                _draw_falls(frame, people)

                # A measurement run must never touch a VideoWriter — the encode
                # is most of what it would otherwise be measuring. Everything
                # below (manual recording AND the automatic fall-clip capture)
                # is skipped during a metrics run for the same reason.
                #
                # need_skel: build the skeleton canvas only when something will
                # actually use it. Set FALL_CLIPS_ENABLED=0 in .env to drop the
                # auto-capture side of this if the stream is struggling — that
                # removes an extra full result.plot() call from every frame
                # instead of only the ones being manually recorded.
                need_skel = _recording or fall_recorder.ENABLED
                if not _metrics_mode and need_skel:
                    # Same annotations, blank canvas. plot() draws onto whatever
                    # `img` it is handed (it deep-copies it first, so `blank` is
                    # untouched), and boxes/labels/conf off leaves only the pose.
                    blank = np.zeros_like(result.orig_img)
                    if SKELETON_BG == "white":
                        blank[:] = 255
                    skel = result.plot(img=blank, boxes=False, labels=False,
                                       conf=False)
                    _draw_falls(skel, people)  # same list, second canvas

                    # Computed every frame (not just while manually recording)
                    # because fall_recorder needs a continuous skeleton feed to
                    # keep its rolling pre-event buffer current — see
                    # fall_recorder.py for the automatic capture logic.
                    if fall_recorder.ENABLED:
                        any_fallen = any(p["fallen"] for p in people)
                        fall_recorder.push_frame(skel, any_fallen)

                    if _recording:
                        combined = np.hstack([result.orig_img, frame])
                        with _lock:
                            if _recording:  # re-check inside the lock
                                # "mp4v" -- back from H.264, which was heavy
                                # enough in software to be part of why the
                                # camera stream was freezing during a
                                # recording. mp4v plays fine in VLC but not
                                # Telegram (frozen first frame); the plan is
                                # to convert a finished clip separately
                                # instead of encoding it live.
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                if _write_queue is None:
                                    h, w = combined.shape[:2]
                                    writer = cv2.VideoWriter(
                                        _filepath, fourcc, RECORD_FPS, (w, h)
                                    )
                                    if not writer.isOpened():
                                        print(f"[detector] WARNING: could not "
                                              f"open a video writer for "
                                              f"{_filepath} -- no video "
                                              f"will be written for this "
                                              f"recording.")
                                    _write_queue = queue.Queue()
                                    threading.Thread(
                                        target=_writer_worker,
                                        args=(writer, _write_queue),
                                        daemon=True,
                                    ).start()
                                if _write_queue_skel is None:
                                    h, w = skel.shape[:2]
                                    writer_skel = cv2.VideoWriter(
                                        _filepath_skel, fourcc, RECORD_FPS, (w, h)
                                    )
                                    if not writer_skel.isOpened():
                                        print(f"[detector] WARNING: could not "
                                              f"open a video writer for "
                                              f"{_filepath_skel} -- no video "
                                              f"will be written for this "
                                              f"recording.")
                                    _write_queue_skel = queue.Queue()
                                    threading.Thread(
                                        target=_writer_worker,
                                        args=(writer_skel, _write_queue_skel),
                                        daemon=True,
                                    ).start()
                                # Both frames queued in the one locked block,
                                # so the two files always receive the same
                                # frames and stay aligned frame-for-frame --
                                # the actual (possibly slow) encode happens
                                # later, on each queue's own worker thread,
                                # never blocking this loop.
                                _write_queue.put(combined)
                                _write_queue_skel.put(skel)

                ok, buf = cv2.imencode(".jpg", frame, jpeg_params)

                # Scored after the encode so latency_ms covers the whole path
                # from frame arrival to a frame ready to send, and outside the
                # lock because add() only touches this run's own list.
                # Bind first: stop_recording() sets _run to None from the Flask
                # thread, and a check-then-call would occasionally hit None and
                # take the whole feed down for a reconnect. Adding a row to an
                # already-finished run is harmless — nothing reads it again.
                run = _run
                if run is not None and _metrics_mode:
                    run.add(result, time.perf_counter())

                if not ok:
                    continue
                _publish(buf.tobytes())

        except Exception as exc:  # noqa: BLE001 - keep the feed alive
            with _stream_lock:
                _stream_stats["last_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[detector] stream error: {exc!r}; reconnecting in 2s...")
        finally:
            with _stream_lock:
                _stream_stats["connected"] = False
            _close_stream(stream)

        # Doubles as the reconnect backoff: returns immediately once stopped.
        _worker_stop.wait(2.0)

    print("[detector] capture loop stopped")


def _watchdog_loop() -> None:
    """Force a reconnect on a wedged stream; hang up once nobody is watching."""
    while not _worker_stop.wait(1.0):
        now = time.time()
        with _stream_lock:
            connected = _stream_stats["connected"]
            last = _stream_stats["last_frame"]
            started = _stream_stats["started"]

        # Two shapes of the same failure, both of which otherwise block in
        # __next__ forever while the page freezes on its last frame — the worst
        # possible failure for a fall monitor, because a frozen feed looks
        # exactly like a quiet room.
        stalled = connected and last and now - last > STREAM_STALL_SEC
        never_started = (not connected and started
                         and now - started > STREAM_CONNECT_SEC)
        if stalled or never_started:
            waited = now - (last if stalled else started)
            print(f"[detector] "
                  f"{'no frames' if stalled else 'no first frame'} for "
                  f"{waited:.0f}s -- forcing a reconnect")
            with _stream_lock:
                _stream_stats["connected"] = False
                _stream_stats["started"] = now   # do not re-fire while it retries
            _close_stream(None)
            continue

        # Never hang up on work in progress.
        if fall_recorder.ENABLED or _recording:
            continue
        zero_since = _viewers_zero_since
        if _viewers == 0 and zero_since and now - zero_since > IDLE_STOP_SEC:
            print("[detector] no viewers -- dropping the RTSP connection")
            threading.Thread(target=stop_worker, daemon=True).start()
            return


def start_worker() -> None:
    """Start the shared capture thread unless it is already running."""
    global _worker
    with _worker_lock:
        if _worker is not None and _worker.is_alive():
            return
        _worker_stop.clear()
        _worker = threading.Thread(target=_capture_loop,
                                   name="detector-capture", daemon=True)
        _worker.start()
        threading.Thread(target=_watchdog_loop,
                         name="detector-watchdog", daemon=True).start()


def stop_worker(timeout: float = 10.0) -> None:
    """Stop the capture thread and close the RTSP session behind it."""
    global _worker
    with _worker_lock:
        worker, _worker = _worker, None
        if worker is None:
            return
        _worker_stop.set()
    # Unblocks the loop if it is sitting inside __next__ waiting on a socket.
    _close_stream(None)
    worker.join(timeout)


def _viewer_enter() -> None:
    global _viewers, _viewers_zero_since
    with _viewer_lock:
        _viewers += 1
        _viewers_zero_since = None
    start_worker()


def _viewer_leave() -> None:
    global _viewers, _viewers_zero_since
    with _viewer_lock:
        _viewers = max(0, _viewers - 1)
        if _viewers == 0:
            _viewers_zero_since = time.time()


def gen_frames():
    """Yield the shared annotated feed to one HTTP client as multipart MJPEG.

    Subscribes to the capture worker rather than opening a connection of its
    own, so the second and third viewer are free. Each viewer always waits for
    the NEWEST frame and never queues: a browser that cannot keep up drops to a
    lower frame rate on its own and the camera never hears about it.
    """
    _viewer_enter()
    seen = -1
    try:
        while True:
            with _frame_cv:
                # The timeout is what keeps this responsive while the stream is
                # down: no frames for 5s just means loop round and check again,
                # holding the HTTP response open so the <img> does not error
                # out and need a page reload once the camera comes back.
                if not _frame_cv.wait_for(lambda: _latest_seq != seen,
                                          timeout=5.0):
                    continue
                seen = _latest_seq
                jpeg = _latest_jpeg
            if jpeg:
                yield _BOUNDARY + jpeg + b"\r\n"
    finally:
        # GeneratorExit lands here when the tab closes. The worker deliberately
        # keeps running: other tabs may still be watching, and the watchdog
        # hangs up on its own once the last viewer has been gone a while.
        _viewer_leave()
