"""Server-side pose tracking + rolling ensemble fall detection (v2).

Pulls frames from the MediaMTX RTSP feed, runs the frozen YOLO26-pose backbone
(the ONNX export by default, so ONNX Runtime carries it on CPU), tracks people
with ByteTrack, and draws the skeleton for the Flask `/video_feed` MJPEG route.

The fall alarm is the four-checkpoint ensemble from notebook 17: the largest
person's keypoints fill a 4 s ring buffer, every second a window is scored by
the four checkpoints (`final_yolo26n`, `hn10_full_hn_s99`, `hn10_coords_hn_s99`,
`omnifall_cs_full`), their mean goes into a rolling buffer of `BUFFER_LEN`
windows, and the alarm latches when the rolling mean clears `FALL_THRESHOLD`.
`fallcore.stream` owns that arithmetic; this file is the camera plumbing.

The v1 box width/height rule still runs, but only as a debug overlay
(`FALL_DEBUG=1`) -- the ensemble decides.

Quick-and-dirty: models load once at import, single shared generator.
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
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import fall_recorder
import metrics as _metrics
import numpy as np
from ultralytics import YOLO

from fallcore import config as fcfg
from fallcore import extract as fextract
from fallcore.stream import EnsembleStream

# ---------------------------------------------------------------------------
# Config — reuse the same env vars the Flask app already reads
# ---------------------------------------------------------------------------
STREAM_USERNAME = os.environ.get("STREAM_USERNAME", "")
STREAM_PASSWORD = os.environ.get("STREAM_PASSWORD", "")
MEDIAMTX_HOST   = os.environ["MEDIAMTX_HOST"]
RTSP_PORT       = os.environ.get("MEDIAMTX_RTSP_PORT", "8554")
STREAM_PATH     = os.environ.get("STREAM_PATH", "cam")

# Pose weights: the ONNX export by default (notebook 18), either the shared
# demo-cam/models copy or the one under the repo's runs/onnx/. Ultralytics
# dispatches .onnx to ONNX Runtime when it is installed; set POSE_MODEL to a
# .pt path to fall back to torch.
REPO_ROOT  = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
_POSE_CANDIDATES = [
    MODELS_DIR / "yolo26n-pose.onnx",
    REPO_ROOT / "runs" / "onnx" / "yolo26n-pose-imgsz640.onnx",
]
MODEL_NAME = os.environ.get("POSE_MODEL") or str(
    next((p for p in _POSE_CANDIDATES if p.exists()), _POSE_CANDIDATES[0]))
POSE_BACKEND = "onnx" if str(MODEL_NAME).lower().endswith(".onnx") else "torch"

IMGSZ      = int(os.environ.get("POSE_IMGSZ", "640"))  # drop to 480 if slow

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "videos")
RECORD_FPS = int(os.environ.get("RECORD_FPS", "10"))  # nominal, not measured

# Background of the skeleton-only companion recording. Black by default because
# the COCO keypoint palette is bright and reads best on a dark field; white is
# there for slides and printed reports, which usually want the opposite.
SKELETON_BG = os.environ.get("SKELETON_BG", "black")  # "black" | "white"

# ---------------------------------------------------------------------------
# The ensemble decision (fallcore.stream owns the arithmetic)
#
# The four checkpoints notebook 17 measured, all zero-shot on this camera. The
# window is the training sampling: 60 rows 2 frames apart = 4 s. One new window
# per second; the rolling mean over the last BUFFER_LEN windows is the decision
# score, latched with hysteresis so it does not strobe as the buffer slides
# past the event.
# ---------------------------------------------------------------------------
_CKPT_NAMES = ("final_yolo26n.pt", "hn10_full_hn_s99.pt",
               "hn10_coords_hn_s99.pt", "omnifall_cs_full.pt")
_CKPT_DIR = REPO_ROOT / "runs" / "checkpoints"
ENSEMBLE_CKPTS = (
    [Path(p) for p in os.environ["ENSEMBLE_CKPTS"].split(os.pathsep)]
    if os.environ.get("ENSEMBLE_CKPTS")
    else [_CKPT_DIR / n for n in _CKPT_NAMES]
)
BUFFER_LEN         = int(os.environ.get("BUFFER_LEN", "4"))
FALL_THRESHOLD     = float(os.environ.get("FALL_THRESHOLD", "0.5"))
FALL_CLEAR_BELOW   = float(os.environ.get("FALL_CLEAR_BELOW", "0.2"))
FALL_CLEAR_WINDOWS = int(os.environ.get("FALL_CLEAR_WINDOWS", "2"))

# Debug only: the v1 box width/height ratio, drawn next to each box when
# FALL_DEBUG=1. It no longer drives the alarm.
FALL_RATIO = float(os.environ.get("FALL_RATIO", "1.0"))
FALL_DEBUG = os.environ.get("FALL_DEBUG", "") not in ("", "0")

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
            _run = _metrics.Run(ts, _metrics.provenance_of(
                model, IMGSZ, pose_backend=POSE_BACKEND,
                ensemble=_stream.classifier.names, buffer=BUFFER_LEN,
                threshold=FALL_THRESHOLD,
                checkpoints=[str(p) for p in ENSEMBLE_CKPTS]))
            _metrics_mode = True
            _filename = _filename_skel = _filepath = _filepath_skel = None
            _write_queue = _write_queue_skel = None
            _start_time = time.time()
            _recording = True
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
        # Both opened lazily in gen_frames() once frame sizes are known — they
        # differ (2560x720 side-by-side vs 1280x720 skeleton).
        _write_queue = None
        _write_queue_skel = None
        _start_time = time.time()
        _recording = True
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
_people: list = []              # this frame's boxes, for the debug overlay
_last_state: dict = {}          # last ensemble state (probabilities, alarm)
_alarm_since: "float | None" = None
_last_update = 0.0              # wall clock of the last frame scored

_STALE_SECONDS = 1.5        # no frames for this long -> report nothing, not "clear"

_RED = (0, 0, 255)


def _largest_person_keypoints(result) -> np.ndarray:
    """This frame's subject as `(17, 3)` `(y, x, conf)`, cache convention.

    The same rule the training cache used: the biggest box is the subject, and
    a frame with no detection becomes zeros -- the classifier was trained with
    exactly those rows, so the absence is information, not something to hide.
    """
    kp = np.zeros((fcfg.N_KEYPOINTS, 3), dtype=np.float32)
    idx = fextract._largest_person(result)
    keypoints = getattr(result, "keypoints", None)
    if idx is not None and keypoints is not None and keypoints.data is not None \
            and len(keypoints.data) > idx:
        k = keypoints.data[idx].cpu().numpy()
        h, w = result.orig_shape
        kp[:, 0] = np.clip(k[:, 1] / max(h, 1), 0.0, 1.0)   # y first
        kp[:, 1] = np.clip(k[:, 0] / max(w, 1), 0.0, 1.0)   # then x
        kp[:, 2] = k[:, 2]
    return kp


def _boxes(result) -> list:
    """Per-box width/height ratios for the debug overlay; never the alarm."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or not len(boxes):
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else None
    out = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        h = float(y2 - y1)
        if h <= 0:                      # guards the division
            continue
        out.append({"track_id": ids[i] if ids else None,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "ratio": float((x2 - x1) / h)})
    return out


def _update_falls(result) -> list:
    """Feed one frame to the ensemble; returns this frame's people for drawing.

    The alarm state lives in `fallcore.stream`: the subject's keypoints go into
    the ring buffer, and on the frames a window is due the four checkpoints are
    scored and the rolling decision updates. `_last_state` carries the result to
    the drawing, the recorder and `/fall/status`.
    """
    global _people, _last_state, _alarm_since, _last_update
    kp = _largest_person_keypoints(result)
    with _fall_lock:
        if _stream.observe(kp):
            _last_state = _stream.state()
            if _last_state["alarm"]:
                if _alarm_since is None:
                    _alarm_since = time.time()
            else:
                _alarm_since = None
        _people = _boxes(result)
        _last_update = time.time()
    return _people


def _draw_falls(canvas, people, state) -> None:
    """Overlay the ensemble alert on top of result.plot()'s own annotations."""
    alarm = bool(state.get("alarm"))
    for p in people:
        x1, y1, x2, y2 = p["bbox"]
        if alarm:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), _RED, 3)
        if FALL_DEBUG:
            # Right-aligned at the box top — the one corner nothing else uses.
            txt = f"{p['ratio']:.2f}"
            (tw, _th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(canvas, txt, (max(x2 - tw - 4, 0), max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        _RED if p["ratio"] > FALL_RATIO else (0, 255, 255), 2,
                        cv2.LINE_AA)

    if alarm:
        _, w = canvas.shape[:2]
        cv2.rectangle(canvas, (0, 0), (w, 40), _RED, -1)
        # ASCII only. putText draws Hershey fonts, which have no glyph for an
        # emoji or an em dash and silently substitute '?'.
        label = (f"FALL DETECTED  P={state.get('p_fall', 0.0):.2f}  "
                 f"mean={state.get('rolling_mean', 0.0):.2f}")
        cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)


def get_fall_status() -> dict:
    with _fall_lock:
        # No frames recently means the feed died or nobody is watching it.
        # Report an empty scene rather than a stale one — the page shows "no
        # feed" for this, which is honest; showing the last known state would
        # let a dead detector read as a safe room.
        live = time.time() - _last_update < _STALE_SECONDS
        state = dict(_last_state)
        alarm = bool(state.get("alarm"))
        people = [{"track_id": p["track_id"], "ratio": round(p["ratio"], 2),
                   "fallen": alarm, "tracked": p["track_id"] is not None}
                  for p in _people] if live else []

        return {
            "live": live,
            "any_fallen": alarm,
            "count": 1 if alarm else 0,
            "tracked": len(people),
            "since_sec": (round(time.time() - _alarm_since, 1)
                          if _alarm_since and alarm else 0),
            "max_ratio": round(max((p["ratio"] for p in people), default=0.0), 2),
            "ratio_threshold": FALL_RATIO,
            "people": people,
            # -- the ensemble decision (v2) ---------------------------------
            "p_fall": round(float(state.get("p_fall", 0.0)), 4),
            "rolling_mean": round(float(state.get("rolling_mean", 0.0)), 4),
            "buffer_len": int(state.get("buffer_len", BUFFER_LEN)),
            "buffer_values": state.get("buffer_values", []),
            "n_windows": int(state.get("n_windows", 0)),
            "ready": bool(state.get("ready", False)),
            "buffered_frames": int(state.get("buffered_frames", 0)),
            "threshold": FALL_THRESHOLD,
            "clear_below": FALL_CLEAR_BELOW,
            "models": state.get("models") or _stream.classifier.names,
            "model_probs": state.get("model_probs", []),
            "pose_backend": POSE_BACKEND,
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


# Load the models once at import (shared across requests). The pose backbone
# is the only heavy one; the four classifiers are ~1 MB each and run once a
# second, so the ensemble is not where the CPU budget goes.
if MODEL_NAME.lower().endswith(".onnx") and not Path(MODEL_NAME).exists():
    raise FileNotFoundError(
        f"pose model {MODEL_NAME!r} not found. Export it with notebook 18 "
        f"(runs/onnx/) or copy it into demo-cam/models/, or set POSE_MODEL.")
_missing = [str(p) for p in ENSEMBLE_CKPTS if not Path(p).exists()]
if _missing:
    raise FileNotFoundError(
        "ensemble checkpoint(s) missing: " + ", ".join(_missing) +
        ". Run notebooks 02/10/14 to produce them, or set ENSEMBLE_CKPTS.")

model = YOLO(MODEL_NAME)
_stream = EnsembleStream(ENSEMBLE_CKPTS, buffer=BUFFER_LEN,
                         threshold=FALL_THRESHOLD, clear_below=FALL_CLEAR_BELOW,
                         clear_windows=FALL_CLEAR_WINDOWS, device="cpu")
print(f"[detector] pose {Path(MODEL_NAME).name} ({POSE_BACKEND}), "
      f"ensemble {_stream.classifier.names} buffer={BUFFER_LEN} "
      f"threshold={FALL_THRESHOLD}")


def gen_frames():
    """Yield annotated JPEG frames as multipart MJPEG.

    Uses Ultralytics streaming inference with the built-in ByteTrack tracker
    (`track(..., stream=True)`) so people keep stable IDs across frames.
    On any stream error we pause briefly and reconnect — good enough for a demo.
    """
    global _write_queue, _write_queue_skel
    rtsp_url = _rtsp_url()
    while True:
        try:
            for result in model.track(
                source=rtsp_url,
                stream=True,
                imgsz=IMGSZ,
                verbose=False,
            ):
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
                _draw_falls(frame, people, _last_state)

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
                    _draw_falls(skel, people, _last_state)  # second canvas

                    # Computed every frame (not just while manually recording)
                    # because fall_recorder needs a continuous skeleton feed to
                    # keep its rolling pre-event buffer current — see
                    # fall_recorder.py for the automatic capture logic.
                    if fall_recorder.ENABLED:
                        fall_recorder.push_frame(
                            skel, bool(_last_state.get("alarm")))

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

                ok, buf = cv2.imencode(".jpg", frame)

                # Scored after the encode so latency_ms covers the whole path
                # from frame arrival to a frame ready to send, and outside the
                # lock because add() only touches this run's own list.
                # Bind first: stop_recording() sets _run to None from the Flask
                # thread, and a check-then-call would occasionally hit None and
                # take the whole feed down for a reconnect. Adding a row to an
                # already-finished run is harmless — nothing reads it again.
                run = _run
                if run is not None and _metrics_mode:
                    run.add(result, time.perf_counter(), fall_state=_last_state)

                if not ok:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
        except Exception as exc:  # noqa: BLE001 - keep the feed alive
            print(f"[detector] stream error: {exc!r}; reconnecting in 2s...")
            # The buffered keypoints span the outage, so the next window must
            # be built from fresh frames. A latched alarm survives -- an
            # outage is not evidence that the person got up.
            with _fall_lock:
                _stream.reset_buffer()
            time.sleep(2)
