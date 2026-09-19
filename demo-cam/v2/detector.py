"""Server-side pose tracking + rolling calibrated-model fall detection (v2).

Pulls frames from the MediaMTX RTSP feed, runs the frozen YOLO26-pose backbone
(the ONNX export by default, so ONNX Runtime carries it on CPU), tracks people
with ByteTrack, and draws the skeleton for the Flask `/video_feed` MJPEG route.

The fall alarm is the **calibrated model**: the largest person's keypoints fill a
4 s ring buffer, every second a window is scored by the backbone and read by a
logistic head fitted on THIS room's own clips
(`probe_augnone_ms_coords_hn_s99.npz`, scripts/fit_calibration_head.py), and that
score goes into a rolling buffer of `BUFFER_LEN` windows; the alarm latches when
the rolling mean clears `FALL_THRESHOLD`. `fallcore.stream` owns the arithmetic
and `fallcore.calibrate` owns the head. Held-out estimate on the camera's five
subjects: window AUC 0.992, rolling F1 0.967 (1 false alarm, 1 miss), against
0.836 for the best single checkpoint and 0.853 for the four-model ensemble.

`CALIBRATION_HEAD` empty falls back to the checkpoint's own classifier;
`ENSEMBLE_CKPTS` takes an os.pathsep list to run several checkpoints (the old
four-model mean) instead. Either way the head is bound to one backbone, and a
mismatch raises rather than returning confident nonsense.

The pose scale is part of that pairing: the head was fitted on yolo26n keypoints
at 640, so `POSE_SCALE=yolo26n` is the configuration that was measured. Another
scale feeds the head an input distribution it has never seen.

The v1 box width/height rule still runs, but only as a debug overlay
(`FALL_DEBUG=1`) -- the calibrated model decides.

Quick-and-dirty: models load once at import, one shared capture thread.
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
from fallcore import onnxpose as fonnx
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
#
# POSE_SCALE selects the backbone size (yolo26n/s/m/...). It is yolo26n because
# that is the scale the keypoint caches were extracted at and the scale the
# fitted head was trained against -- at any other scale the head reads an input
# distribution it never saw (and m costs ~83 ms/frame here against ~23 for n).
IMGSZ      = int(os.environ.get("POSE_IMGSZ", "640"))  # drop to 480 if slow
POSE_SCALE = os.environ.get("POSE_SCALE", "yolo26n")

# ONNX Runtime thread pool for the pose graph. Ultralytics builds its session
# with no session options at all -- every core, spinning between operators --
# which burns CPU without buying wall time: measured at 640 raw, 8.4 ms wall /
# 70 ms CPU per frame at the default, against 16.4 / 33 with two threads. 0
# leaves ONNX Runtime's default. Only applies to the ONNX backend.
POSE_THREADS = int(os.environ.get("POSE_THREADS", "2"))

REPO_ROOT  = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
_POSE_CANDIDATES = [
    MODELS_DIR / f"{POSE_SCALE}-pose.onnx",
    REPO_ROOT / "runs" / "onnx" / f"{POSE_SCALE}-pose-imgsz{IMGSZ}.onnx",
]
MODEL_NAME = os.environ.get("POSE_MODEL") or str(
    next((p for p in _POSE_CANDIDATES if p.exists()), _POSE_CANDIDATES[0]))
POSE_BACKEND = "onnx" if str(MODEL_NAME).lower().endswith(".onnx") else "torch"

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "videos")
RECORD_FPS = int(os.environ.get("RECORD_FPS", "10"))  # nominal, not measured

# Background of the skeleton-only companion recording. Black by default because
# the COCO keypoint palette is bright and reads best on a dark field; white is
# there for slides and printed reports, which usually want the opposite.
SKELETON_BG = os.environ.get("SKELETON_BG", "black")  # "black" | "white"

# ---------------------------------------------------------------------------
# The fall decision: one backbone + a head fitted on this room
# (fallcore.stream owns the arithmetic, fallcore.calibrate the head)
#
# The window is the training sampling: 60 rows 2 frames apart = 4 s. One new
# window per second; the rolling mean over the last BUFFER_LEN windows is the
# decision score, latched with hysteresis so it does not strobe as the buffer
# slides past the event.
#
# The default is the calibrated pair: `augnone_ms_coords_hn_s99.pt` (multi-scale
# pose augmentation, `coords` features, CAUCAFall hard negatives) with the
# logistic head fitted on this camera's own 60 clips. Held out on the camera's
# five subjects that is rolling F1 0.967 against 0.836 for the best single
# checkpoint and 0.853 for the four-model ensemble, at one forward pass per
# window instead of four (~0.7 ms against ~3.2 ms, and 6 MB of weights against
# 29). The head needs the room's own falls and ADLs -- it is a commissioning
# artefact, not a pretrained one, so a new room needs one fitted there.
# ---------------------------------------------------------------------------
_CKPT_NAMES = ("augnone_ms_coords_hn_s99.pt",)
_HEAD_NAME = "probe_augnone_ms_coords_hn_s99.npz"
_CKPT_DIR = REPO_ROOT / "runs" / "checkpoints"
ENSEMBLE_CKPTS = (
    [Path(p) for p in os.environ["ENSEMBLE_CKPTS"].split(os.pathsep)]
    if os.environ.get("ENSEMBLE_CKPTS")
    else [_CKPT_DIR / n for n in _CKPT_NAMES]
)
# One head per checkpoint, or None to use that checkpoint's own classifier.
# `CALIBRATION_HEAD=` (empty) in .env runs the checkpoints uncalibrated;
# pointing it at another room's probe is refused rather than silently applied,
# because a head is a function of one backbone's embedding space.
_CALIBRATION_HEAD = os.environ.get("CALIBRATION_HEAD",
                                   str(_CKPT_DIR / _HEAD_NAME)).strip()
HEADS = ([Path(_CALIBRATION_HEAD)] if _CALIBRATION_HEAD else
         [None] * len(ENSEMBLE_CKPTS))
BUFFER_LEN         = int(os.environ.get("BUFFER_LEN", "4"))
FALL_THRESHOLD     = float(os.environ.get("FALL_THRESHOLD", "0.5"))
FALL_CLEAR_BELOW   = float(os.environ.get("FALL_CLEAR_BELOW", "0.2"))
FALL_CLEAR_WINDOWS = int(os.environ.get("FALL_CLEAR_WINDOWS", "2"))

# Frames between the rows that build a classifier window. The default 2 matches
# training (every 2nd frame of ~30 fps = ~67 ms per row). If the pose backbone
# cannot run at ~30 fps and the loader is effectively decimating for you (the
# yolo26m CPU path runs ~14 fps, ~73 ms per row), set this to 1: the rows then
# arrive at roughly the trained spacing and a 60-row window still covers ~4 s.
# Leaving it at 2 there would stretch the window to ~9 s of wall clock.
STREAM_FRAME_STRIDE = int(os.environ.get(
    "STREAM_FRAME_STRIDE", str(fcfg.PREPROCESS.frame_stride)))

# Decode-and-score one frame in every VID_STRIDE. Ultralytics' loader still
# grabs every frame and only fully retrieves every Nth, so this buys CPU back
# WITHOUT making us a slow reader -- which is what MediaMTX calls a client that
# stops pulling frames off its socket. At 2, rows arrive at ~15 fps, exactly the
# spacing the caches were built with (every 2nd frame of 30 fps, ~67 ms), so
# pair it with STREAM_FRAME_STRIDE=1. Measured: posing every 2nd frame
# reproduces the full-rate rows bit-for-bit.
VID_STRIDE = max(1, int(os.environ.get("VID_STRIDE", "1")))

# Device for the pose model. Empty means "pick from the model": an ONNX export
# runs on ONNX Runtime, and asking Ultralytics for CUDA while the session is on
# CPU fails with "no data transfer registered for copying tensors from
# Device:[DeviceType:1] to Device:[DeviceType:0]" on the first frame. A GPU
# deployment with an onnxruntime-gpu build sets POSE_DEVICE=cuda.
POSE_DEVICE = os.environ.get("POSE_DEVICE", "")

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
# frame. Longer, because this window also covers Ultralytics building its
# predictor and the RTSP handshake itself.
STREAM_CONNECT_SEC = float(os.environ.get("STREAM_CONNECT_SEC",
                                          str(STREAM_STALL_SEC * 3)))

# Seconds after the last viewer leaves before the RTSP connection is dropped.
# Ignored while fall clips are enabled or a recording is running -- detection is
# the point, a browser being open is not.
IDLE_STOP_SEC = float(os.environ.get("IDLE_STOP_SEC", "30"))

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
_missing += [str(h) for h in HEADS if h is not None and not Path(h).exists()]
if _missing:
    raise FileNotFoundError(
        "model file(s) missing: " + ", ".join(_missing) +
        ". Run notebooks 02/10 to produce the checkpoints and "
        "scripts/fit_calibration_head.py for the head, or set "
        "ENSEMBLE_CKPTS / CALIBRATION_HEAD (empty = uncalibrated).")

if POSE_BACKEND == "onnx" and POSE_THREADS > 0:
    # Before the session exists: Ultralytics builds it with no session options,
    # so this is the only seam that sets the thread pool.
    fonnx.use_tuned_sessions(threads=POSE_THREADS, spinning=False)

model = YOLO(MODEL_NAME)
_stream = EnsembleStream(ENSEMBLE_CKPTS, heads=HEADS, buffer=BUFFER_LEN,
                         threshold=FALL_THRESHOLD, clear_below=FALL_CLEAR_BELOW,
                         clear_windows=FALL_CLEAR_WINDOWS,
                         frame_stride=STREAM_FRAME_STRIDE, device="cpu")
print(f"[detector] pose {Path(MODEL_NAME).name} ({POSE_BACKEND}, "
      f"threads={POSE_THREADS or 'ORT default'}), decision "
      f"{_stream.classifier.names} buffer={BUFFER_LEN} "
      f"threshold={FALL_THRESHOLD} frame_stride={STREAM_FRAME_STRIDE} "
      f"vid_stride={VID_STRIDE}")


# ---------------------------------------------------------------------------
# One reader, many viewers
#
# This used to be a plain generator, and Flask called it once per request. Every
# browser that opened /video_feed therefore got its OWN model.track(), which
# opened its OWN RTSP connection to MediaMTX and ran its OWN pose loop here. Two
# tabs meant two readers on the Pi and two inference loops fighting over the same
# CPU -- and a loop that is busy doing inference is a loop that is not draining
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
#
# Ported from the v1 fix (`fix stream getting frozen issue`, Fall-Recording),
# keeping v2's own decisions: ONNX/torch pose via `model.track`, the four-model
# ensemble, the recorder, the metrics tap and the fall-clip buffer.
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
    interest. `age_sec` over STREAM_STALL_SEC means the watchdog is about to
    force a reconnect.
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
    s["frame_stride"] = STREAM_FRAME_STRIDE
    s["jpeg_quality"] = JPEG_QUALITY
    s["pose_backend"] = POSE_BACKEND
    s["stall_sec"] = STREAM_STALL_SEC
    s["idle_stop_sec"] = IDLE_STOP_SEC
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
    through to its reconnect. That is exactly how the stall watchdog interrupts
    a wedged stream.
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


def _watchdog_loop() -> None:
    """Force a reconnect on a wedged stream; hang up once nobody is watching."""
    while not _worker_stop.wait(1.0):
        now = time.time()
        with _stream_lock:
            connected = _stream_stats["connected"]
            last = _stream_stats["last_frame"]
            started = _stream_stats["started"]

        # Two shapes of the same failure, both of which otherwise block in
        # __next__ forever while the page freezes on its last frame -- the worst
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


def _capture_loop() -> None:
    """The one thread that talks to the camera. Reconnects on anything.

    Everything v2 needs per frame happens here: the pose pass, the tap the
    metrics run reads, the ensemble decision, the drawings, the recorder queues
    and the publish into the viewer slot. This thread is also what drains the
    RTSP socket, so nothing slow may run inline -- the encoders stay on their own
    worker threads (see _writer_worker).
    """
    global _write_queue, _write_queue_skel
    rtsp_url = _rtsp_url()
    device = POSE_DEVICE or ("cpu" if POSE_BACKEND == "onnx" else None)
    track_kwargs = {"device": device} if device else {}
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
                **track_kwargs,
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
                    run.add(result, time.perf_counter(), fall_state=_last_state)

                if not ok:
                    continue
                _publish(buf.tobytes())

        except Exception as exc:  # noqa: BLE001 - keep the feed alive
            with _stream_lock:
                _stream_stats["last_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[detector] stream error: {exc!r}; reconnecting in 2s...")
            # The buffered keypoints span the outage, so the next window must
            # be built from fresh frames. A latched alarm survives -- an
            # outage is not evidence that the person got up.
            with _fall_lock:
                _stream.reset_buffer()
        finally:
            with _stream_lock:
                _stream_stats["connected"] = False
            _close_stream(stream)

        # Doubles as the reconnect backoff: returns immediately once stopped.
        _worker_stop.wait(2.0)

    print("[detector] capture loop stopped")
