"""Automatic pre-buffered clip capture on fall detection.

Independent of the manual Record button (start_recording/stop_recording in
detector.py). push_frame() is called once per frame from the capture loop's
single inference loop -- so it has to be cheap. Encoding a frame into an
mp4 (cv2.VideoWriter.write()) is not always instant, and if that encode ran
directly inside push_frame(), a slow one would stall the inference loop,
which stalls how fast frames get pulled off the RTSP source, which can back
up the network connection all the way to the Pi -- this is why the camera
was freezing shortly after a fall started recording when this used H.264
encoding (heavy enough in software to matter here; see the fourcc comment
in _open_writer() for the current codec choice and why).

So the actual cv2.VideoWriter.write() calls happen on a separate background
thread per clip, fed through a queue.Queue. push_frame() only ever does a
cheap queue.put() and returns immediately; the inference loop is never
blocked waiting for a frame to finish encoding.

FLOW
----
Every frame, the shared capture loop hands push_frame() the skeleton-only canvas (pose
drawn on a blank background, no camera image -- same one detector.py already
builds for the manual-recording companion file) plus whether anyone is
currently latched as fallen.

While idle, frames just accumulate in a fixed-length ring buffer holding the
last FALL_CLIP_PRE_SECONDS of video (~3s by default). The moment a fall
latches, a new clip file opens and a writer thread starts for it; the
buffered frames are hand off to that thread first -- this is the "3 seconds
before" part -- and then the triggering frame and every frame after it are
queued to it for a FIXED MAX_CLIP_SECONDS. This does NOT stop early just
because the fall clears: FALL_CLEAR_FRAMES clears in well under a second, so
stopping on "cleared" made every clip only ~1s of post-trigger footage no
matter what MAX_CLIP_SECONDS was set to. Total file length is therefore
PRE_EVENT_SECONDS + MAX_CLIP_SECONDS, always.

Once a clip closes, no new one can start for FALL_CLIP_COOLDOWN_SEC. Without
this, a fall that flickers across the latch/clear debounce in detector.py
(or someone who falls again shortly after getting up) would produce a burst
of near-duplicate files instead of one clip per incident.
"""
import os
import time
import queue
import shutil
import threading
import subprocess
from collections import deque
from datetime import datetime

import cv2

CLIPS_DIR = os.path.join(os.path.dirname(__file__), "fall_clips")

# Same nominal FPS the manual recorder writes at (see RECORD_FPS in
# detector.py) -- kept as its own env read rather than importing detector's
# constant so this module has no import-order dependency on it.
FPS = int(os.environ.get("RECORD_FPS", "10"))

# Path/name of the ffmpeg binary used to convert a finished clip into a
# format Telegram (and most browsers) will actually play. Override if
# ffmpeg is not on PATH. See _convert_for_telegram() below.
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")

# --- the knobs you're most likely to want to adjust --------------------------
# Set to "0" to disable automatic capture entirely (the manual Record button's
# own skeleton file is unaffected). Useful when troubleshooting stream
# performance: disabling this skips an extra full result.plot() call on every
# single frame, not just while a clip is actually being written -- see the
# `need_skel` check in detector.py's _capture_loop().
ENABLED = os.environ.get("FALL_CLIPS_ENABLED", "1") not in ("0", "")

# Seconds of rolling pre-event footage kept before a fall and flushed into
# the front of every clip.
PRE_EVENT_SECONDS = float(os.environ.get("FALL_CLIP_PRE_SECONDS", "3"))

# vvv THE "10s buffer before another file can be generated" vvv
# Minimum gap between one clip finishing and the next one being allowed to
# start. Raise this if falls near the same spot are producing multiple
# clips; lower it if back-to-back incidents need to each get their own file.
COOLDOWN_SECONDS = float(os.environ.get("FALL_CLIP_COOLDOWN_SEC", "10"))

# How long to keep recording after a fall triggers. This is a FIXED
# duration, not a "stop early if things look clear" cap -- every clip runs
# for exactly this long after the trigger frame, whether or not the fall
# latch clears in the meantime (see push_frame()). Total file length is
# PRE_EVENT_SECONDS + this.
MAX_CLIP_SECONDS = float(os.environ.get("FALL_CLIP_MAX_SECONDS", "10"))
# ------------------------------------------------------------------------------

_buffer_len = max(1, round(PRE_EVENT_SECONDS * FPS))
_buffer: "deque" = deque(maxlen=_buffer_len)  # skeleton frames, oldest first

_lock = threading.Lock()
_state = {
    "recording": False,
    "converting": False,
    "filename": None,
    "started_at": None,
    "cooldown_until": 0.0,
    "last_clip_filename": None,
    "last_clip_at": None,
}

_write_queue: "queue.Queue | None" = None
_writer_started = 0.0


def _open_writer(size):
    os.makedirs(CLIPS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fall_{ts}.mp4"
    filepath = os.path.join(CLIPS_DIR, filename)
    # "mp4v" -- back from H.264, which was heavy enough in software to be
    # part of why the camera stream was freezing during a recording (the
    # write() calls run on a background thread now -- see _writer_worker --
    # but the CPU cost was still real and still competed with everything
    # else on the machine). mp4v plays fine in VLC but not in Telegram
    # (shows a frozen first frame); the plan is to convert a finished clip
    # to something Telegram-friendly as a separate, decoupled step rather
    # than encoding it live.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filepath, fourcc, FPS, size)
    if not writer.isOpened():
        print(f"[fall_recorder] WARNING: could not open a video writer for "
              f"{filepath} -- no video will actually be written for "
              f"this clip.")
    return writer, filename, filepath


def _writer_worker(writer, q: "queue.Queue", filepath: str) -> None:
    """Runs on its own thread for the life of one clip.

    Pulls frames off `q` and writes them -- this is where the (possibly
    slow) encode actually happens, off the inference thread. A `None` on the
    queue is the signal that the clip is done: drain stops, file closes.
    Once the raw file is closed, hand it to _convert_for_telegram() -- still
    on this same background thread, still nowhere near the inference loop.
    """
    while True:
        frame = q.get()
        if frame is None:
            break
        writer.write(frame)
    writer.release()

    _convert_for_telegram(filepath)
    with _lock:
        if _state.get("last_clip_filename") == os.path.basename(filepath):
            _state["converting"] = False


def _convert_for_telegram(filepath: str) -> None:
    """Re-encode a just-closed mp4v clip to H.264/yuv420p with the moov atom
    moved to the front of the file (-movflags +faststart), in place.

    This is what actually fixes Telegram (and most browsers/phones) showing
    the clip as a frozen first frame instead of playing it -- Telegram's
    player is strict about needing H.264 + yuv420p + a front-loaded moov
    atom, none of which OpenCV's mp4v output gives it.

    Runs on the writer thread, AFTER writer.release() has already flushed
    and closed the raw mp4v file -- so this never touches the live capture
    path. Worst case, encoding a ~10-15s low-motion skeleton clip with
    "veryfast" software libx264 takes a second or two, well after the fall
    is over; the RTSP read loop and the MJPEG fan-out are completely unaffected
    either way.

    Uses the ffmpeg CLI rather than re-opening the file with a second
    cv2.VideoWriter(*"avc1", ...): the whole reason this codebase is back on
    mp4v is that the OpenCV build on this machine apparently can't produce
    an H.264 stream Telegram accepts. A standalone ffmpeg binary commonly
    *does* support libx264 even when the paired cv2 wheel doesn't (H.264 is
    often left out of opencv-python wheels for licensing reasons, but the
    system ffmpeg binary is a separate build with its own codecs). ffmpeg is
    also simply better suited to this: -movflags +faststart isn't something
    cv2.VideoWriter can do at all, and one ffmpeg process is far faster than
    decoding every frame back out through cv2.VideoCapture and re-encoding
    it frame-by-frame in a Python loop.
    """
    if shutil.which(FFMPEG_BIN) is None:
        print(f"[fall_recorder] WARNING: '{FFMPEG_BIN}' not found on PATH -- "
              f"leaving {filepath} in its original mp4v format (Telegram "
              f"will likely show it as a frozen frame). Install ffmpeg "
              f"(e.g. `brew install ffmpeg` on macOS) to enable conversion.")
        return

    tmp_path = filepath + ".converting.mp4"
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", filepath,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        tmp_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as exc:
        print(f"[fall_recorder] WARNING: ffmpeg conversion crashed for "
              f"{filepath}: {exc}. Leaving original mp4v file as-is.")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return

    if result.returncode != 0 or not os.path.exists(tmp_path):
        print(f"[fall_recorder] WARNING: ffmpeg conversion failed for "
              f"{filepath} (exit {result.returncode}): "
              f"{result.stderr[-500:]}. Leaving original mp4v file as-is.")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return

    os.replace(tmp_path, filepath)  # atomic swap -- filename never changes
    print(f"[fall_recorder] converted {os.path.basename(filepath)} for Telegram playback")


def _start_clip_locked(size, now: float) -> None:
    global _write_queue, _writer_started
    writer, filename, filepath = _open_writer(size)
    q: "queue.Queue" = queue.Queue()
    # Pre-event flush: everything sitting in the ring buffer right now is the
    # PRE_EVENT_SECONDS immediately before this trigger frame. Handed to the
    # worker thread like any other frame, not written here directly, so this
    # (recording-start) call stays as cheap as every other push_frame() call.
    for buffered in _buffer:
        q.put(buffered)
    _buffer.clear()
    threading.Thread(target=_writer_worker, args=(writer, q, filepath), daemon=True).start()
    _write_queue = q
    _writer_started = now
    _state["recording"] = True
    _state["filename"] = filename
    _state["started_at"] = now


def _close_clip_locked(now: float) -> None:
    global _write_queue
    if _write_queue is not None:
        _write_queue.put(None)  # worker finishes draining, then releases
    _write_queue = None
    _state["recording"] = False
    _state["converting"] = True
    _state["cooldown_until"] = now + COOLDOWN_SECONDS
    _state["last_clip_filename"] = _state["filename"]
    _state["last_clip_at"] = now
    _state["filename"] = None
    _state["started_at"] = None


def push_frame(skel_frame, fallen: bool) -> None:
    """Call exactly once per inference frame with the skeleton-only canvas.

    `fallen` is whatever the caller's own latch says right now, e.g.
    ``any(p["fallen"] for p in people)`` from detector._update_falls().

    Never blocks on video encoding -- see the module docstring.
    """
    now = time.time()
    h, w = skel_frame.shape[:2]
    size = (w, h)

    with _lock:
        if _state["recording"]:
            _write_queue.put(skel_frame)  # cheap: handed to the writer thread
            elapsed = now - _writer_started
            # Fixed duration -- deliberately ignores `fallen` clearing early.
            # See MAX_CLIP_SECONDS above for why.
            if elapsed >= MAX_CLIP_SECONDS:
                _close_clip_locked(now)
            return

        if fallen and now >= _state["cooldown_until"]:
            _start_clip_locked(size, now)
            _write_queue.put(skel_frame)  # the frame that tripped the trigger
            return

        # Idle (or still cooling down): just keep the pre-event buffer fresh.
        _buffer.append(skel_frame)


def get_status() -> dict:
    with _lock:
        now = time.time()
        return {
            "recording": _state["recording"],
            "converting": _state["converting"],
            "filename": _state["filename"],
            "elapsed_sec": (round(now - _state["started_at"], 1)
                             if _state["recording"] else 0),
            "cooldown_remaining_sec": max(0.0, round(_state["cooldown_until"] - now, 1)),
            "last_clip_filename": _state["last_clip_filename"],
            "last_clip_at": _state["last_clip_at"],
            "pre_event_seconds": PRE_EVENT_SECONDS,
            "cooldown_seconds": COOLDOWN_SECONDS,
            "max_clip_seconds": MAX_CLIP_SECONDS,
        }
