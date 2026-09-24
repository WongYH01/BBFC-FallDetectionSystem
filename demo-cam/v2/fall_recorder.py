"""Automatic pre-buffered clip capture on fall detection.

**Skeleton only.** The clips this module writes contain no camera imagery:
the frames handed to push_frame() are the skeleton canvas detector.py builds
(`result.plot(img=zeros, boxes=False, labels=False, conf=False)`), i.e. pose
drawn on a black or white background. Nothing here ever sees `orig_img`. The
manual Record button in detector.py is the one path that stores raw camera
footage (the side-by-side mp4), which is why the detection service sets
`ALLOW_MANUAL_RECORDING=0`.

Independent of that manual recorder. push_frame() is called once per frame
from gen_frames()'s single inference loop -- so it has to be cheap. Encoding a
frame into an mp4 (cv2.VideoWriter.write()) is not always instant, and if that
encode ran directly inside push_frame(), a slow one would stall the inference
loop, which stalls how fast frames get pulled off the RTSP source, which can
back up the network connection all the way to the Pi -- this is why the camera
was freezing shortly after a fall started recording when this used H.264
encoding (heavy enough in software to matter here; see the fourcc comment
in _open_writer() for the current codec choice and why).

So the actual cv2.VideoWriter.write() calls happen on a separate background
thread per clip, fed through a queue.Queue. push_frame() only ever does a
cheap queue.put() and returns immediately; the inference loop is never
blocked waiting for a frame to finish encoding.

The file only ever appears under its final `fall_<ts>.mp4` name once it is
complete: the writer works on `fall_<ts>.raw.mp4` and the finished file is
renamed into place after the ffmpeg conversion (or with the original mp4v
bytes if conversion is unavailable). A process that dies mid-clip therefore
leaves a `.raw.mp4` that nobody mistakes for a playable clip, which is what
lets the detection service reconcile clips across a restart. A writer that
fails to open produces no file at all -- the `clip_ready` event still fires,
and the service records the clip as lost rather than uploading an empty one.

EVENTS
------
The detection service needs to know when a clip starts (to raise the alert
event) and when it is finished (to upload it). Both are published as small
dicts on an internal deque that the service drains with `pop_events()`:

    {"kind": "clip_started", "filename", "filepath", "started_at", "confidence"}
    {"kind": "clip_ready",   "filename", "filepath", "ready_at"}

Events are appended under the module lock and read by another thread, so the
deque is the only shared state; callers that never call pop_events() (the
demo app) are unaffected apart from the deque filling to its bound.

FLOW
----
Every frame, gen_frames() hands push_frame() the skeleton-only canvas (pose
drawn on a blank background, no camera image -- same one detector.py already
builds for the manual-recording companion file), whether anyone is currently
latched as fallen, and the rolling decision score that latched it.

While idle, frames just accumulate in a ring buffer holding the last
FALL_CLIP_PRE_SECONDS of video (~3s by default). The buffer is trimmed by the
AGE of its frames, not by a frame count computed from RECORD_FPS: that count
assumed 10 fps, and at the 15.3 fps this actually runs at it bought 2.0 s of
lead-in instead of 3. For the same reason the mp4 is written -- and the
finished file restamped -- at the measured rate, not at RECORD_FPS, which was
stretching a 14 s incident into 22 s of slow-motion playback. The moment a fall
latches, a new clip file opens and a writer thread starts for it; the
buffered frames are hand off to that thread first -- this is the "3 seconds
before" part -- and then the triggering frame and every frame after it are
queued to it for a FIXED MAX_CLIP_SECONDS. This does NOT stop early just
because the fall clears: FALL_CLEAR_FRAMES clears in well under a second, so
stopping on "cleared" made every clip only ~1s of post-trigger footage no
matter what MAX_CLIP_SECONDS was set to. Total file length is therefore
PRE_EVENT_SECONDS + MAX_CLIP_SECONDS, always.

Once a clip closes, no new one can start for FALL_CLIP_COOLDOWN_SEC, AND not
until the alarm has been seen clear again. The cooldown alone covers a fall
that flickers across the latch/clear debounce; the clear is what covers a latch
that simply holds, which is the normal case -- someone lying on the floor
scores ~0.98 every window, so "latched and out of cooldown" was true forever and
produced a new clip, and a new alert, every 20 s until they got up.

A second fall therefore needs the first alarm to have cleared, which is the
same rule the notification service's escalation assumes: one event per
incident, escalated if nobody answers, rather than one event per 20 seconds.
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
# Only the fallback now: clips are written at the rate frames are actually
# pushed (`_measured_fps`), because this constant is a guess and the real rate
# depends on VID_STRIDE, the backbone and the machine.
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
# `need_skel` check in detector.py's gen_frames().
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

# Recent push timestamps, for the measured frame rate. The nominal RECORD_FPS
# is only a fallback now: the real rate is whatever the pose stage manages,
# which depends on VID_STRIDE, the backbone and the machine (measured 15.3/s on
# the deployment container against a nominal 10), and it is wrong in both
# directions to assume it.
_push_times: "deque" = deque(maxlen=120)

# Skeleton frames, oldest first, as (timestamp, frame). Trimmed by AGE rather
# than by count, so the lead-in is PRE_EVENT_SECONDS whatever the rate does;
# the maxlen is only a memory guard for an implausibly fast feed.
_buffer: "deque" = deque(maxlen=max(1, round(PRE_EVENT_SECONDS * 60)))

# Clip lifecycle facts, drained by the detection service (see module docstring).
# Bounded: a caller that never drains must not grow memory.
_events: "deque" = deque(maxlen=128)

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

#: False from the moment a clip starts until the alarm is seen clear again.
#: A clip needs a RISING edge, not merely "latched and out of cooldown": the
#: alarm holds for as long as the subject is down (a motionless body on the
#: floor scores ~0.98 every window), so without this one incident opened a new
#: clip -- and raised a new alert event -- every MAX_CLIP + COOLDOWN seconds
#: until they got up.
_armed = True
#: Per-clip counters, shared with that clip's writer thread so the finished
#: file can be stamped with the rate it was really captured at.
_clip_meta: dict | None = None


def _measured_fps() -> float | None:
    """Frames per second over the recent push history, or None if too few.

    Rate over a window rather than a smoothed 1/dt: an exponential average of
    instantaneous rates is biased upward by jitter (it is an arithmetic mean of
    reciprocals), which is exactly the error that would stretch a clip again.
    """
    if len(_push_times) < 15:
        return None
    span = _push_times[-1] - _push_times[0]
    if span <= 0:
        return None
    fps = (len(_push_times) - 1) / span
    return fps if 1.0 <= fps <= 120.0 else None


def _open_writer(size, fps: float):
    os.makedirs(CLIPS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fall_{ts}.mp4"
    filepath = os.path.join(CLIPS_DIR, filename)
    # The writer works on `<name>.raw.mp4`; the finished file is renamed into
    # the final name only after the encode is closed (and converted). A crash
    # mid-clip then leaves a `.raw.mp4` nobody reads, not a truncated
    # `fall_*.mp4` that the detection service would reconcile and upload.
    # The extension has to stay .mp4 -- OpenCV picks the container from it,
    # and a trailing `.raw` makes VideoWriter fail to open.
    rawpath = filepath[:-len(".mp4")] + ".raw.mp4"
    # "mp4v" -- back from H.264, which was heavy enough in software to be
    # part of why the camera stream was freezing during a recording (the
    # write() calls run on a background thread now -- see _writer_worker --
    # but the CPU cost was still real and still competed with everything
    # else on the machine). mp4v plays fine in VLC but not in Telegram
    # (shows a frozen first frame); the plan is to convert a finished clip
    # to something Telegram-friendly as a separate, decoupled step rather
    # than encoding it live.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(rawpath, fourcc, fps, size)
    if not writer.isOpened():
        print(f"[fall_recorder] WARNING: could not open a video writer for "
              f"{rawpath} -- no video will actually be written for "
              f"this clip.")
    return writer, filename, filepath, rawpath


def _writer_worker(writer, q: "queue.Queue", rawpath: str, filepath: str,
                   meta: dict) -> None:
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

    # The header was written at the rate estimated when the clip opened; by now
    # the whole clip has been seen, so the conversion is stamped with the rate
    # it was actually captured at.
    _convert_for_telegram(rawpath, filepath, meta.get("fps"))
    with _lock:
        _events.append({
            "kind": "clip_ready",
            "filename": os.path.basename(filepath),
            "filepath": filepath,
            "ready_at": time.time(),
        })
        if _state.get("last_clip_filename") == os.path.basename(filepath):
            _state["converting"] = False


def _convert_for_telegram(source: str, final: str,
                          fps: float | None = None) -> None:
    """Re-encode a just-closed mp4v clip to H.264/yuv420p with the moov atom
    moved to the front of the file (-movflags +faststart), then move it to
    its final name.

    This is what actually fixes Telegram (and most browsers/phones) showing
    the clip as a frozen first frame instead of playing it -- Telegram's
    player is strict about needing H.264 + yuv420p + a front-loaded moov
    atom, none of which OpenCV's mp4v output gives it.

    Runs on the writer thread, AFTER writer.release() has already flushed
    and closed the raw mp4v file -- so this never touches the live capture
    path. Worst case, encoding a ~10-15s low-motion skeleton clip with
    "veryfast" software libx264 takes a second or two, well after the fall
    is over; the RTSP read loop and gen_frames() are completely unaffected
    either way.

    `final` is written with os.replace() in every path that produces a file,
    so it is never observable in a half-written state. When conversion is
    unavailable (no ffmpeg, a crash, a non-zero exit) the raw mp4v bytes are
    moved to `final` anyway rather than dropped: Telegram may show a frozen
    frame, but the incident is still recorded. If the writer never opened,
    `source` does not exist and there is nothing to move.

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
    if not os.path.exists(source) or os.path.getsize(source) == 0:
        # The writer never opened (bad codec, no space, wrong size). There is
        # no clip to publish: leaving an empty file under the final name would
        # make the detection service upload a playable-looking 0-byte video.
        print(f"[fall_recorder] WARNING: no usable raw clip at {source} -- "
              f"the writer never wrote anything, so no clip was produced.")
        if os.path.exists(source):
            os.remove(source)
        return

    if shutil.which(FFMPEG_BIN) is None:
        print(f"[fall_recorder] WARNING: '{FFMPEG_BIN}' not found on PATH -- "
              f"keeping {os.path.basename(final)} in its original mp4v "
              f"format (Telegram will likely show it as a frozen frame). "
              f"Install ffmpeg to enable conversion.")
        os.replace(source, final)
        return

    tmp_path = final + ".converting.mp4"
    # `-r` BEFORE `-i` is an input option: it reinterprets the raw file's
    # timestamps at the true capture rate instead of dropping or duplicating
    # frames, so the clip plays back in real time and its duration is the
    # duration of the incident.
    rate = [] if not fps else ["-r", f"{fps:.3f}"]
    cmd = [
        FFMPEG_BIN, "-y",
        *rate, "-i", source,
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
              f"{os.path.basename(final)}: {exc}. Keeping the original mp4v "
              f"file as-is.")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        os.replace(source, final)
        return

    if result.returncode != 0 or not os.path.exists(tmp_path):
        print(f"[fall_recorder] WARNING: ffmpeg conversion failed for "
              f"{os.path.basename(final)} (exit {result.returncode}): "
              f"{result.stderr[-500:]}. Keeping the original mp4v file "
              f"as-is.")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        os.replace(source, final)
        return

    os.replace(tmp_path, final)
    os.remove(source)
    print(f"[fall_recorder] converted {os.path.basename(final)} for Telegram playback")


def _start_clip_locked(size, now: float, confidence: float | None = None) -> None:
    global _write_queue, _writer_started, _clip_meta, _armed
    _armed = False
    fps = _measured_fps() or float(FPS)
    writer, filename, filepath, rawpath = _open_writer(size, fps)
    q: "queue.Queue" = queue.Queue()
    # Pre-event flush: everything sitting in the ring buffer right now is the
    # PRE_EVENT_SECONDS immediately before this trigger frame. Handed to the
    # worker thread like any other frame, not written here directly, so this
    # (recording-start) call stays as cheap as every other push_frame() call.
    # The clip starts at the oldest buffered frame, not at the trigger: that is
    # the timestamp the achieved rate has to be measured from.
    first_ts = _buffer[0][0] if _buffer else now
    for _ts, buffered in _buffer:
        q.put(buffered)
    _clip_meta = {"frames": len(_buffer) + 1, "first_ts": first_ts,
                  "last_ts": now, "fps": fps}
    _buffer.clear()
    threading.Thread(target=_writer_worker,
                     args=(writer, q, rawpath, filepath, _clip_meta),
                     daemon=True).start()
    _write_queue = q
    _writer_started = now
    _state["recording"] = True
    _state["filename"] = filename
    _state["started_at"] = now
    _events.append({
        "kind": "clip_started",
        "filename": filename,
        "filepath": filepath,
        "started_at": now,
        "confidence": confidence,
    })


def _close_clip_locked(now: float) -> None:
    global _write_queue, _clip_meta
    if _clip_meta is not None:
        span = _clip_meta["last_ts"] - _clip_meta["first_ts"]
        if span > 0 and _clip_meta["frames"] > 1:
            # Written into the dict the worker thread already holds, before the
            # sentinel below tells it to finish -- the queue orders the two.
            _clip_meta["fps"] = (_clip_meta["frames"] - 1) / span
        _clip_meta = None
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


def push_frame(skel_frame, fallen: bool,
               confidence: float | None = None) -> None:
    """Call exactly once per inference frame with the skeleton-only canvas.

    `fallen` is whatever the caller's own latch says right now, e.g.
    ``any(p["fallen"] for p in people)`` from detector._update_falls().

    `confidence` is the rolling decision score that raised the latch; it is
    carried on the `clip_started` event so the detection service can put the
    number the model actually decided on into the alert. It never influences
    the recording itself.

    Never blocks on video encoding -- see the module docstring.
    """
    global _armed

    now = time.time()
    h, w = skel_frame.shape[:2]
    size = (w, h)

    with _lock:
        _push_times.append(now)
        if _state["recording"]:
            _write_queue.put(skel_frame)  # cheap: handed to the writer thread
            if _clip_meta is not None:
                _clip_meta["frames"] += 1
                _clip_meta["last_ts"] = now
            elapsed = now - _writer_started
            # Fixed duration -- deliberately ignores `fallen` clearing early.
            # See MAX_CLIP_SECONDS above for why.
            if elapsed >= MAX_CLIP_SECONDS:
                _close_clip_locked(now)
            return

        if fallen and _armed and now >= _state["cooldown_until"]:
            _start_clip_locked(size, now, confidence)
            _write_queue.put(skel_frame)  # the frame that tripped the trigger
            return

        if not fallen:
            # Cleared: the next latch is a new incident and may record again.
            _armed = True

        # Idle (or still cooling down): just keep the pre-event buffer fresh,
        # dropping whatever is older than the configured lead-in.
        _buffer.append((now, skel_frame))
        while _buffer and now - _buffer[0][0] > PRE_EVENT_SECONDS:
            _buffer.popleft()


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
            # What the clips are really being captured at, and how much
            # lead-in is buffered right now -- the two numbers that used to be
            # assumed from RECORD_FPS and were wrong by half.
            "measured_fps": (round(_measured_fps(), 1)
                             if _measured_fps() else None),
            "buffered_lead_sec": (round(now - _buffer[0][0], 1)
                                  if _buffer else 0.0),
        }


def pop_events() -> list:
    """Drain the clip lifecycle facts published since the last call.

    Non-blocking, thread-safe, and cheap enough for a poll loop. The demo app
    never calls this; the detection service polls it to raise alert events on
    `clip_started` and upload the file on `clip_ready`. See the module
    docstring for the shapes.
    """
    with _lock:
        events = list(_events)
        _events.clear()
    return events
