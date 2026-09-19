import os
from flask import Flask, render_template, Response, jsonify, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Imported after load_dotenv() so the model + RTSP config read the .env values.
from detector import (  # noqa: E402
    gen_frames, start_recording, stop_recording, get_status, get_fall_status,
    get_fall_clip_status, get_stream_status,
)

# ---------------------------------------------------------------------------
# Config pulled entirely from .env
# ---------------------------------------------------------------------------
STREAM_USERNAME = os.environ.get("STREAM_USERNAME", "")
STREAM_PASSWORD = os.environ.get("STREAM_PASSWORD", "")
MEDIAMTX_HOST   = os.environ["MEDIAMTX_HOST"]
WEBRTC_PORT     = os.environ.get("MEDIAMTX_WEBRTC_PORT", "8889")
RTSP_PORT       = os.environ.get("MEDIAMTX_RTSP_PORT", "8554")
STREAM_PATH     = os.environ.get("STREAM_PATH", "cam")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        mediamtx_host=MEDIAMTX_HOST,
        webrtc_port=WEBRTC_PORT,
        rtsp_port=RTSP_PORT,
        stream_path=STREAM_PATH,
        stream_username=STREAM_USERNAME,
        stream_password=STREAM_PASSWORD,
    )

@app.route("/video_feed")
def video_feed():
    """MJPEG stream of the YOLO-pose annotated feed.

    Every client here shares ONE RTSP connection and ONE inference loop — see
    the "one reader, many viewers" note in detector.py. Opening this route in a
    second tab no longer costs the Pi a second reader.
    """
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stream/status")
def stream_status():
    """Health of the shared camera reader: fps, viewers, reconnects, last error.

    Check this first when the feed misbehaves. A climbing `reconnects` with no
    viewers joining or leaving means the camera path on the Pi is dropping —
    look at `journalctl -u mediamtx.service` there, not at this end. `age_sec`
    climbing past `stall_sec` means the watchdog is about to force a reconnect.
    """
    return jsonify(get_stream_status())

@app.route("/record/start", methods=["POST"])
def record_start():
    # {"metrics": true} starts a measurement run instead of a recording:
    # metrics/*.json + *.csv, and no video at all.
    payload = request.get_json(silent=True) or {}
    return jsonify(start_recording(metrics=bool(payload.get("metrics"))))

@app.route("/record/stop", methods=["POST"])
def record_stop():
    return jsonify(stop_recording())

@app.route("/record/status")
def record_status():
    return jsonify(get_status())

@app.route("/fall/status")
def fall_status():
    """Current fall state — polled by the sidebar alert panel."""
    return jsonify(get_fall_status())

@app.route("/fall_clips/status")
def fall_clips_status():
    """Status of the automatic pre-buffered fall-clip recorder."""
    return jsonify(get_fall_clip_status())

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # threaded=True so the page and the long-lived MJPEG connections coexist.
    #
    # use_reloader=False is not optional here. The reloader runs the whole app
    # in a child process while the parent has already imported this module —
    # which means torch and the pose weights loaded twice, and on a restart the
    # old process's capture thread can still be holding an RTSP session open
    # while the new one opens its own. Two readers on the Pi, for nothing.
    # Set FLASK_DEBUG=1 if you want the debugger; the auto-restart stays off.
    debug = os.environ.get("FLASK_DEBUG", "") not in ("", "0")
    app.run(host="0.0.0.0", port=5005, debug=debug, threaded=True,
            use_reloader=False)
