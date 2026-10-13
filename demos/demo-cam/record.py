"""Browser recorder for the Pi camera: record the WebRTC preview itself.

    cd demo-cam
    .venv\\Scripts\\python.exe record.py        # then open http://localhost:5006

The page plays the camera over WebRTC (WHEP), the same player as app.py's
left pane, and records that stream in the browser with MediaRecorder; on stop
the clip is uploaded here and saved to demo-cam/recordings/<name>.mp4|.webm
(gitignored). Nothing is stored on the Pi and nothing runs there.

Why not RTSP: pulling RTSP to this PC over the internet link drops frames
(TCP head-of-line blocking -> MediaMTX discards for a slow reader -> "error
while decoding MB ..."), and those gaps end up in the file. WebRTC is built
for lossy links and stays smooth.

Trade-offs: the browser re-encodes (slightly below the camera's own H.264),
the frame rate is variable, and the tab must stay open while recording.
"""
from __future__ import annotations

import base64
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

STREAM_USERNAME = os.environ.get("STREAM_USERNAME", "")
STREAM_PASSWORD = os.environ.get("STREAM_PASSWORD", "")
MEDIAMTX_HOST = os.environ["MEDIAMTX_HOST"]
WEBRTC_PORT = os.environ.get("MEDIAMTX_WEBRTC_PORT", "8889")
STREAM_PATH = os.environ.get("STREAM_PATH", "cam")
# Clips are held in browser memory until stop, so cap their length.
MAX_CLIP_SEC = int(os.environ.get("RECORD_MAX_SEC", "1800"))
PORT = int(os.environ.get("RECORD_PORT", "5006"))

OUT_DIR = HERE / "recordings"
EXTENSIONS = {"mp4", "webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 ** 3


def safe_stem(raw: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip())
    stem = re.sub(r"\.(mp4|webm)$", "", stem, flags=re.IGNORECASE).strip("._")
    return stem or datetime.now().strftime("clip_%Y%m%d_%H%M%S")


def free_path(stem: str, ext: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{stem}.{ext}"
    if path.exists():
        path = OUT_DIR / f"{stem}_{datetime.now():%H%M%S}.{ext}"
    return path


@app.route("/")
def index():
    auth = None
    if STREAM_USERNAME:
        token = base64.b64encode(f"{STREAM_USERNAME}:{STREAM_PASSWORD}".encode()).decode()
        auth = f"Basic {token}"
    return render_template(
        "record.html",
        whep_url=f"http://{MEDIAMTX_HOST}:{WEBRTC_PORT}/{STREAM_PATH}/whep",
        auth_header=auth,
        out_dir=str(OUT_DIR),
        max_sec=MAX_CLIP_SEC,
    )


@app.post("/upload")
def upload():
    video = request.files.get("video")
    ext = request.form.get("ext", "")
    if video is None or ext not in EXTENSIONS:
        return jsonify(error="Missing video or unsupported format."), 400
    path = free_path(safe_stem(request.form.get("name", "")), ext)
    video.save(path)
    size = path.stat().st_size
    if size == 0:
        path.unlink()
        return jsonify(error="The recording was empty."), 400
    return jsonify(name=path.name, size_mb=round(size / 1e6, 1))


if __name__ == "__main__":
    print(f"recorder on http://localhost:{PORT}  (clips -> {OUT_DIR})")
    app.run(host="127.0.0.1", port=PORT, threaded=True, use_reloader=False)
