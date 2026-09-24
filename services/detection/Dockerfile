# Detection service: demo-cam v2 + the notification outbox.
#
# Build from the REPOSITORY ROOT (the compose file does this for you):
#   docker build -f services/detection/Dockerfile -t bbfc/detection .
#
# Weights are not baked in. The compose file bind-mounts
# demo-cam/models/ and runs/checkpoints/ read-only; fetch them on the host
# with scripts/fetch_weights.py first.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg is what fall_recorder.py shells out to for the Telegram-playable
# H.264 conversion of the skeleton clip; libgomp1 is ONNX Runtime's OpenMP
# runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch/torchvision first, from the CPU wheel index only. Installing them from
# PyPI would drag in the CUDA runtime for no reason: this image is CPU-only.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0 torchvision==0.28.0

COPY services/detection/requirements.txt /app/services/detection/requirements.txt
RUN pip install -r /app/services/detection/requirements.txt
# ultralytics depends on opencv-python, and both packages install into the same
# cv2/ directory: uninstalling just opencv-python deletes files the headless
# build also owns, leaving no importable cv2. Remove both, then reinstall the
# headless build so it is the only cv2 in the image.
RUN pip uninstall -y opencv-python opencv-python-headless \
    && pip install opencv-python-headless==5.0.0.93

# The detection stack and the service, exactly as the repo lays them out:
# fallcore at the root, demo-cam/v2 next to it, the service under services/.
COPY fallcore /app/fallcore
COPY demo-cam/v2 /app/demo-cam/v2
COPY services/detection/service /app/services/detection/service

# The app runs as a non-root user; the clip directory and the spool live on
# volumes. Every volume mountpoint must EXIST in the image and be owned by that
# user: Docker copies ownership from the image path when it initialises an empty
# named volume, and creates the mountpoint root-owned when the path is missing.
# /data alone is not enough -- the spool volume mounts at /data/outbox, so that
# is the path that has to be there, or the service cannot write its journal.
#
# The cache dirs are set because a system account has no writable HOME by
# default: without them Ultralytics tries to pip-install onnx on every start,
# fails, and matplotlib and fontconfig log errors for the same reason.
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib
RUN useradd -r -m -d /home/appuser -s /usr/sbin/nologin appuser \
    && mkdir -p /app/demo-cam/v2/fall_clips /app/demo-cam/v2/videos /data/outbox \
    && chown -R appuser:appuser /app/demo-cam /data /home/appuser
USER appuser

EXPOSE 5005

# /stream/status answers as long as the Flask app is up, camera or not.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5005/stream/status', timeout=4)" || exit 1

WORKDIR /app/services/detection
CMD ["python", "-m", "service.main"]
