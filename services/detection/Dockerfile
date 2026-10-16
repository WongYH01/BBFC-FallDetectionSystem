# Detection service: v2 + the notification outbox.
#
# Build from services/detection (the root docker-compose.yml does this for you):
#   docker build --secret id=detection_env,src=.env -t bbfc/detection .
#
# The weights are baked in: the build fetches them itself, so nothing has to be
# on the host. .env must hold FALL_WEIGHTS_TOKEN (see .env.example).

# The three deployment files, fetched by the same script a host run uses, so the
# pinned revision and the SHA-256 checks live in one place. .env reaches this
# step as a BuildKit secret: the token is never in a layer, the image history or
# the final image. A rebuild for a code change reuses this stage's cache and
# does not download again.
FROM python:3.12-slim AS weights
ENV PIP_NO_CACHE_DIR=1     PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install huggingface_hub==1.29.0
COPY scripts/fetch_weights.py /fetch/fetch_weights.py
RUN --mount=type=secret,id=detection_env,required=true     python /fetch/fetch_weights.py --root /weights         --env-file /run/secrets/detection_env

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

COPY requirements.txt /app/requirements.txt
# ultralytics depends on opencv-python, and both packages install into the same
# cv2/ directory: uninstalling just opencv-python deletes files the headless
# build also owns, leaving no importable cv2. Remove both, then reinstall the
# headless build so it is the only cv2 in the image. Same RUN as the install, so
# the discarded opencv-python files never reach a layer.
RUN pip install -r /app/requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install opencv-python-headless==5.0.0.93

# Trivy flags the setuptools, wheel and jaraco.context that python:3.12-slim ships
# (HIGH, fixable). Upgrade them past the fixed versions.
RUN pip install --upgrade "setuptools>=78.1.1" "wheel>=0.46.2" "jaraco.context>=6.1.0"

# The detection stack and the service, laid out as in services/detection:
# fallcore, v2 and service side by side under /app, with the weights at the
# paths v2 resolves by default (models/, runs/checkpoints/).
COPY --from=weights /weights/models /app/models
COPY --from=weights /weights/runs /app/runs
COPY fallcore /app/fallcore
COPY v2 /app/v2
COPY service /app/service

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
    && mkdir -p /app/v2/fall_clips /app/v2/videos /data/outbox \
    && chown -R appuser:appuser /app/v2 /data /home/appuser
USER appuser

EXPOSE 5005

# /stream/status answers as long as the Flask app is up, camera or not.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5005/stream/status', timeout=4)" || exit 1

CMD ["python", "-m", "service.main"]
