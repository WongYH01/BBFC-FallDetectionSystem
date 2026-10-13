#!/usr/bin/env bash
# Restart mediamtx when the Pi's hardware H.264 encoder dies.
#
# The failure it watches for:
#
#   mediamtx[949]: encoder_hardware_h264_encode(): ioctl(VIDIOC_QBUF) failed
#
# repeating several times a second. That is the rpicamera helper unable to queue
# buffers to /dev/video11. Once it starts looping it does not recover on its own
# — the camera path stays down and every reader is dropped — and there is no
# upstream fix, so a restart is the honest answer. mediamtx comes back in about
# two seconds and clients reconnect by themselves.
#
# Deliberately tolerant: a handful of these can appear during a normal start and
# clear on their own, so it only acts on a burst inside a short window.
#
# Install:
#   sudo install -m 755 mediamtx-watchdog.sh /usr/local/bin/mediamtx-watchdog
#   sudo install -m 644 mediamtx-watchdog.service /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl enable --now mediamtx-watchdog.service
#
# Watch it:
#   journalctl -u mediamtx-watchdog.service -f

set -euo pipefail

UNIT="${UNIT:-mediamtx.service}"
PATTERN="${PATTERN:-VIDIOC_QBUF}"
THRESHOLD="${THRESHOLD:-20}"     # errors within WINDOW before restarting
WINDOW="${WINDOW:-10}"           # seconds
COOLDOWN="${COOLDOWN:-120}"      # seconds to wait after a restart

log() { logger -t mediamtx-watchdog "$*"; echo "$(date -Is) $*"; }

count=0
window_start=$(date +%s)
last_restart=0

log "watching $UNIT for '$PATTERN' (>=$THRESHOLD in ${WINDOW}s)"

# --since=now so a backlog of old errors in the journal does not trigger an
# immediate restart on boot.
journalctl -u "$UNIT" -f --since=now -o cat | while read -r line; do
  case "$line" in
    *"$PATTERN"*) ;;
    *) continue ;;
  esac

  now=$(date +%s)

  if (( now - window_start > WINDOW )); then
    window_start=$now
    count=0
  fi

  count=$((count + 1))
  (( count < THRESHOLD )) && continue

  if (( now - last_restart < COOLDOWN )); then
    log "threshold hit again but still inside the ${COOLDOWN}s cooldown; not restarting"
    count=0
    continue
  fi

  log "$count '$PATTERN' errors in ${WINDOW}s -- restarting $UNIT"
  systemctl restart "$UNIT" || log "restart FAILED"
  last_restart=$now
  count=0
  window_start=$now
done
