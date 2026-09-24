"""Detection service entry point.

Wires the unmodified demo-cam v2 app to the notification service:

* the shared capture thread is started at boot, not on the first viewer, so
  the room is watched whether or not anybody has the page open;
* `fall_recorder`'s clip lifecycle events feed the durable outbox;
* the outbox worker POSTs the alert and uploads the skeleton clip;
* the Flask UI stays available on SERVICE_UI_PORT for operators (the compose
  file publishes it on the host's loopback only -- it has no auth).

Run it as a module, from services/detection:

    python -m service.main

Environment: see .env.example. ROOM_ID and ROOM_NAME are required.
"""
from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

from .config import REPO_ROOT, SERVICE_DIR, V2_DIR, Settings
from .notifier import NotificationClient
from .outbox import run_outbox
from .spool import Outbox
from .watcher import Watcher


def _load_env() -> None:
    """Load .env files without overriding real environment variables.

    Compose passes the settings as environment variables; a local run reads
    services/detection/.env, falling back to demo-cam/.env for the stream and
    model settings shared with the demo.
    """
    load_dotenv(SERVICE_DIR / ".env")
    load_dotenv(REPO_ROOT / "demo-cam" / ".env")


def _import_v2(settings: Settings):
    """Import demo-cam v2's modules after the environment is in place.

    `detector` and `app` read the environment (and load the pose model) at
    import time, so this must not happen until `.env` is loaded.
    """
    # The repo root carries the `fallcore` package; demo-cam/v2 carries
    # `detector`, `fall_recorder` and `app`. Running as `python -m service.main`
    # from services/detection puts neither on sys.path.
    for path in (str(REPO_ROOT), str(V2_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    detector = importlib.import_module("detector")
    fall_recorder = importlib.import_module("fall_recorder")
    demo_app = importlib.import_module("app")
    return detector, fall_recorder, demo_app


def main() -> int:
    _load_env()
    settings = Settings.from_env()
    print("[service] detection service starting\n" + settings.describe())

    detector, fall_recorder, demo_app = _import_v2(settings)

    if not fall_recorder.ENABLED:
        # The capture watchdog drops the RTSP connection after IDLE_STOP_SEC
        # when no viewer is attached, unless the fall recorder is running.
        # With clips disabled the service would go blind with nobody watching.
        raise SystemExit(
            "FALL_CLIPS_ENABLED is off. The detection service needs it on: "
            "the automatic skeleton clip is the alert payload, and it is also "
            "what keeps the RTSP reader alive with no browser attached.")

    recorded_clips = Path(fall_recorder.CLIPS_DIR)
    if recorded_clips != settings.clips_dir:
        print(f"[service] WARNING: clips are written to {recorded_clips} but "
              f"CLIPS_DIR says {settings.clips_dir}; restart recovery will "
              f"look in the wrong place")

    outbox = Outbox(settings.spool_dir)
    client = NotificationClient(settings.notification_base_url,
                                settings.notify_timeout_sec)
    watcher = Watcher(settings, outbox, fall_recorder.pop_events)

    if settings.reconcile_clips:
        recovered = watcher.reconcile()
        if recovered:
            print(f"[service] recovered {recovered} clip(s) from before the restart")

    stop = threading.Event()
    threading.Thread(target=watcher.run, args=(stop,),
                     name="fall-watcher", daemon=True).start()
    threading.Thread(target=run_outbox, args=(settings, client, outbox, stop),
                     name="notify-outbox", daemon=True).start()

    if settings.autostart_detector:
        detector.start_worker()
        print("[service] capture thread started (always-on)")
    else:
        print("[service] DETECTOR_AUTOSTART=0 -- detection starts on first viewer")

    print(f"[service] UI on http://{settings.ui_host}:{settings.ui_port}")
    try:
        demo_app.app.run(host=settings.ui_host, port=settings.ui_port,
                         debug=False, threaded=True, use_reloader=False)
    finally:
        stop.set()
        detector.stop_worker()
    return 0


if __name__ == "__main__":
    sys.exit(main())
