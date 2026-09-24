"""Service settings, all from the environment (see ../.env.example).

One instance of the service serves one room/camera. `ROOM_ID` and `ROOM_NAME`
are required rather than defaulted: an alert that reaches the notification
service with a placeholder room is worse than a service that refuses to start.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: services/detection -- the directory holding .env, the Dockerfile and tests.
SERVICE_DIR = Path(__file__).resolve().parents[1]
#: Repository root; the service reuses demo-cam/v2 and fallcore from here.
REPO_ROOT = Path(__file__).resolve().parents[3]
#: The v2 detection app (detector.py, fall_recorder.py, app.py).
V2_DIR = REPO_ROOT / "demo-cam" / "v2"
#: Where demo-cam/v2 writes the skeleton-only fall clips by default.
DEFAULT_CLIPS_DIR = V2_DIR / "fall_clips"


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(
            f"{name} is not set. Copy services/detection/.env.example to "
            f"services/detection/.env and fill it in.")
    return value


def _flag(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip() not in ("0", "false", "False")


@dataclass(frozen=True)
class Settings:
    # -- identity -----------------------------------------------------------
    room_id: str
    room_name: str

    # -- notification service ------------------------------------------------
    notification_base_url: str
    notify_timeout_sec: float
    notify_max_backoff_sec: float

    # -- outbox --------------------------------------------------------------
    spool_dir: Path
    poll_interval_sec: float
    event_cooldown_sec: float
    keep_uploaded_clips: bool
    max_clip_bytes: int

    # -- detection -----------------------------------------------------------
    clips_dir: Path
    reconcile_clips: bool
    autostart_detector: bool

    # -- ui ------------------------------------------------------------------
    ui_host: str
    ui_port: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            room_id=_require("ROOM_ID"),
            room_name=_require("ROOM_NAME"),
            notification_base_url=_env(
                "NOTIFICATION_BASE_URL",
                "http://notification:8081/api/v1/notification").rstrip("/"),
            notify_timeout_sec=float(_env("NOTIFY_TIMEOUT_SEC", "15")),
            notify_max_backoff_sec=float(_env("NOTIFY_MAX_BACKOFF_SEC", "60")),
            spool_dir=Path(_env("SPOOL_DIR", str(SERVICE_DIR / "spool"))),
            poll_interval_sec=float(_env("POLL_INTERVAL_SEC", "0.25")),
            event_cooldown_sec=float(_env("EVENT_COOLDOWN_SEC", "0")),
            keep_uploaded_clips=_flag("KEEP_UPLOADED_CLIPS", False),
            max_clip_bytes=int(_env("MAX_CLIP_BYTES", "8388608")),
            clips_dir=Path(_env("CLIPS_DIR", str(DEFAULT_CLIPS_DIR))),
            reconcile_clips=_flag("RECONCILE_CLIPS", True),
            autostart_detector=_flag("DETECTOR_AUTOSTART", True),
            ui_host=_env("SERVICE_UI_HOST", "127.0.0.1"),
            ui_port=int(_env("SERVICE_UI_PORT", "5005")),
        )

    def describe(self) -> str:
        """One block for the startup log; contains no secrets."""
        return (
            f"room={self.room_id!r} ({self.room_name!r})\n"
            f"notification={self.notification_base_url}\n"
            f"spool={self.spool_dir}\n"
            f"clips={self.clips_dir} (keep uploaded: {self.keep_uploaded_clips})\n"
            f"ui=http://{self.ui_host}:{self.ui_port}\n"
            f"autostart detector={self.autostart_detector} "
            f"reconcile clips={self.reconcile_clips}"
        )
