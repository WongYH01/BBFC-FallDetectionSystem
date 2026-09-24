"""Turn demo-cam v2's clip lifecycle events into spooled notifications.

`fall_recorder` publishes two facts (see demo-cam/v2/fall_recorder.py):

    clip_started  -- the alarm latched and a skeleton clip began recording
    clip_ready    -- the clip is complete and playable

The alert goes out on `clip_started`, so the notification is not delayed by
the 13 s the clip takes to record and convert; the clip is attached on
`clip_ready`. One event per clip: the recorder's own cooldown and the latch
hysteresis already collapse a single incident into one clip.

A restart between the two facts would lose the `clip_ready` (the deque lives
in memory), so `reconcile()` re-attaches any completed clip file whose event
is still pending. That is only safe because fall_recorder renames the file
into `fall_*.mp4` after it is complete -- a half-written clip exists only as
`fall_*.raw.mp4`.
"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

from .config import Settings
from .spool import Outbox


class Watcher:
    def __init__(self, settings: Settings, outbox: Outbox, pop_events):
        self.settings = settings
        self.outbox = outbox
        self.pop_events = pop_events
        self.last_event_at = 0.0

    # -- one event -----------------------------------------------------------

    def handle(self, event: dict) -> None:
        kind = event.get("kind")
        if kind == "clip_started":
            self._on_clip_started(event)
        elif kind == "clip_ready":
            self._on_clip_ready(event)
        else:
            print(f"[watcher] ignoring unknown event kind {kind!r}")

    def _on_clip_started(self, event: dict) -> None:
        now = time.time()
        cooldown = self.settings.event_cooldown_sec
        if cooldown and now - self.last_event_at < cooldown:
            print(f"[watcher] fall suppressed by EVENT_COOLDOWN_SEC="
                  f"{cooldown:g} (last alert {now - self.last_event_at:.0f}s ago)")
            return
        self.last_event_at = now

        event_id = uuid.uuid4().hex
        confidence = event.get("confidence")
        filename = event.get("filename")
        self.outbox.add_event(
            event_id=event_id,
            room_id=self.settings.room_id,
            room_name=self.settings.room_name,
            confidence=confidence,
            clip_filename=filename,
        )
        print(f"[watcher] fall latched -> event {event_id} "
              f"(P={confidence if confidence is not None else 'n/a'}, "
              f"clip {filename})")

    def _on_clip_ready(self, event: dict) -> None:
        path = event.get("filepath")
        filename = event.get("filename")
        if not path or not Path(path).exists():
            print(f"[watcher] clip_ready for {filename!r} but the file is "
                  f"missing at {path!r}; the alert stands, the clip is lost")
            closed = self.outbox.finish_clip_by_filename(filename)
            if closed:
                print(f"[watcher] closed event {closed} without a clip")
            return
        if not self.outbox.attach_clip_by_filename(filename, path):
            print(f"[watcher] clip_ready for {filename!r} matches no pending "
                  f"event (already uploaded, or suppressed)")

    # -- restart recovery ----------------------------------------------------

    def reconcile(self) -> int:
        """Attach completed clips to pending events that lost their `clip_ready`."""
        attached = 0
        for item in self.outbox.pending():
            if item.get("clip"):
                continue
            filename = item.get("clipFilename")
            if not filename:
                continue
            path = self.settings.clips_dir / filename
            if path.exists() and self.outbox.attach_clip(item["eventId"], str(path)):
                attached += 1
                print(f"[watcher] reconciled {filename} with event "
                      f"{item['eventId']}")
        return attached

    # -- loop ----------------------------------------------------------------

    def run(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            for event in self.pop_events():
                try:
                    self.handle(event)
                except Exception as exc:  # noqa: BLE001 - the loop must survive
                    print(f"[watcher] error handling {event!r}: {exc!r}")
            stop_event.wait(self.settings.poll_interval_sec)
