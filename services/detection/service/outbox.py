"""Deliver spooled notifications to the notification service.

The worker walks the outbox's pending items in order and, for each:

1. POSTs the event (idempotent -- a restart that re-POSTs gets 200 duplicate);
2. uploads the clip once it exists, then deletes the local file unless
   `KEEP_UPLOADED_CLIPS=1`.

Transient failures back off exponentially up to `NOTIFY_MAX_BACKOFF_SEC` and
retry forever: a fall alert is not something to give up on. Permanent failures
(400 invalid payload, 413 clip too large) are logged and dropped, with the
clip file kept for inspection. `deliver_item` is a plain function so tests can
drive it without the loop.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from .config import Settings
from .notifier import NotificationClient, PermanentNotifyError, RetryableNotifyError
from .spool import Outbox


def deliver_item(settings: Settings, client: NotificationClient, outbox: Outbox,
                 item: dict, event_acked: set[str]) -> None:
    """Deliver one pending item; raises NotifyError to have the caller retry.

    `event_acked` is in-memory only: after a restart the event is POSTed once
    more and the notification service answers 200 duplicate, which is cheap
    and keeps the worker from re-POSTing on every poll while it waits for the
    clip.
    """
    event_id = item["eventId"]

    if event_id not in event_acked:
        confidence = item.get("confidence")
        confidence = min(1.0, max(0.0, float(confidence if confidence is not None else 0.0)))
        status = client.post_event(event_id, item["roomId"], item["roomName"],
                                   confidence)
        event_acked.add(event_id)
        print(f"[outbox] event {event_id} {status} "
              f"({item['roomName']}, P={confidence:.3f})")

    path = item.get("clip")
    if not path:
        return

    clip = Path(path)
    if not clip.exists():
        print(f"[outbox] event {event_id}: clip {clip} is gone; the alert "
              f"was sent, dropping the clip")
        outbox.mark_done(event_id)
        return

    size = clip.stat().st_size
    if size > settings.max_clip_bytes:
        print(f"[outbox] event {event_id}: clip {clip.name} is {size} bytes, "
              f"over MAX_CLIP_BYTES={settings.max_clip_bytes} -- not "
              f"uploading it; the file is kept for inspection")
        outbox.mark_done(event_id)
        return

    status = client.put_clip(event_id, clip)
    print(f"[outbox] clip {clip.name} for {event_id} {status} ({size} bytes)")

    if not settings.keep_uploaded_clips:
        try:
            os.remove(clip)
        except OSError as exc:
            print(f"[outbox] could not remove {clip}: {exc}")

    outbox.mark_done(event_id)


def run_outbox(settings: Settings, client: NotificationClient, outbox: Outbox,
               stop_event: threading.Event) -> None:
    """The worker loop; one attempt per due item per poll."""
    event_acked: set[str] = set()
    failures: dict[str, tuple[int, float]] = {}  # eventId -> (attempts, next_ts)

    while not stop_event.is_set():
        pending = outbox.pending()
        # Forget the "already POSTed" markers of items that have completed, and
        # ONLY those: the marker has to outlive the passes where deliver_item
        # returns early because the clip is still recording, or the event is
        # re-POSTed on every poll (the notification service answers 200
        # duplicate, so this was invisible apart from the log).
        event_acked &= {i["eventId"] for i in pending}
        for item in pending:
            event_id = item["eventId"]
            attempts, next_ts = failures.get(event_id, (0, 0.0))
            if time.time() < next_ts:
                continue
            try:
                deliver_item(settings, client, outbox, item, event_acked)
                failures.pop(event_id, None)
            except PermanentNotifyError as exc:
                print(f"[outbox] dropping event {event_id}: {exc}")
                outbox.mark_done(event_id)
                failures.pop(event_id, None)
            except RetryableNotifyError as exc:
                attempts += 1
                delay = min(settings.notify_max_backoff_sec,
                            2.0 ** min(attempts, 6))
                failures[event_id] = (attempts, time.time() + delay)
                print(f"[outbox] event {event_id} attempt {attempts} failed, "
                      f"retrying in {delay:.0f}s: {exc}")
            except Exception as exc:  # noqa: BLE001 - the worker must survive
                # Anything the notifier does not classify: a spool this process
                # cannot write, a full disk, a bug in here. Before this existed
                # such an error escaped the loop and ended the thread, and since
                # the container's healthcheck only pings the Flask app, the
                # service went on reporting healthy while every later fall was
                # never delivered and never retried. A dead delivery worker must
                # be loud and temporary, not silent and permanent.
                attempts += 1
                delay = min(settings.notify_max_backoff_sec,
                            2.0 ** min(attempts, 6))
                failures[event_id] = (attempts, time.time() + delay)
                print(f"[outbox] event {event_id} attempt {attempts} hit an "
                      f"unexpected {type(exc).__name__}: {exc}; retrying in "
                      f"{delay:.0f}s")
        stop_event.wait(settings.poll_interval_sec)
