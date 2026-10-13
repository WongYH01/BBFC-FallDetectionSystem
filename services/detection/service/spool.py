"""Durable outbox: an append-only JSONL journal of fall notifications.

A fall is a safety event, so nothing about the alert path may be lost to a
restart. The journal records three facts, one JSON object per line:

    {"op": "event", "eventId", "roomId", "roomName", "confidence",
     "clipFilename", "createdAt"}
    {"op": "clip",  "eventId", "path"}
    {"op": "done",  "eventId"}

`event` is written the moment the alarm latches (the alert does not wait for
the clip); `clip` when the skeleton clip is complete; `done` after the
notification service has accepted both. Replaying the journal rebuilds the set
of unfinished items, so a restart between the latch and the upload loses
nothing. Re-POSTing an event is harmless -- the notification service dedups on
`eventId` and answers 200 -- which is why no "ack" fact is journalled.

The file is compacted when the done lines dominate, keeping the replay cheap
without ever rewriting a line that still matters.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

_JOURNAL_NAME = "outbox.jsonl"


class Outbox:
    """Thread-safe append-only journal plus the pending items derived from it.

    The watcher thread adds facts; the outbox worker reads `pending()` and
    marks items done. All mutation goes through the lock, so both threads can
    share one instance.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / _JOURNAL_NAME
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}
        self._lines = 0
        self._load()

    # -- journal -------------------------------------------------------------

    def _load(self) -> None:
        """Rebuild the pending items from the journal; skip damaged lines."""
        if not self.path.exists():
            return
        damaged = 0
        with open(self.path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                self._lines += 1
                try:
                    fact = json.loads(line)
                except json.JSONDecodeError:
                    damaged += 1
                    continue
                self._apply(fact)
        if damaged:
            print(f"[spool] skipped {damaged} damaged journal line(s) in "
                  f"{self.path}")
        if self._items:
            print(f"[spool] {len(self._items)} pending notification(s) "
                  f"recovered from {self.path}")

    def _apply(self, fact: dict) -> None:
        op, event_id = fact.get("op"), fact.get("eventId")
        if not op or not event_id:
            return
        if op == "event":
            self._items[event_id] = {
                "eventId": event_id,
                "roomId": fact.get("roomId"),
                "roomName": fact.get("roomName"),
                "confidence": fact.get("confidence"),
                "clipFilename": fact.get("clipFilename"),
                "createdAt": fact.get("createdAt", time.time()),
                "clip": None,
            }
        elif op == "clip" and event_id in self._items:
            self._items[event_id]["clip"] = fact.get("path")
        elif op == "done":
            self._items.pop(event_id, None)

    def _append_locked(self, fact: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(fact, separators=(",", ":")) + "\n")
        self._lines += 1

    def _compact_locked(self) -> None:
        """Rewrite the journal as just the pending items, atomically."""
        tmp = self.path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for item in self._items.values():
                fh.write(json.dumps({
                    "op": "event", "eventId": item["eventId"],
                    "roomId": item["roomId"], "roomName": item["roomName"],
                    "confidence": item["confidence"],
                    "clipFilename": item["clipFilename"],
                    "createdAt": item["createdAt"],
                }, separators=(",", ":")) + "\n")
                if item["clip"]:
                    fh.write(json.dumps({
                        "op": "clip", "eventId": item["eventId"],
                        "path": item["clip"],
                    }, separators=(",", ":")) + "\n")
        tmp.replace(self.path)
        self._lines = sum(2 if item["clip"] else 1 for item in self._items.values())

    # -- API -----------------------------------------------------------------

    def add_event(self, event_id: str, room_id: str, room_name: str,
                  confidence: float | None, clip_filename: str | None) -> None:
        with self._lock:
            self._items[event_id] = {
                "eventId": event_id,
                "roomId": room_id,
                "roomName": room_name,
                "confidence": confidence,
                "clipFilename": clip_filename,
                "createdAt": time.time(),
                "clip": None,
            }
            self._append_locked({
                "op": "event", "eventId": event_id,
                "roomId": room_id, "roomName": room_name,
                "confidence": confidence, "clipFilename": clip_filename,
                "createdAt": self._items[event_id]["createdAt"],
            })

    def attach_clip(self, event_id: str, path: str) -> bool:
        """Record the finished clip for an event; False if the event is unknown."""
        with self._lock:
            item = self._items.get(event_id)
            if item is None:
                return False
            item["clip"] = str(path)
            self._append_locked({"op": "clip", "eventId": event_id,
                                 "path": str(path)})
            return True

    def attach_clip_by_filename(self, filename: str, path: str) -> bool:
        """Attach a clip to the pending event that names it.

        The mapping survives restarts because `clipFilename` is journalled
        with the event, so a `clip_ready` event lost to a restart is recovered
        by reconciling the directory against the pending items.
        """
        with self._lock:
            for item in self._items.values():
                if item["clipFilename"] == filename and not item["clip"]:
                    item["clip"] = str(path)
                    self._append_locked({"op": "clip", "eventId": item["eventId"],
                                         "path": str(path)})
                    return True
            return False

    def finish_clip_by_filename(self, filename: str) -> str | None:
        """Close out the pending event for `filename` with no clip attached.

        Used when the recorder reports the clip finished but no file exists
        (the writer never opened). The alert has already gone out; the clip is
        lost, and holding the event pending forever would be a slow leak.
        Returns the eventId, or None if no such pending event.
        """
        with self._lock:
            for item in self._items.values():
                if item["clipFilename"] == filename and not item["clip"]:
                    self._items.pop(item["eventId"], None)
                    self._append_locked({"op": "done", "eventId": item["eventId"]})
                    if self._lines > max(64, 4 * len(self._items)):
                        self._compact_locked()
                    return item["eventId"]
            return None

    def mark_done(self, event_id: str) -> None:
        with self._lock:
            if self._items.pop(event_id, None) is None:
                return
            self._append_locked({"op": "done", "eventId": event_id})
            if self._lines > max(64, 4 * len(self._items)):
                self._compact_locked()

    def pending(self) -> list[dict]:
        """A snapshot of the unfinished items, oldest first."""
        with self._lock:
            return [dict(item) for item in self._items.values()]
