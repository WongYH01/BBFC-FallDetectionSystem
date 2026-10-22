"""Durable outbox: an append-only JSONL journal of fall notifications.

A fall is a safety event, so nothing about the alert path may be lost to a
restart. The journal records these facts, one JSON object per line:

    {"op": "event",     "eventId", "roomId", "roomName", "confidence",
     "clipFilename", "createdAt"}
    {"op": "keypoints", "eventId", "path"}
    {"op": "clip",      "eventId", "path"}
    {"op": "clip_sent", "eventId"}
    {"op": "done",      "eventId"}

`event` is written the moment the alarm latches (the alert does not wait for
the clip); `keypoints` and `clip` together when the skeleton clip and its
keypoint file are complete, either of which may be missing; `clip_sent` once
the clip is uploaded, so a retry that is only waiting on the keypoint file
does not send the clip again; `done` after the notification service has
accepted everything. Replaying the journal rebuilds the set of unfinished
items, so a restart between the latch and the upload loses nothing.
Re-POSTing an event is harmless -- the notification service dedups on
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
            self._items[event_id] = _new_item(
                event_id, fact.get("roomId"), fact.get("roomName"),
                fact.get("confidence"), fact.get("clipFilename"),
                fact.get("createdAt", time.time()))
        elif op in ("clip", "keypoints") and event_id in self._items:
            self._items[event_id][op] = fact.get("path")
        elif op == "clip_sent" and event_id in self._items:
            self._items[event_id]["clipSent"] = True
        elif op == "done":
            self._items.pop(event_id, None)

    def _append_locked(self, fact: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(fact, separators=(",", ":")) + "\n")
        self._lines += 1

    def _compact_locked(self) -> None:
        """Rewrite the journal as just the pending items, atomically."""
        tmp = self.path.with_suffix(".jsonl.tmp")
        lines = 0
        with open(tmp, "w", encoding="utf-8") as fh:
            for item in self._items.values():
                for fact in _facts(item):
                    fh.write(json.dumps(fact, separators=(",", ":")) + "\n")
                    lines += 1
        tmp.replace(self.path)
        self._lines = lines

    # -- API -----------------------------------------------------------------

    def add_event(self, event_id: str, room_id: str, room_name: str,
                  confidence: float | None, clip_filename: str | None) -> None:
        with self._lock:
            item = _new_item(event_id, room_id, room_name, confidence,
                             clip_filename, time.time())
            self._items[event_id] = item
            self._append_locked(_facts(item)[0])

    def attach_files(self, event_id: str, clip: str | None,
                     keypoints: str | None) -> bool:
        """Record what a finished clip left for an event: the clip, its
        keypoint file, or both. False if the event is unknown.

        An attached file is what tells the outbox worker the event is ready
        to finish, so the two must arrive together: they are recorded under
        one lock, which the worker's `pending()` also takes, and the keypoint
        line is journalled first, so a replay of a journal cut off between
        the two lines does not see the clip alone either. Either may be None
        (a clip whose writer never opened still has keypoints), not both.
        """
        with self._lock:
            item = self._items.get(event_id)
            if item is None:
                return False
            self._attach_locked(item, clip, keypoints)
            return True

    def attach_files_by_filename(self, filename: str, clip: str | None,
                                 keypoints: str | None) -> bool:
        """`attach_files` for the pending event that names the clip `filename`.

        The mapping survives restarts because `clipFilename` is journalled
        with the event, so a `clip_ready` event lost to a restart is recovered
        by reconciling the directory against the pending items.
        """
        with self._lock:
            item = self._waiting_locked(filename)
            if item is None:
                return False
            self._attach_locked(item, clip, keypoints)
            return True

    def attach_clip(self, event_id: str, path: str) -> bool:
        """Record the finished clip for an event; False if the event is unknown."""
        return self.attach_files(event_id, path, None)

    def attach_clip_by_filename(self, filename: str, path: str) -> bool:
        """Attach a clip to the pending event that names it."""
        return self.attach_files_by_filename(filename, path, None)

    def mark_clip_sent(self, event_id: str) -> None:
        """The clip is uploaded; only the keypoint file may be left to send."""
        with self._lock:
            item = self._items.get(event_id)
            if item is None or item["clipSent"]:
                return
            item["clipSent"] = True
            self._append_locked({"op": "clip_sent", "eventId": event_id})

    def finish_clip_by_filename(self, filename: str) -> str | None:
        """Close out the pending event for `filename` with no clip attached.

        Used when the recorder reports the clip finished but left no file at
        all (the writer never opened, and no keypoint file either). The alert
        has already gone out; the clip is lost, and holding the event pending
        forever would be a slow leak. Returns the eventId, or None if no such
        pending event.
        """
        with self._lock:
            item = self._waiting_locked(filename)
            if item is None:
                return None
            self._items.pop(item["eventId"], None)
            self._append_locked({"op": "done", "eventId": item["eventId"]})
            if self._lines > max(64, 4 * len(self._items)):
                self._compact_locked()
            return item["eventId"]

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

    # -- helpers (lock held) -------------------------------------------------

    def _waiting_locked(self, filename: str) -> dict | None:
        """The pending event for clip `filename` that has no files yet."""
        for item in self._items.values():
            if item["clipFilename"] == filename and not item["clip"] \
                    and not item["keypoints"]:
                return item
        return None

    def _attach_locked(self, item: dict, clip: str | None,
                       keypoints: str | None) -> None:
        if clip is None and keypoints is None:
            raise ValueError("attach a clip, a keypoint file or both")
        for op, path in (("keypoints", keypoints), ("clip", clip)):
            if path is not None:
                item[op] = str(path)
                self._append_locked({"op": op, "eventId": item["eventId"],
                                     "path": str(path)})


def _new_item(event_id: str, room_id: str | None, room_name: str | None,
              confidence: float | None, clip_filename: str | None,
              created_at: float) -> dict:
    return {
        "eventId": event_id,
        "roomId": room_id,
        "roomName": room_name,
        "confidence": confidence,
        "clipFilename": clip_filename,
        "createdAt": created_at,
        "clip": None,
        "keypoints": None,
        "clipSent": False,
    }


def _facts(item: dict) -> list[dict]:
    """The journal lines that rebuild `item`, in the order they apply."""
    facts = [{
        "op": "event", "eventId": item["eventId"],
        "roomId": item["roomId"], "roomName": item["roomName"],
        "confidence": item["confidence"],
        "clipFilename": item["clipFilename"],
        "createdAt": item["createdAt"],
    }]
    for op in ("keypoints", "clip"):
        if item[op]:
            facts.append({"op": op, "eventId": item["eventId"], "path": item[op]})
    if item["clipSent"]:
        facts.append({"op": "clip_sent", "eventId": item["eventId"]})
    return facts
