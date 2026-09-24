"""Outbox delivery: ordering, clip cleanup, acked events, permanent drops."""
from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from service.config import Settings
from service.notifier import PermanentNotifyError, RetryableNotifyError
from service.outbox import deliver_item
from service.spool import Outbox


class FakeClient:
    def __init__(self, post_error=None, put_error=None):
        self.events: list[tuple] = []
        self.clips: list[tuple] = []
        self.post_error = post_error
        self.put_error = put_error

    def post_event(self, event_id, room_id, room_name, confidence):
        if self.post_error:
            raise self.post_error
        self.events.append((event_id, room_id, room_name, confidence))
        return "accepted"

    def put_clip(self, event_id, path):
        if self.put_error:
            raise self.put_error
        self.clips.append((event_id, Path(path).name))
        return "stored"


def make_settings(tmp: Path, clips: Path, **overrides) -> Settings:
    base = Settings(
        room_id="r1", room_name="Room 1",
        notification_base_url="http://stub",
        notify_timeout_sec=1.0, notify_max_backoff_sec=1.0,
        spool_dir=tmp, poll_interval_sec=0.01, event_cooldown_sec=0.0,
        keep_uploaded_clips=False, max_clip_bytes=1 << 20,
        clips_dir=clips, reconcile_clips=True, autostart_detector=False,
        ui_host="127.0.0.1", ui_port=0,
    )
    return dataclasses.replace(base, **overrides)


class OutboxDeliveryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.clips = root / "clips"
        self.clips.mkdir()
        self.settings = make_settings(root / "spool", self.clips)
        self.outbox = Outbox(self.settings.spool_dir)
        self.outbox.add_event("e1", "r1", "Room 1", 0.9, "fall_1.mp4")

    def tearDown(self):
        self._tmp.cleanup()

    def _clip(self, name="fall_1.mp4", data=b"video-bytes") -> Path:
        path = self.clips / name
        path.write_bytes(data)
        self.outbox.attach_clip("e1", str(path))
        return path

    def test_full_delivery_posts_event_uploads_clip_deletes_file(self):
        clip = self._clip()
        client = FakeClient()
        item = self.outbox.pending()[0]
        deliver_item(self.settings, client, self.outbox, item, set())

        self.assertEqual(client.events, [("e1", "r1", "Room 1", 0.9)])
        self.assertEqual(client.clips, [("e1", "fall_1.mp4")])
        self.assertFalse(clip.exists())
        self.assertEqual(self.outbox.pending(), [])

    def test_event_is_posted_once_while_waiting_for_the_clip(self):
        client = FakeClient()
        acked: set[str] = set()
        item = self.outbox.pending()[0]
        deliver_item(self.settings, client, self.outbox, item, acked)
        deliver_item(self.settings, client, self.outbox, item, acked)
        self.assertEqual(len(client.events), 1)
        self.assertEqual(len(self.outbox.pending()), 1)

    def test_keep_uploaded_clips_leaves_the_file(self):
        self.settings = dataclasses.replace(self.settings,
                                            keep_uploaded_clips=True)
        clip = self._clip()
        deliver_item(self.settings, FakeClient(), self.outbox,
                     self.outbox.pending()[0], set())
        self.assertTrue(clip.exists())

    def test_oversize_clip_is_dropped_but_kept_on_disk(self):
        self.settings = dataclasses.replace(self.settings, max_clip_bytes=4)
        clip = self._clip(data=b"way too many bytes")
        client = FakeClient()
        deliver_item(self.settings, client, self.outbox,
                     self.outbox.pending()[0], set())
        self.assertEqual(client.events, [("e1", "r1", "Room 1", 0.9)])
        self.assertEqual(client.clips, [])
        self.assertTrue(clip.exists())
        self.assertEqual(self.outbox.pending(), [])

    def test_missing_clip_file_is_dropped_after_the_alert(self):
        self.outbox.attach_clip("e1", str(self.clips / "gone.mp4"))
        client = FakeClient()
        deliver_item(self.settings, client, self.outbox,
                     self.outbox.pending()[0], set())
        self.assertEqual(len(client.events), 1)
        self.assertEqual(client.clips, [])
        self.assertEqual(self.outbox.pending(), [])

    def test_retryable_failure_keeps_the_item_pending(self):
        self._clip()
        client = FakeClient(post_error=RetryableNotifyError("down"))
        with self.assertRaises(RetryableNotifyError):
            deliver_item(self.settings, client, self.outbox,
                         self.outbox.pending()[0], set())
        self.assertEqual(len(self.outbox.pending()), 1)

    def test_permanent_failure_propagates_for_the_worker_to_drop(self):
        self._clip()
        client = FakeClient(put_error=PermanentNotifyError("413"))
        with self.assertRaises(PermanentNotifyError):
            deliver_item(self.settings, client, self.outbox,
                         self.outbox.pending()[0], set())
        self.assertEqual(len(self.outbox.pending()), 1)

    def test_confidence_is_clamped(self):
        self.outbox.mark_done("e1")
        self.outbox.add_event("e2", "r1", "Room 1", 1.4, None)
        client = FakeClient()
        deliver_item(self.settings, client, self.outbox,
                     self.outbox.pending()[0], set())
        self.assertEqual(client.events, [("e2", "r1", "Room 1", 1.0)])


if __name__ == "__main__":
    unittest.main()
