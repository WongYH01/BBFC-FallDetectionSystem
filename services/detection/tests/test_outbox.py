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
    def __init__(self, post_error=None, put_error=None, keypoints_errors=()):
        self.events: list[tuple] = []
        self.clips: list[tuple] = []
        self.keypoints: list[tuple] = []
        self.calls: list[str] = []
        self.post_error = post_error
        self.put_error = put_error
        # Raised by successive put_keypoints calls, then they succeed.
        self.keypoints_errors = list(keypoints_errors)

    def post_event(self, event_id, room_id, room_name, confidence):
        if self.post_error:
            raise self.post_error
        self.calls.append("event")
        self.events.append((event_id, room_id, room_name, confidence))
        return "accepted"

    def put_clip(self, event_id, path):
        if self.put_error:
            raise self.put_error
        self.calls.append("clip")
        self.clips.append((event_id, Path(path).name))
        return "stored"

    def put_keypoints(self, event_id, path):
        if self.keypoints_errors:
            raise self.keypoints_errors.pop(0)
        self.calls.append("keypoints")
        self.keypoints.append((event_id, Path(path).name))
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


class KeypointFileDeliveryTest(unittest.TestCase):
    """The keypoint file goes after the clip, and never at the clip's expense."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.clips = root / "clips"
        self.clips.mkdir()
        self.settings = make_settings(root / "spool", self.clips)
        self.outbox = Outbox(self.settings.spool_dir)
        self.outbox.add_event("e1", "r1", "Room 1", 0.9, "fall_1.mp4")
        self.clip = self.clips / "fall_1.mp4"
        self.clip.write_bytes(b"video-bytes")
        self.npz = self.clips / "fall_1.npz"
        self.npz.write_bytes(b"PK\x03\x04keypoints")

    def tearDown(self):
        self._tmp.cleanup()

    def _deliver(self, client, acked=None, outbox=None):
        outbox = outbox or self.outbox
        deliver_item(self.settings, client, outbox, outbox.pending()[0],
                     set() if acked is None else acked)

    def test_event_then_clip_then_keypoint_file(self):
        self.outbox.attach_files("e1", str(self.clip), str(self.npz))
        client = FakeClient()
        self._deliver(client)

        self.assertEqual(client.calls, ["event", "clip", "keypoints"])
        self.assertEqual(client.keypoints, [("e1", "fall_1.npz")])
        self.assertFalse(self.clip.exists())
        self.assertFalse(self.npz.exists())
        self.assertEqual(self.outbox.pending(), [])

    def test_waits_for_the_clip_to_finish(self):
        client = FakeClient()
        self._deliver(client)
        self.assertEqual(client.calls, ["event"])
        self.assertEqual(len(self.outbox.pending()), 1)

    def test_keypoint_file_alone_when_the_clip_was_lost(self):
        self.clip.unlink()
        self.outbox.attach_files("e1", None, str(self.npz))
        client = FakeClient()
        self._deliver(client)

        self.assertEqual(client.calls, ["event", "keypoints"])
        self.assertFalse(self.npz.exists())
        self.assertEqual(self.outbox.pending(), [])

    def test_a_keypoint_retry_does_not_resend_the_clip(self):
        self.outbox.attach_files("e1", str(self.clip), str(self.npz))
        client = FakeClient(keypoints_errors=[RetryableNotifyError("503")])
        acked: set[str] = set()
        with self.assertRaises(RetryableNotifyError):
            self._deliver(client, acked)

        # The clip went, and was cleaned up; the keypoint file waits.
        self.assertEqual(client.clips, [("e1", "fall_1.mp4")])
        self.assertFalse(self.clip.exists())
        self.assertTrue(self.npz.exists())
        self.assertTrue(self.outbox.pending()[0]["clipSent"])

        self._deliver(client, acked)
        self.assertEqual(client.calls, ["event", "clip", "keypoints"])
        self.assertEqual(self.outbox.pending(), [])

    def test_a_keypoint_retry_after_a_restart_does_not_resend_the_clip(self):
        self.settings = dataclasses.replace(self.settings,
                                            keep_uploaded_clips=True)
        self.outbox.attach_files("e1", str(self.clip), str(self.npz))
        with self.assertRaises(RetryableNotifyError):
            self._deliver(FakeClient(
                keypoints_errors=[RetryableNotifyError("timeout")]))

        # A new process: the journal is replayed, the in-memory acks are gone.
        # The clip is still on disk (KEEP_UPLOADED_CLIPS), so only the journal
        # can tell the worker it was already sent.
        client = FakeClient()
        self._deliver(client, outbox=Outbox(self.settings.spool_dir))
        self.assertEqual(client.calls, ["event", "keypoints"])

    def test_a_refused_keypoint_file_is_kept_for_inspection(self):
        self.outbox.attach_files("e1", str(self.clip), str(self.npz))
        client = FakeClient(keypoints_errors=[PermanentNotifyError("415")])
        # Propagates for the worker to drop, as any permanent failure does --
        # with the clip already delivered and removed.
        with self.assertRaises(PermanentNotifyError):
            self._deliver(client)
        self.assertEqual(client.clips, [("e1", "fall_1.mp4")])
        self.assertFalse(self.clip.exists())
        self.assertTrue(self.npz.exists())

    def test_a_missing_keypoint_file_does_not_hold_up_the_event(self):
        self.outbox.attach_files("e1", str(self.clip), str(self.npz))
        self.npz.unlink()
        client = FakeClient()
        self._deliver(client)
        self.assertEqual(client.calls, ["event", "clip"])
        self.assertEqual(self.outbox.pending(), [])

    def test_an_oversize_clip_still_sends_its_keypoint_file(self):
        self.settings = dataclasses.replace(self.settings, max_clip_bytes=4)
        self.outbox.attach_files("e1", str(self.clip), str(self.npz))
        client = FakeClient()
        self._deliver(client)
        self.assertEqual(client.calls, ["event", "keypoints"])
        self.assertTrue(self.clip.exists())
        self.assertEqual(self.outbox.pending(), [])

    def test_keep_uploaded_clips_keeps_the_keypoint_file_too(self):
        self.settings = dataclasses.replace(self.settings,
                                            keep_uploaded_clips=True)
        self.outbox.attach_files("e1", str(self.clip), str(self.npz))
        self._deliver(FakeClient())
        self.assertTrue(self.clip.exists())
        self.assertTrue(self.npz.exists())


if __name__ == "__main__":
    unittest.main()
