"""Watcher: clip lifecycle events -> outbox, plus restart reconciliation."""
from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from service.config import Settings
from service.spool import Outbox
from service.watcher import Watcher


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


class WatcherTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.spool = root / "spool"
        self.clips = root / "clips"
        self.clips.mkdir()
        self.outbox = Outbox(self.spool)

    def tearDown(self):
        self._tmp.cleanup()

    def _watcher(self, **overrides) -> Watcher:
        settings = make_settings(self.spool, self.clips, **overrides)
        return Watcher(settings, self.outbox, pop_events=list)

    def _outbox_reloaded(self) -> list:
        return Outbox(self.spool).pending()

    def _start(self, watcher: Watcher, filename="fall_1.mp4",
               confidence=0.91) -> str:
        watcher.handle({
            "kind": "clip_started", "filename": filename,
            "filepath": str(self.clips / filename), "started_at": 0.0,
            "confidence": confidence,
        })
        pending = self.outbox.pending()
        self.assertEqual(len(pending), 1)
        return pending[0]["eventId"]

    def test_clip_started_spools_the_event(self):
        event_id = self._start(self._watcher())
        item = self.outbox.pending()[0]
        self.assertTrue(event_id)
        self.assertEqual(item["roomId"], "r1")
        self.assertEqual(item["roomName"], "Room 1")
        self.assertEqual(item["confidence"], 0.91)
        self.assertEqual(item["clipFilename"], "fall_1.mp4")
        self.assertIsNone(item["clip"])

    def test_clip_ready_attaches_existing_file(self):
        watcher = self._watcher()
        event_id = self._start(watcher)
        clip = self.clips / "fall_1.mp4"
        clip.write_bytes(b"x")
        watcher.handle({"kind": "clip_ready", "filename": "fall_1.mp4",
                        "filepath": str(clip), "ready_at": 1.0})
        item = self.outbox.pending()[0]
        self.assertEqual(item["eventId"], event_id)
        self.assertEqual(item["clip"], str(clip))

    def test_clip_ready_with_missing_file_closes_the_event(self):
        # The writer failed to open: the alert stands, the clip is lost, and
        # the event must not stay pending forever.
        watcher = self._watcher()
        self._start(watcher)
        watcher.handle({"kind": "clip_ready", "filename": "fall_1.mp4",
                        "filepath": str(self.clips / "fall_1.mp4"),
                        "ready_at": 1.0})
        self.assertEqual(self.outbox.pending(), [])
        self.assertEqual(self._outbox_reloaded(), [])

    def test_clip_ready_without_event_is_ignored(self):
        watcher = self._watcher()
        clip = self.clips / "fall_orphan.mp4"
        clip.write_bytes(b"x")
        watcher.handle({"kind": "clip_ready", "filename": "fall_orphan.mp4",
                        "filepath": str(clip), "ready_at": 1.0})
        self.assertEqual(self.outbox.pending(), [])

    def test_cooldown_suppresses_a_second_event(self):
        watcher = self._watcher(event_cooldown_sec=3600.0)
        self._start(watcher, filename="fall_1.mp4")
        watcher.handle({
            "kind": "clip_started", "filename": "fall_2.mp4",
            "filepath": str(self.clips / "fall_2.mp4"), "started_at": 0.0,
            "confidence": 0.9,
        })
        self.assertEqual(len(self.outbox.pending()), 1)

    def test_reconcile_attaches_a_lost_ready_event(self):
        watcher = self._watcher()
        self._start(watcher, filename="fall_7.mp4")
        clip = self.clips / "fall_7.mp4"
        clip.write_bytes(b"x")
        self.assertEqual(watcher.reconcile(), 1)
        self.assertEqual(self.outbox.pending()[0]["clip"], str(clip))
        # Nothing left to do on a second pass.
        self.assertEqual(watcher.reconcile(), 0)

    def test_reconcile_ignores_half_written_raw_files(self):
        watcher = self._watcher()
        self._start(watcher, filename="fall_8.mp4")
        (self.clips / "fall_8.raw.mp4").write_bytes(b"partial")
        self.assertEqual(watcher.reconcile(), 0)
        self.assertIsNone(self.outbox.pending()[0]["clip"])

    # -- the keypoint file ---------------------------------------------------

    def _ready(self, watcher: Watcher, clip: bool = True,
               keypoints: bool = True, filename: str = "fall_1.mp4") -> None:
        path = self.clips / filename
        npz = path.with_suffix(".npz")
        if clip:
            path.write_bytes(b"x")
        if keypoints:
            npz.write_bytes(b"PK\x03\x04")
        watcher.handle({"kind": "clip_ready", "filename": filename,
                        "filepath": str(path),
                        "keypoints_filepath": str(npz), "ready_at": 1.0})

    def test_clip_ready_attaches_the_keypoint_file_with_the_clip(self):
        watcher = self._watcher()
        self._start(watcher)
        self._ready(watcher)
        [item] = self._outbox_reloaded()
        self.assertEqual(item["clip"], str(self.clips / "fall_1.mp4"))
        self.assertEqual(item["keypoints"], str(self.clips / "fall_1.npz"))

    def test_clip_ready_without_a_clip_still_sends_the_keypoint_file(self):
        # The video writer never opened, but the keypoints were recorded: the
        # event stays open to send them instead of being closed out.
        watcher = self._watcher()
        self._start(watcher)
        self._ready(watcher, clip=False)
        [item] = self._outbox_reloaded()
        self.assertIsNone(item["clip"])
        self.assertEqual(item["keypoints"], str(self.clips / "fall_1.npz"))

    def test_clip_ready_naming_a_missing_keypoint_file_sends_the_clip_alone(self):
        watcher = self._watcher()
        self._start(watcher)
        self._ready(watcher, keypoints=False)
        [item] = self._outbox_reloaded()
        self.assertEqual(item["clip"], str(self.clips / "fall_1.mp4"))
        self.assertIsNone(item["keypoints"])

    def test_clip_ready_with_neither_file_closes_the_event(self):
        watcher = self._watcher()
        self._start(watcher)
        self._ready(watcher, clip=False, keypoints=False)
        self.assertEqual(self._outbox_reloaded(), [])

    def test_reconcile_attaches_the_keypoint_file_with_its_clip(self):
        watcher = self._watcher()
        self._start(watcher, filename="fall_7.mp4")
        (self.clips / "fall_7.mp4").write_bytes(b"x")
        (self.clips / "fall_7.npz").write_bytes(b"PK\x03\x04")
        self.assertEqual(watcher.reconcile(), 1)
        [item] = self._outbox_reloaded()
        self.assertEqual(item["clip"], str(self.clips / "fall_7.mp4"))
        self.assertEqual(item["keypoints"], str(self.clips / "fall_7.npz"))
        self.assertEqual(watcher.reconcile(), 0)

    def test_reconcile_attaches_a_keypoint_file_whose_clip_was_lost(self):
        watcher = self._watcher()
        self._start(watcher, filename="fall_7.mp4")
        (self.clips / "fall_7.npz").write_bytes(b"PK\x03\x04")
        self.assertEqual(watcher.reconcile(), 1)
        [item] = self._outbox_reloaded()
        self.assertIsNone(item["clip"])
        self.assertEqual(item["keypoints"], str(self.clips / "fall_7.npz"))

    def test_reconcile_ignores_a_half_written_keypoint_file(self):
        watcher = self._watcher()
        self._start(watcher, filename="fall_8.mp4")
        (self.clips / "fall_8.raw.mp4").write_bytes(b"partial")
        (self.clips / "fall_8.npz.tmp").write_bytes(b"PK\x03")
        self.assertEqual(watcher.reconcile(), 0)
        [item] = self.outbox.pending()
        self.assertIsNone(item["clip"])
        self.assertIsNone(item["keypoints"])

    def test_unknown_event_kind_is_ignored(self):
        watcher = self._watcher()
        watcher.handle({"kind": "something_else"})
        self.assertEqual(self.outbox.pending(), [])


if __name__ == "__main__":
    unittest.main()
