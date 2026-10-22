"""Outbox journal: replay, durability and compaction."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from service.spool import Outbox


class SpoolTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _outbox(self) -> Outbox:
        return Outbox(self.dir)

    def test_event_clip_done_roundtrip(self):
        outbox = self._outbox()
        outbox.add_event("e1", "r1", "Room 1", 0.87, "fall_1.mp4")
        pending = outbox.pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["eventId"], "e1")
        self.assertEqual(pending[0]["roomId"], "r1")
        self.assertEqual(pending[0]["confidence"], 0.87)
        self.assertEqual(pending[0]["clipFilename"], "fall_1.mp4")
        self.assertIsNone(pending[0]["clip"])

        self.assertTrue(outbox.attach_clip("e1", "C:/clips/fall_1.mp4"))
        self.assertEqual(outbox.pending()[0]["clip"], "C:/clips/fall_1.mp4")

        outbox.mark_done("e1")
        self.assertEqual(outbox.pending(), [])

    def test_pending_survives_reload(self):
        outbox = self._outbox()
        outbox.add_event("e1", "r1", "Room 1", 0.5, "fall_1.mp4")
        outbox.attach_clip("e1", "/clips/fall_1.mp4")

        reloaded = self._outbox()
        pending = reloaded.pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["clip"], "/clips/fall_1.mp4")
        self.assertEqual(pending[0]["clipFilename"], "fall_1.mp4")

    def test_done_is_not_recovered(self):
        outbox = self._outbox()
        outbox.add_event("e1", "r1", "Room 1", 0.5, "fall_1.mp4")
        outbox.mark_done("e1")
        self.assertEqual(self._outbox().pending(), [])

    def test_unknown_clip_is_ignored_on_replay(self):
        outbox = self._outbox()
        self.assertFalse(outbox.attach_clip("nope", "/clips/x.mp4"))
        self.assertEqual(self._outbox().pending(), [])

    def test_corrupt_line_is_skipped(self):
        outbox = self._outbox()
        outbox.add_event("e1", "r1", "Room 1", 0.5, "fall_1.mp4")
        with open(outbox.path, "a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        reloaded = self._outbox()
        self.assertEqual([i["eventId"] for i in reloaded.pending()], ["e1"])

    def test_attach_clip_by_filename(self):
        outbox = self._outbox()
        outbox.add_event("e1", "r1", "Room 1", 0.5, "fall_1.mp4")
        outbox.add_event("e2", "r1", "Room 1", 0.6, "fall_2.mp4")
        self.assertTrue(outbox.attach_clip_by_filename("fall_2.mp4", "/c/2.mp4"))
        pending = {i["eventId"]: i for i in outbox.pending()}
        self.assertEqual(pending["e2"]["clip"], "/c/2.mp4")
        self.assertIsNone(pending["e1"]["clip"])
        self.assertFalse(outbox.attach_clip_by_filename("fall_9.mp4", "/c/9.mp4"))

    def test_finish_clip_by_filename_closes_the_event(self):
        outbox = self._outbox()
        outbox.add_event("e1", "r1", "Room 1", 0.5, "fall_1.mp4")
        outbox.add_event("e2", "r1", "Room 1", 0.5, "fall_2.mp4")
        self.assertEqual(outbox.finish_clip_by_filename("fall_1.mp4"), "e1")
        self.assertEqual([i["eventId"] for i in outbox.pending()], ["e2"])
        self.assertEqual(self._outbox().pending()[0]["eventId"], "e2")
        # A clip that did attach is not closed by a late "lost" report.
        outbox.attach_clip("e2", "/c/2.mp4")
        self.assertIsNone(outbox.finish_clip_by_filename("fall_2.mp4"))

    # -- the keypoint file ---------------------------------------------------

    def test_keypoint_file_attaches_with_the_clip_and_survives_reload(self):
        outbox = self._outbox()
        outbox.add_event("e1", "r1", "Room 1", 0.5, "fall_1.mp4")
        self.assertIsNone(outbox.pending()[0]["keypoints"])
        self.assertTrue(outbox.attach_files_by_filename(
            "fall_1.mp4", "/c/fall_1.mp4", "/c/fall_1.npz"))

        for box in (outbox, self._outbox()):
            item = box.pending()[0]
            self.assertEqual(item["clip"], "/c/fall_1.mp4")
            self.assertEqual(item["keypoints"], "/c/fall_1.npz")
            self.assertFalse(item["clipSent"])

    def test_keypoints_are_journalled_before_the_clip(self):
        # The clip is what tells the worker an event is ready to finish; a
        # replay cut off between the two lines must not see it without the
        # keypoint file that goes with it.
        outbox = self._outbox()
        outbox.add_event("e1", "r1", "Room 1", 0.5, "fall_1.mp4")
        outbox.attach_files("e1", "/c/fall_1.mp4", "/c/fall_1.npz")
        with open(outbox.path, "r", encoding="utf-8") as fh:
            ops = [json.loads(line)["op"] for line in fh if line.strip()]
        self.assertEqual(ops, ["event", "keypoints", "clip"])

    def test_keypoints_alone_when_the_clip_was_lost(self):
        outbox = self._outbox()
        outbox.add_event("e1", "r1", "Room 1", 0.5, "fall_1.mp4")
        self.assertTrue(outbox.attach_files_by_filename(
            "fall_1.mp4", None, "/c/fall_1.npz"))
        item = self._outbox().pending()[0]
        self.assertIsNone(item["clip"])
        self.assertEqual(item["keypoints"], "/c/fall_1.npz")
        # Something is attached, so a late "lost" report does not close it.
        self.assertIsNone(outbox.finish_clip_by_filename("fall_1.mp4"))

    def test_attaching_nothing_is_an_error(self):
        outbox = self._outbox()
        outbox.add_event("e1", "r1", "Room 1", 0.5, "fall_1.mp4")
        with self.assertRaises(ValueError):
            outbox.attach_files("e1", None, None)
        self.assertFalse(outbox.attach_files("nope", "/c/x.mp4", None))

    def test_clip_sent_survives_reload(self):
        outbox = self._outbox()
        outbox.add_event("e1", "r1", "Room 1", 0.5, "fall_1.mp4")
        outbox.attach_files("e1", "/c/fall_1.mp4", "/c/fall_1.npz")
        outbox.mark_clip_sent("e1")
        outbox.mark_clip_sent("e1")  # a second mark journals nothing
        self.assertTrue(self._outbox().pending()[0]["clipSent"])
        with open(outbox.path, "r", encoding="utf-8") as fh:
            ops = [json.loads(line)["op"] for line in fh if line.strip()]
        self.assertEqual(ops.count("clip_sent"), 1)

    def test_compaction_keeps_the_keypoint_file_and_clip_sent(self):
        outbox = self._outbox()
        outbox.add_event("keep", "r1", "Room 1", 0.5, "fall_keep.mp4")
        outbox.attach_files("keep", "/c/fall_keep.mp4", "/c/fall_keep.npz")
        outbox.mark_clip_sent("keep")
        for i in range(200):
            outbox.add_event(f"e{i}", "r1", "Room 1", 0.5, f"fall_{i}.mp4")
            outbox.mark_done(f"e{i}")

        with open(outbox.path, "r", encoding="utf-8") as fh:
            ids = [json.loads(line)["eventId"] for line in fh if line.strip()]
        self.assertNotIn("e0", ids)  # compacted since `keep` was attached
        [item] = self._outbox().pending()
        self.assertEqual(item["eventId"], "keep")
        self.assertEqual(item["clip"], "/c/fall_keep.mp4")
        self.assertEqual(item["keypoints"], "/c/fall_keep.npz")
        self.assertTrue(item["clipSent"])

    def test_a_journal_from_before_keypoint_files_still_replays(self):
        # Lines written by the previous release: no keypoint facts at all.
        self.dir.mkdir(exist_ok=True)
        with open(self.dir / "outbox.jsonl", "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"op": "event", "eventId": "e1", "roomId": "r1",
                                 "roomName": "Room 1", "confidence": 0.5,
                                 "clipFilename": "fall_1.mp4",
                                 "createdAt": 1.0}) + "\n")
            fh.write(json.dumps({"op": "clip", "eventId": "e1",
                                 "path": "/c/fall_1.mp4"}) + "\n")
        item = self._outbox().pending()[0]
        self.assertEqual(item["clip"], "/c/fall_1.mp4")
        self.assertIsNone(item["keypoints"])
        self.assertFalse(item["clipSent"])

    def test_journal_is_compacted(self):
        outbox = self._outbox()
        for i in range(200):
            outbox.add_event(f"e{i}", "r1", "Room 1", 0.5, f"fall_{i}.mp4")
            outbox.mark_done(f"e{i}")
        outbox.add_event("keep", "r1", "Room 1", 0.5, "fall_keep.mp4")

        with open(outbox.path, "r", encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        ids = [f["eventId"] for f in lines]
        # Compaction rewrites the journal to the live items; the only trace of
        # the 200 finished events that may remain is the few facts written
        # since the last compaction.
        self.assertNotIn("e0", ids)
        self.assertLess(len(lines), 16)
        self.assertEqual([i["eventId"] for i in self._outbox().pending()], ["keep"])


if __name__ == "__main__":
    unittest.main()
