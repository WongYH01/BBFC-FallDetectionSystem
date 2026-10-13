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
