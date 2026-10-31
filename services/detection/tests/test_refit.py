"""fallcore.refit: the weekly Refit, through its one entry point.

The seam is `refit.run(folder, decision, fitted_at)`. Every test lays out a
temporary folder the way the bucket is laid out -- Keypoint Files by Outcome
and room, the pinned Commissioning Recording and held-out session, and their
manifest -- and asserts only on the verdict and its report. No network, no
weights: nothing here builds the Fall Decision's models.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fallcore import config as cfg
from fallcore import keypoint_file, refit
from fallcore.fall_decision import FallDecision
from fallcore.keypoint_file import Frame, Person

#: Whichever side of the split these land on is fixed by SHA-256 of the name:
#: these are 0 mod 4, Eval Alerts...
EVAL_IDS = ("event-02", "event-04", "event-05", "event-10", "event-12")
#: ...and these are not, so they are training Alerts.
TRAINING_IDS = ("event-01", "event-03", "event-06", "event-07", "event-08",
                "event-09", "event-11", "event-13", "event-14", "event-15")

FITTED = datetime(2026, 10, 20, tzinfo=timezone.utc)
BEFORE = FITTED.timestamp() - 86400.0      # an Alert raised the day before
AFTER = FITTED.timestamp() + 86400.0       # and one the day after

GENUINE, FALSE_ALARM = "genuine", "false_alarm"


def body(fill: float) -> Person:
    return Person(track_id=1, subject=1, fallen=False, area=0.1,
                  centre=(0.5, 0.5),
                  keypoints=np.full((cfg.N_KEYPOINTS, 3), fill, np.float32))


class RefitTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.decision = FallDecision()
        self.pin("keypoints/commissioning/picam2_fall_01.npy", b"commissioning")
        self.pin("keypoints/heldout/p3_fall_01.npy", b"held-out session")

    def tearDown(self):
        self._tmp.cleanup()

    # -- the bucket ---------------------------------------------------------

    def pin(self, name: str, data: bytes) -> Path:
        """Put a pinned object in the folder and list it in the manifest."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        with open(self.root / refit.MANIFEST, "a", encoding="utf-8") as fh:
            fh.write(f"{hashlib.sha256(data).hexdigest()}  {name}\n")
        return path

    def alert(self, event_id: str, outcome: str | None, t: float = AFTER,
              pose: str | None = None, room: str = "room-1") -> None:
        """A Keypoint File for one Alert, filed as the notification service
        files it: under its Outcome once triaged, under `incoming/` before."""
        folder = (f"keypoints/labelled/{outcome}/{room}" if outcome
                  else f"keypoints/incoming/{room}")
        path = self.root / folder / f"{event_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = [Frame(t=t - 1.0, aspect=16 / 9, alarm=False, score=0.1,
                        people=(body(0.2),), lead=1),
                  Frame(t=t, aspect=16 / 9, alarm=True, score=0.9,
                        people=(body(0.8),), lead=1)]
        keypoint_file.save(path, frames, trigger=1,
                           pose=pose or self.decision.pose)

    def run_refit(self, fitted_at=FITTED):
        return refit.run(self.root, self.decision, fitted_at)

    # -- the pinned base data -----------------------------------------------

    def test_a_pinned_object_that_changed_is_an_error_not_a_verdict(self):
        (self.root / "keypoints/heldout/p3_fall_01.npy").write_bytes(b"edited")
        with self.assertRaises(refit.PinnedDataError):
            self.run_refit()

    def test_an_object_the_manifest_does_not_pin_is_an_error(self):
        (self.root / "keypoints/heldout/p3_extra.npy").write_bytes(b"new clip")
        with self.assertRaises(refit.PinnedDataError):
            self.run_refit()

    def test_a_pinned_object_that_is_missing_is_an_error(self):
        (self.root / "keypoints/commissioning/picam2_fall_01.npy").unlink()
        with self.assertRaises(refit.PinnedDataError):
            self.run_refit()

    # -- the skip rule ------------------------------------------------------

    def test_fewer_than_five_new_training_alerts_is_skipped(self):
        for event_id in TRAINING_IDS[:2]:
            self.alert(event_id, GENUINE)
        for event_id in TRAINING_IDS[2:4]:
            self.alert(event_id, FALSE_ALARM)

        verdict = self.run_refit()

        self.assertEqual(verdict.kind, refit.SKIPPED)
        self.assertIn("4 new training alerts", verdict.reason.lower())
        self.assertIn("fewer than 5", verdict.reason.lower())

    def test_no_new_genuine_training_alert_is_skipped(self):
        for event_id in TRAINING_IDS[:6]:
            self.alert(event_id, FALSE_ALARM)

        verdict = self.run_refit()

        self.assertEqual(verdict.kind, refit.SKIPPED)
        self.assertIn("no new genuine", verdict.reason.lower())

    def test_a_week_with_enough_new_data_is_not_skipped(self):
        self.alert(TRAINING_IDS[0], GENUINE)
        for event_id in TRAINING_IDS[1:5]:
            self.alert(event_id, FALSE_ALARM)

        verdict = self.run_refit()

        # Fitting and the Promotion Gate are later tickets.
        self.assertEqual(verdict.kind, refit.NOT_YET_IMPLEMENTED)

    def test_alerts_from_before_the_deployed_head_was_fitted_are_not_new(self):
        for event_id in TRAINING_IDS[:5]:
            self.alert(event_id, GENUINE, t=BEFORE)
        for event_id in TRAINING_IDS[5:9]:
            self.alert(event_id, GENUINE, t=AFTER)

        verdict = self.run_refit()

        self.assertEqual(verdict.kind, refit.SKIPPED)
        self.assertIn("4 new training alerts", verdict.reason.lower())
        # Not new, but still training data the candidate is fitted on.
        self.assertEqual(verdict.report.training, 9)
        self.assertEqual(verdict.report.new_training, 4)

    def test_a_head_with_no_fit_date_has_seen_no_alert(self):
        # The shipped head predates the fit date: it was fitted on the
        # Commissioning Recording alone, so every triaged Alert is new to it.
        self.alert(TRAINING_IDS[0], GENUINE, t=BEFORE)
        for event_id in TRAINING_IDS[1:5]:
            self.alert(event_id, FALSE_ALARM, t=BEFORE)

        verdict = self.run_refit(fitted_at=None)

        self.assertEqual(verdict.kind, refit.NOT_YET_IMPLEMENTED)
        self.assertEqual(verdict.report.new_training, 5)

    # -- the eval split -----------------------------------------------------

    def test_eval_alerts_are_never_training_alerts(self):
        for event_id in EVAL_IDS:
            self.alert(event_id, GENUINE)
        for event_id in TRAINING_IDS[:4]:
            self.alert(event_id, FALSE_ALARM)

        verdict = self.run_refit()

        # Five new Genuine Alerts arrived, but all are Eval Alerts.
        self.assertEqual(verdict.kind, refit.SKIPPED)
        self.assertIn("4 new training alerts", verdict.reason.lower())
        self.assertEqual(verdict.report.eval, 5)
        self.assertEqual(verdict.report.eval_genuine, 5)
        self.assertEqual(verdict.report.training, 4)
        # The training set, by its hash: SHA-256 of the sorted Event IDs.
        expected = "\n".join(sorted(TRAINING_IDS[:4])).encode()
        self.assertEqual(verdict.report.training_hash,
                         hashlib.sha256(expected).hexdigest())

    def test_the_split_is_the_same_on_every_run(self):
        for event_id in EVAL_IDS[:3] + TRAINING_IDS[:6]:
            self.alert(event_id, GENUINE)
        first = self.run_refit()
        # A new Alert next week moves nobody already filed.
        self.alert(EVAL_IDS[3], FALSE_ALARM)
        self.alert(TRAINING_IDS[6], FALSE_ALARM)
        second = self.run_refit()

        self.assertEqual((first.report.eval, first.report.training), (3, 6))
        self.assertEqual((second.report.eval, second.report.training), (4, 7))
        self.assertEqual(self.run_refit().report, second.report)

    # -- what is left out ---------------------------------------------------

    def test_untriaged_and_other_backbone_alerts_are_left_out_and_counted(self):
        for event_id in TRAINING_IDS[:4]:
            self.alert(event_id, GENUINE)
        for event_id in TRAINING_IDS[4:7]:
            self.alert(event_id, None)                  # never triaged
        self.alert(TRAINING_IDS[7], GENUINE, pose="yolo26n_r640")
        self.alert(EVAL_IDS[0], FALSE_ALARM, pose="yolo26n_r640")

        verdict = self.run_refit()

        self.assertEqual(verdict.kind, refit.SKIPPED)
        self.assertIn("4 new training alerts", verdict.reason.lower())
        self.assertEqual(verdict.report.untriaged, 3)
        self.assertEqual(verdict.report.other_pose, 2)
        self.assertEqual((verdict.report.training, verdict.report.eval), (4, 0))

    # -- what is published ---------------------------------------------------

    def test_the_report_names_no_alert_and_no_time(self):
        # The run summary is public: counts and the training hash only.
        self.alert(TRAINING_IDS[0], GENUINE)
        self.alert(EVAL_IDS[0], GENUINE)
        self.alert(TRAINING_IDS[1], None)
        skipped = self.run_refit()
        for event_id in TRAINING_IDS[2:6]:
            self.alert(event_id, FALSE_ALARM)
        carried_on = self.run_refit()

        for verdict in (skipped, carried_on):
            text = str(verdict)
            self.assertIn(verdict.kind, text)
            self.assertIn(verdict.report.training_hash, text)
            self.assertNotIn("event-", text)
            self.assertNotIn("2026", text)
            self.assertNotIn(str(int(AFTER))[:6], text)
        self.assertEqual((skipped.kind, carried_on.kind),
                         (refit.SKIPPED, refit.NOT_YET_IMPLEMENTED))

    def test_the_fit_date_must_say_its_time_zone(self):
        with self.assertRaises(ValueError):
            self.run_refit(fitted_at=datetime(2026, 10, 20))


if __name__ == "__main__":
    unittest.main()
