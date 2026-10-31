"""fallcore.refit: the weekly Refit, through its one entry point.

The seam is `refit.run(folder, decision, checkpoint, now)`. Every test lays
out a temporary folder the way the bucket is laid out -- Keypoint Files by
Outcome and room, the pinned Commissioning Recording and held-out session, and
their manifest -- deploys a Camera Head, and asserts only on the verdict and
its report. No network, no weights: nothing here builds the Fall Decision's
models.

The bodies are the per-subject stream tests' bodies -- upright, descending,
lying -- each lying down at a position across the room. A stub Checkpoint
stands in for the real one: a window's embedding says where the body in its
last row lies down, one column per position, and is all zeros while it is
upright. A Camera Head on it is a weight per position, so the head a Refit
fits calls a fall wherever its training windows lying there were mostly Fall,
and nowhere it has never seen anyone lie. The deployed head calls a fall
wherever anyone lies down, as it did on every Alert it raised. The real
fitting, rolling buffer, latch and descent gate run underneath.
"""
from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fallcore import config as cfg
from fallcore import keypoint_file, refit
from fallcore.calibrate import ProbeHead, validate
from fallcore.fall_decision import FallDecision
from fallcore.keypoint_file import Frame, Person

#: Whichever side of the split these land on is fixed by SHA-256 of the name:
#: these are 0 mod 4, Eval Alerts...
EVAL_IDS = ("event-02", "event-04", "event-05", "event-10", "event-12")
#: ...and these are not, so they are training Alerts.
TRAINING_IDS = ("event-01", "event-03", "event-06", "event-07", "event-08",
                "event-09", "event-11", "event-13", "event-14", "event-15")

FITTED = datetime(2026, 10, 20, tzinfo=timezone.utc)
NOW = datetime(2026, 10, 27, tzinfo=timezone.utc)
BEFORE = FITTED.timestamp() - 86400.0      # an Alert raised the day before
AFTER = FITTED.timestamp() + 86400.0       # and one the day after

GENUINE, FALSE_ALARM = refit.Outcome.GENUINE, refit.Outcome.FALSE_ALARM

HZ = 15          # a Keypoint File's frames per second, the live rate
WINDOW = 60      # rows per window
FEATURES = "coords"
#: Where a body can lie down, one embedding column each.
POSITIONS = tuple(round(0.30 + 0.05 * i, 2) for i in range(9))
#: An Alert's frames: standing, half a second going down, lying. The alarm
#: latches 37 frames after the body is down, so the windows, which end every
#: second from the first full one, end on whole seconds from the trigger.
STANDING, DOWN, LYING = 150, HZ // 2, 120
TRIGGER = STANDING + DOWN + 37


class LyingAt:
    """Stub Checkpoint: a window's embedding is one-hot on the position the
    body in its last row lies down at, all zeros while upright or unseen."""

    features = FEATURES
    seq_len = WINDOW

    def embed(self, windows) -> np.ndarray:
        out = np.zeros((len(windows), len(POSITIONS)))
        for i, window in enumerate(windows):
            last = window[-1]
            seen = last[:, 2] > 0.3
            if seen.any() and np.ptp(last[seen, 0]) < 0.2:
                x = round(float(last[seen, 1].mean()), 2)
                if x in POSITIONS:
                    out[i, POSITIONS.index(x)] = 1.0
        return out

    def classifier(self, head: ProbeHead):
        return OnHead(self, head)


class OnHead:
    """The stub Checkpoint with a Camera Head on it, as the camera runs it."""

    names = ["stub"]

    def __init__(self, checkpoint: LyingAt, head: ProbeHead):
        self.checkpoint, self.heads = checkpoint, [head]

    def proba(self, window, aspect=1.0):
        p = float(self.heads[0].proba(self.checkpoint.embed([window]))[0])
        return p, [p]


def body(hip_y: float, x: float, lying: float = 0.0) -> np.ndarray:
    """A `(17, 3)` cache row: a body with its hips at `hip_y`, centred on `x`.

    `lying` goes from 0 (upright: head 0.3 above the hips, feet 0.3 below) to
    1 (flat: every keypoint at hip height, spread 0.4 across).
    """
    row = np.zeros((cfg.N_KEYPOINTS, 3), dtype=np.float32)
    span = np.linspace(-0.3, 0.3, cfg.N_KEYPOINTS)
    row[:, 0] = hip_y + (1 - lying) * span
    row[:, 1] = x + lying * span * (0.2 / 0.3)
    row[cfg.LEFT_HIP, 0] = row[cfg.RIGHT_HIP, 0] = hip_y
    row[:, 2] = 0.9
    return row


def descending(x: float, hz: int = HZ, standing: int = STANDING,
               lying: int = LYING) -> list[np.ndarray]:
    """Rows of someone standing at `x`, going down over half a second, then
    lying: a fall or a Controlled Descent, as its label says."""
    rows = [body(0.5, x) for _ in range(standing)]
    steps = hz // 2
    rows += [body(0.5 + 0.35 * (i + 1) / steps, x, lying=(i + 1) / steps)
             for i in range(steps)]
    rows += [body(0.85, x, lying=1.0) for _ in range(lying)]
    return rows


def upright(x: float, n: int) -> list[np.ndarray]:
    return [body(0.5, x) for _ in range(n)]


def frames(t0: float, *people: list[np.ndarray]) -> list[Frame]:
    """Keypoint File frames of each person's rows from `t0`, subjects and
    track ids 1, 2, ...; the first person's box the largest and the lead."""
    out = []
    for f in range(max(len(rows) for rows in people)):
        here = []
        for i, rows in enumerate(people):
            if f >= len(rows):
                continue
            r = rows[f]
            here.append(Person(track_id=i + 1, keypoints=r, fallen=False,
                               centre=(float(r[:, 1].mean()),
                                       float(r[:, 0].mean())),
                               area=0.1 / (i + 1), subject=i + 1))
        out.append(Frame(t=t0 + f / HZ, aspect=1.0, alarm=False, score=0.0,
                         people=tuple(here), lead=1))
    return out


def npy(rows: list[np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.stack(rows))
    return buf.getvalue()


#: The Commissioning Recording: falls at these positions, and two clips of
#: someone walking about. Recorded at 30 fps, as picam was.
COMMISSIONING_FALLS = (0.30, 0.35)
#: The held-out session: falls where the Commissioning Recording has them,
#: and one Controlled Descent.
HELDOUT_FALLS, HELDOUT_DESCENTS = (0.30, 0.35), (0.40,)


class RefitTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        # The weights fetcher's tree, outside the mirrored bucket.
        self._weights = tempfile.TemporaryDirectory()
        self.weights = Path(self._weights.name)
        self.checkpoint = LyingAt()
        self.deploy(FITTED)
        self.commission()

    def tearDown(self):
        self._tmp.cleanup()
        self._weights.cleanup()

    # -- the deployed Camera Head ---------------------------------------------

    def deploy(self, fitted_at: datetime | None, raw: str | None = None):
        """Deploy a Camera Head fitted at `fitted_at` (None: one that predates
        recording its fit date, as the shipped head does). It calls a fall
        wherever anyone lies down."""
        head = self.weights / "head.npz"
        dim = len(POSITIONS)
        ProbeHead(mean=np.zeros(dim), scale=np.ones(dim),
                  coef=np.full(dim, 10.0), intercept=np.array([-5.0]),
                  backbone="stub.pt", features=FEATURES, seq_len=WINDOW,
                  pose=FallDecision.pose,
                  fitted_at=(raw if raw is not None else
                             "" if fitted_at is None else
                             fitted_at.isoformat())).save(head)
        self.decision = FallDecision(checkpoints=(self.weights / "stub.pt",),
                                     heads=(head,), frame_stride=1)

    # -- the bucket ---------------------------------------------------------

    def pin(self, name: str, data: bytes) -> Path:
        """Put a pinned object in the folder and list it in the manifest."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        with open(self.root / refit.MANIFEST, "a", encoding="utf-8") as fh:
            fh.write(f"{hashlib.sha256(data).hexdigest()}  {name}\n")
        return path

    def pin_clip(self, name: str, frames_: list[Frame]) -> Path:
        """Pin a held-out clip, a Keypoint File."""
        scratch = self.weights / "clip.npz"
        keypoint_file.save(scratch, frames_, TRIGGER, pose=self.decision.pose)
        path = self.pin(name, scratch.read_bytes())
        scratch.unlink()
        return path

    def commission(self):
        """The pinned base data: the Commissioning Recording, largest-box
        keypoint caches by label, and the held-out session, Keypoint Files
        by label."""
        for i, x in enumerate(COMMISSIONING_FALLS):
            self.pin(f"{refit.COMMISSIONING}/fall/picam_fall_{i}.npy",
                     npy(descending(x, hz=30, standing=120, lying=240)))
        for i in range(2):
            self.pin(f"{refit.COMMISSIONING}/adl/picam_walk_{i}.npy",
                     npy(upright(0.5, 300)))
        t0 = BEFORE - 86400.0
        for i, x in enumerate(HELDOUT_FALLS):
            self.pin_clip(f"{refit.HELDOUT}/fall/p3_fall_{i}.npz",
                          frames(t0, descending(x)))
        for i, x in enumerate(HELDOUT_DESCENTS):
            self.pin_clip(f"{refit.HELDOUT}/adl/p3_lie_down_{i}.npz",
                          frames(t0, descending(x)))

    def filed(self, name: str, outcome: refit.Outcome | None,
              room: str = "room-1") -> Path:
        """Where the notification service files an Alert's upload: under its
        Outcome once triaged, under `incoming/` before."""
        folder = (self.root / refit.LABELLED / outcome.value if outcome
                  else self.root / refit.INCOMING)
        path = folder / room / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def alert(self, event_id: str, outcome: refit.Outcome | None,
              t: float = AFTER, pose: str | None = None, x: float = 0.70,
              others: tuple[list[np.ndarray], ...] = ()) -> Path:
        """A Keypoint File for one Alert raised at `t` by someone going down
        at `x`, with `others` in view, filed as the notification service
        files it."""
        path = self.filed(f"{event_id}.npz", outcome)
        keypoint_file.save(path, frames(t - TRIGGER / HZ, descending(x),
                                        *others),
                           trigger=TRIGGER, pose=pose or self.decision.pose)
        return path

    def week(self, genuine=(0.50,), false_alarms=(0.55, 0.60, 0.65, 0.70),
             eval_genuine=(0.50,), eval_false_alarms=(0.60,)):
        """A week of triaged Alerts, by where each one's subject went down."""
        training, evals = iter(TRAINING_IDS), iter(EVAL_IDS)
        for outcome, xs, ids in ((GENUINE, genuine, training),
                                 (FALSE_ALARM, false_alarms, training),
                                 (GENUINE, eval_genuine, evals),
                                 (FALSE_ALARM, eval_false_alarms, evals)):
            for x in xs:
                self.alert(next(ids), outcome, x=x)

    def run_refit(self, now=NOW):
        return refit.run(self.root, self.decision, checkpoint=self.checkpoint,
                         now=now)

    # -- the verdict ----------------------------------------------------------

    def test_no_fall_lost_and_two_fewer_false_alarms_is_promoted(self):
        # Quiet on the held-out Controlled Descent, where it never saw anyone
        # lie, and on the False Alarm Eval Alert at 0.60, where it was taught
        # one; it still catches the Genuine one at 0.50.
        self.week()

        verdict = self.run_refit()

        self.assertEqual(verdict.kind, refit.PROMOTED, str(verdict))
        self.assertIn("Promotion Gate passed", str(verdict))

    def test_the_candidate_head_records_what_it_was_fitted_on(self):
        self.week()

        verdict = self.run_refit()
        # As the detector reads it: from its file, then the existing check.
        path = verdict.head.save(self.weights / "candidate.npz")
        head = ProbeHead.load(path)
        validate(head, self.decision.checkpoints[0], FEATURES, WINDOW,
                 pose=self.decision.pose)

        self.assertEqual(head.pose, self.decision.pose)
        self.assertEqual(datetime.fromisoformat(head.fitted_at), NOW)
        training = sorted(TRAINING_IDS[:5])
        self.assertEqual(head.training_hash, hashlib.sha256(
            "\n".join(training).encode()).hexdigest())
        self.assertEqual(head.training_hash, verdict.report.training_hash)
        self.assertIn("Promotion Gate passed", head.gate)
        self.assertIn("held-out session (3 clips, 2 falls): falls 2 -> 2, "
                      "False Alarms 1 -> 0", head.gate)

    def test_a_held_out_fall_lost_is_gate_failed(self):
        # False Alarms lying where a held-out fall lies teach the candidate
        # to miss it, however many False Alarms it removes.
        self.week(false_alarms=(0.35, 0.35, 0.35, 0.35))

        verdict = self.run_refit()

        self.assertEqual(verdict.kind, refit.GATE_FAILED, str(verdict))
        self.assertEqual((verdict.gate.heldout.lost, verdict.gate.evals.lost),
                         (1, 0))
        self.assertIn("1 fall lost", verdict.reason)
        self.assertIsNone(verdict.head)

    def test_a_genuine_eval_alert_lost_is_gate_failed(self):
        # A fall where the candidate was taught a False Alarm, with three
        # False Alarms removed.
        self.week(eval_genuine=(0.65,), eval_false_alarms=(0.55, 0.60))

        verdict = self.run_refit()

        self.assertEqual(verdict.kind, refit.GATE_FAILED, str(verdict))
        self.assertEqual((verdict.gate.heldout.lost, verdict.gate.evals.lost),
                         (0, 1))
        self.assertIsNone(verdict.head)

    def test_only_one_fewer_false_alarm_is_gate_failed(self):
        # The False Alarm Eval Alert lies where a Genuine fall was taught, so
        # only the held-out Controlled Descent goes quiet.
        self.week(eval_false_alarms=(0.50,))

        verdict = self.run_refit()

        self.assertEqual(verdict.kind, refit.GATE_FAILED, str(verdict))
        self.assertIn("1 fewer False Alarm, fewer than 2", verdict.reason)
        self.assertIn("Promotion Gate failed", str(verdict))
        self.assertIsNone(verdict.head)

    # -- the labels (ADR 0002) ------------------------------------------------
    #
    # A subject in view from an Alert's first frame has a window ending every
    # second from 9 s before the trigger frame to 5 s after it: 15 windows.

    def false_alarms(self, n: int = 4):
        for event_id in TRAINING_IDS[1:1 + n]:
            self.alert(event_id, FALSE_ALARM)

    def test_a_genuine_alert_is_a_fall_around_the_trigger_and_adl_well_before(self):
        self.alert(TRAINING_IDS[0], GENUINE, x=0.50)
        self.false_alarms()

        fit = self.run_refit().fit

        # Fall: the windows ending 2 s before to 3 s after, both included.
        self.assertEqual(fit.fall, 6)
        # ADL: ending more than 6 s before, so 9, 8 and 7 s; then every
        # window of the four single-subject False Alarms. From 6 s to 3 s
        # before, and the 4 and 5 s after, lying on the floor, are neither.
        self.assertEqual(fit.adl, 3 + 4 * 15)

    def test_only_the_subject_that_raised_a_genuine_alert_is_labelled(self):
        # A carer kneeling beside the resident who fell.
        self.alert(TRAINING_IDS[0], GENUINE, x=0.50,
                   others=(descending(0.55),))
        self.false_alarms()

        fit = self.run_refit().fit

        self.assertEqual((fit.fall, fit.adl), (6, 3 + 4 * 15))

    def test_everyone_in_a_false_alarm_is_adl(self):
        self.alert(TRAINING_IDS[0], GENUINE, x=0.50)
        self.false_alarms(3)
        self.alert(TRAINING_IDS[4], FALSE_ALARM, x=0.60,
                   others=(upright(0.40, STANDING + DOWN + LYING),))

        fit = self.run_refit().fit

        self.assertEqual((fit.fall, fit.adl), (6, 3 + 3 * 15 + 2 * 15))

    def test_no_eval_alerts_windows_reach_the_fit(self):
        self.week()
        before = self.run_refit()
        self.alert(EVAL_IDS[2], GENUINE, x=0.40)
        self.alert(EVAL_IDS[3], FALSE_ALARM, x=0.30,
                   others=(upright(0.60, STANDING),))

        after = self.run_refit()

        self.assertEqual(after.report.eval, before.report.eval + 2)
        self.assertEqual(after.fit, before.fit)

    # -- the pinned base data -----------------------------------------------

    def test_a_pinned_object_that_changed_is_an_error_not_a_verdict(self):
        (self.root / refit.HELDOUT / "fall/p3_fall_0.npz").write_bytes(b"edited")
        with self.assertRaises(refit.PinnedDataError):
            self.run_refit()

    def test_an_object_the_manifest_does_not_pin_is_an_error(self):
        (self.root / refit.HELDOUT / "fall/p3_extra.npz").write_bytes(b"new clip")
        with self.assertRaises(refit.PinnedDataError):
            self.run_refit()

    def test_a_pinned_object_that_is_missing_is_an_error(self):
        (self.root / refit.COMMISSIONING / "fall/picam_fall_0.npy").unlink()
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

        self.assertIn(verdict.kind, (refit.PROMOTED, refit.GATE_FAILED))

    def test_alerts_raised_before_the_deployed_camera_head_was_fitted_are_not_new(self):
        for event_id in TRAINING_IDS[:5]:
            self.alert(event_id, GENUINE, t=BEFORE)
        for event_id in TRAINING_IDS[5:9]:
            self.alert(event_id, GENUINE, t=AFTER)

        verdict = self.run_refit()

        self.assertEqual(verdict.kind, refit.SKIPPED)
        self.assertIn("4 new training alerts", verdict.reason.lower())
        # Not new, but still training data the Candidate Head is fitted on.
        self.assertEqual(verdict.report.training, 9)
        self.assertEqual(verdict.report.new_training, 4)

    def test_a_deployed_camera_head_with_no_fit_date_has_seen_no_alert(self):
        # The shipped Camera Head predates the fit date: it was fitted on the
        # Commissioning Recording alone, so every triaged Alert is new to it.
        self.alert(TRAINING_IDS[0], GENUINE, t=BEFORE)
        for event_id in TRAINING_IDS[1:5]:
            self.alert(event_id, FALSE_ALARM, t=BEFORE)

        self.deploy(fitted_at=None)

        verdict = self.run_refit()

        self.assertIn(verdict.kind, (refit.PROMOTED, refit.GATE_FAILED))
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
        self.week()
        promoted = self.run_refit()
        # A Genuine Eval Alert where the candidate was taught a False Alarm.
        self.alert(EVAL_IDS[2], GENUINE, x=0.65)
        failed = self.run_refit()

        for verdict in (skipped, promoted, failed):
            text = str(verdict)
            self.assertIn(verdict.kind, text)
            self.assertIn(verdict.report.training_hash, text)
            # So is the gate report a promoted head carries.
            if verdict.head is not None:
                text += verdict.head.gate
            self.assertNotIn("event-", text)
            self.assertNotIn("2026", text)
            self.assertNotIn(str(int(AFTER))[:6], text)
        self.assertEqual((skipped.kind, promoted.kind, failed.kind),
                         (refit.SKIPPED, refit.PROMOTED, refit.GATE_FAILED))

    def test_keypoint_files_it_cannot_read_are_left_out_and_counted(self):
        for event_id in TRAINING_IDS[:4]:
            self.alert(event_id, GENUINE)
        # A format version from a newer detector...
        newer = self.alert(TRAINING_IDS[4], GENUINE)
        with np.load(newer) as f:
            arrays = dict(f)
        arrays["version"] = np.int32(keypoint_file.FORMAT_VERSION + 1)
        with open(newer, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        # ...and an upload the notification service filed under another
        # extension, triaged and not.
        self.filed(f"{TRAINING_IDS[5]}.parquet", FALSE_ALARM).write_bytes(b"PAR1")
        self.filed(f"{TRAINING_IDS[6]}.parquet", None).write_bytes(b"PAR1")

        verdict = self.run_refit()

        self.assertEqual(verdict.kind, refit.SKIPPED)
        self.assertIn("4 new training alerts", verdict.reason.lower())
        self.assertEqual(verdict.report.unreadable, 2)
        self.assertEqual(verdict.report.untriaged, 1)
        self.assertEqual(verdict.report.training, 4)
        self.assertNotIn("event-", str(verdict))

    def test_the_fit_date_must_say_its_time_zone(self):
        # A naive time would be read as the runner's local time.
        with self.assertRaises(ValueError):
            self.run_refit(now=datetime(2026, 10, 27))
        self.deploy(None, raw="2026-10-20T00:00:00")
        with self.assertRaises(ValueError):
            self.run_refit()


if __name__ == "__main__":
    unittest.main()
