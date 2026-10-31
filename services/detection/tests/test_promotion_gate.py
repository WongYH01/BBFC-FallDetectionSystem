"""fallcore.promotion_gate: a Candidate Head judged against the deployed one.

The seam is `promotion_gate.judge(decision, deployed, candidate, heldout,
evals, seq_len)`. Each test builds clips of synthetic bodies -- upright,
descending, lying -- as Keypoint File frames, and hands the gate a stub
classifier per head in place of the checkpoint and its Camera Head. A stub
calls a window a fall when the body in its last row is lying down at one of
the positions it was given, so each case chooses which clips each head fires
on, and the real rolling buffer, latch, descent gate and per-subject streams
decide underneath. Assertions are on the verdict and the report only.
"""
from __future__ import annotations

import unittest

import numpy as np

from fallcore import config as cfg
from fallcore import promotion_gate
from fallcore.fall_decision import FallDecision
from fallcore.keypoint_file import Frame, Person
from fallcore.promotion_gate import Clip

HZ = 15          # rows per second, the live rate at frame_stride 1
WINDOW = 60      # rows per window
T0 = 1_792_000_000.0                 # capture time of every clip's first frame

#: The deployed rule at the live rate: buffer 4, threshold 0.5, latch, and
#: the descent gate, one stream per tracked subject.
DECISION = FallDecision(frame_stride=1)


class Calls:
    """Stub classifier: P(fall) `calls[x]` when the window's last row is a
    body lying down at `x`, 0.01 otherwise."""

    names = ["stub"]

    def __init__(self, calls: dict[float, float]):
        self.calls = calls

    @classmethod
    def at(cls, *xs: float) -> Calls:
        """A head that calls a fall on whoever lies down at `xs`."""
        return cls({x: 0.99 for x in xs})

    def proba(self, window, aspect=1.0):
        last = window[-1]
        seen = last[:, 2] > 0.3
        p = 0.01
        if seen.any() and np.ptp(last[seen, 0]) < 0.2:
            x = round(float(last[seen, 1].mean()), 2)
            p = self.calls.get(x, 0.01)
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


def descending(x: float) -> list[np.ndarray]:
    """Rows of someone standing at `x`, going down over half a second, then
    lying: a fall or a Controlled Descent, as the clip's label says."""
    rows = [body(0.5, x) for _ in range(WINDOW)]
    steps = HZ // 2
    rows += [body(0.5 + 0.35 * (i + 1) / steps, x, lying=(i + 1) / steps)
             for i in range(steps)]
    rows += [body(0.85, x, lying=1.0) for _ in range(2 * WINDOW)]
    return rows


def frames(*people: list[np.ndarray]) -> list[Frame]:
    """Keypoint File frames of each person's rows, track ids 1, 2, ...; the
    first person's box the largest."""
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
        out.append(Frame(t=T0 + f / HZ, aspect=1.0, alarm=False, score=0.0,
                         people=tuple(here), lead=1))
    return out


def fall(x: float) -> Clip:
    return Clip(frames(descending(x)), fall=True)


def descent(x: float) -> Clip:
    """A Controlled Descent: the same movement as a fall, not one."""
    return Clip(frames(descending(x)), fall=False)


def judge(deployed, candidate, heldout, evals):
    return promotion_gate.judge(DECISION, deployed, candidate, heldout, evals,
                                seq_len=WINDOW)


#: Where each clip's subject lies down. The held-out session: two falls and
#: three Controlled Descents...
HELDOUT_FALLS, HELDOUT_DESCENTS = (0.30, 0.35), (0.40, 0.45, 0.48)
#: ...and the Eval Alerts: two Genuine, two False Alarms.
EVAL_GENUINE, EVAL_FALSE_ALARMS = (0.50, 0.55), (0.60, 0.65)
#: The deployed head fires on every clip, as it fired on every Eval Alert.
DEPLOYED = Calls.at(*HELDOUT_FALLS, *HELDOUT_DESCENTS, *EVAL_GENUINE,
                    *EVAL_FALSE_ALARMS)


def sets():
    heldout = ([fall(x) for x in HELDOUT_FALLS]
               + [descent(x) for x in HELDOUT_DESCENTS])
    evals = ([fall(x) for x in EVAL_GENUINE]
             + [descent(x) for x in EVAL_FALSE_ALARMS])
    return heldout, evals


class PromotionGateTest(unittest.TestCase):
    def test_no_fall_lost_and_two_fewer_false_alarms_passes(self):
        # Quiet on one held-out descent and one False Alarm Eval Alert.
        candidate = Calls.at(*HELDOUT_FALLS, 0.40, 0.45, *EVAL_GENUINE, 0.60)

        report = judge(DEPLOYED, candidate, *sets())

        self.assertTrue(report.passed)

    def test_one_held_out_fall_lost_fails(self):
        # However many False Alarms it removes.
        candidate = Calls.at(0.30, *EVAL_GENUINE)

        report = judge(DEPLOYED, candidate, *sets())

        self.assertFalse(report.passed)
        self.assertEqual(report.heldout.lost, 1)

    def test_one_genuine_eval_alert_lost_fails(self):
        candidate = Calls.at(*HELDOUT_FALLS, 0.50)

        report = judge(DEPLOYED, candidate, *sets())

        self.assertFalse(report.passed)
        self.assertEqual((report.heldout.lost, report.evals.lost), (0, 1))

    def test_only_one_fewer_false_alarm_fails(self):
        # Inside the Seed Floor, so not measured.
        candidate = Calls.at(*HELDOUT_FALLS, 0.40, 0.45, *EVAL_GENUINE,
                             *EVAL_FALSE_ALARMS)

        report = judge(DEPLOYED, candidate, *sets())

        self.assertFalse(report.passed)
        self.assertEqual(report.heldout.lost + report.evals.lost, 0)

    def test_a_fall_caught_elsewhere_does_not_pay_for_one_lost(self):
        # The deployed head misses one held-out fall; the candidate catches
        # it and loses the other. Falls caught are level, but one was lost.
        deployed = Calls.at(0.30, *HELDOUT_DESCENTS, *EVAL_GENUINE,
                            *EVAL_FALSE_ALARMS)
        candidate = Calls.at(0.35, *EVAL_GENUINE)

        report = judge(deployed, candidate, *sets())

        self.assertEqual(report.heldout.deployed.falls,
                         report.heldout.candidate.falls)
        self.assertFalse(report.passed)

    def test_window_auc_is_reported_but_never_decides(self):
        # Quiet on every descent, so it ranks the windows better, but it
        # scores one fall's lying windows under the threshold and loses it.
        candidate = Calls({0.30: 0.49, 0.35: 0.99, 0.50: 0.99, 0.55: 0.99})

        report = judge(DEPLOYED, candidate, *sets())

        self.assertGreater(report.heldout.candidate.auc,
                           report.heldout.deployed.auc)
        self.assertGreater(report.evals.candidate.auc,
                           report.evals.deployed.auc)
        self.assertEqual(report.heldout.lost, 1)
        self.assertFalse(report.passed)

    def test_the_report_gives_falls_and_false_alarms_before_and_after_per_set(self):
        candidate = Calls.at(*HELDOUT_FALLS, 0.40, 0.45, *EVAL_GENUINE, 0.60)

        report = judge(DEPLOYED, candidate, *sets())
        text = str(report)

        self.assertIn("Promotion Gate passed", text)
        self.assertIn("held-out session (5 clips, 2 falls): falls 2 -> 2, "
                      "False Alarms 3 -> 2", text)
        self.assertIn("Eval Alerts (4, 2 Genuine): falls 2 -> 2, "
                      "False Alarms 2 -> 1", text)
        self.assertIn(f"window AUC {report.heldout.deployed.auc:.3f} -> "
                      f"{report.heldout.candidate.auc:.3f}", text)
        self.assertEqual(text.count("window AUC"), 2)
        # Published in the run summary and the pull request: counts only,
        # no capture time.
        self.assertNotIn(str(int(T0))[:6], text)

    def test_a_failed_report_says_why(self):
        lost = str(judge(DEPLOYED, Calls.at(0.30, *EVAL_GENUINE), *sets()))
        one = str(judge(DEPLOYED,
                        Calls.at(*HELDOUT_FALLS, 0.40, 0.45, *EVAL_GENUINE,
                                 *EVAL_FALSE_ALARMS), *sets()))

        self.assertIn("Promotion Gate failed", lost)
        self.assertIn("1 fall lost (1 on the held-out session, 0 on the Eval "
                      "Alerts)", lost)
        self.assertIn("Promotion Gate failed", one)
        self.assertIn("1 fewer False Alarm, fewer than 2", one)

    def test_window_auc_on_a_set_with_one_kind_of_clip_is_not_a_number(self):
        report = judge(DEPLOYED, DEPLOYED, [fall(0.30)], [])

        self.assertIsNone(report.heldout.deployed.auc)
        self.assertIsNone(report.evals.candidate.auc)


class WindowAucLabelsSubjectsAsTheOutcomeDoesTest(unittest.TestCase):
    """A Genuine Outcome is about the subject that raised the Alert alone
    (ADR 0002); a False Alarm is about everyone in it."""

    #: Fires on the resident lying at 0.50 and on nobody else.
    HEAD = Calls.at(0.50)

    def auc(self, *evals: Clip) -> float | None:
        return promotion_gate.judge(DECISION, self.HEAD, self.HEAD, [],
                                    list(evals), seq_len=WINDOW
                                    ).evals.candidate.auc

    def test_a_carer_beside_a_genuine_fall_is_left_out(self):
        # Kneeling beside the resident who fell: a Controlled Descent.
        alone = Clip(frames(descending(0.50)), fall=True)
        with_carer = Clip(frames(descending(0.50), descending(0.56)),
                          fall=True, raised_by=1)

        self.assertEqual(self.auc(with_carer, descent(0.60)),
                         self.auc(alone, descent(0.60)))

    def test_without_raised_by_every_subject_of_a_fall_clip_is_a_fall(self):
        alone = Clip(frames(descending(0.50)), fall=True)
        with_carer = Clip(frames(descending(0.50), descending(0.56)),
                          fall=True)

        self.assertNotEqual(self.auc(with_carer, descent(0.60)),
                            self.auc(alone, descent(0.60)))

    def test_everyone_in_a_false_alarm_is_an_adl(self):
        # The head fires on the second person lying down at 0.66: an ADL
        # windowed high, which ranks worse than the false alarm alone.
        self.HEAD = Calls.at(0.50, 0.66)
        genuine = Clip(frames(descending(0.50)), fall=True, raised_by=1)
        alone = Clip(frames(descending(0.60)), fall=False, raised_by=1)
        both = Clip(frames(descending(0.60), descending(0.66)), fall=False,
                    raised_by=1)

        self.assertLess(self.auc(genuine, both), self.auc(genuine, alone))

    def test_a_genuine_alert_is_caught_whoever_raises_the_alarm(self):
        # The room's alarm is what a responder sees, so a fall clip counts
        # caught on anyone's alarm; only window AUC follows `raised_by`.
        carer_only = Calls.at(0.56)
        clip = Clip(frames(descending(0.50), descending(0.56)), fall=True,
                    raised_by=1)

        report = promotion_gate.judge(DECISION, carer_only, carer_only, [],
                                      [clip], seq_len=WINDOW)

        self.assertEqual(report.evals.candidate.falls, 1)


class TheFallDecisionRunsUnderneathTest(unittest.TestCase):
    """The gate scores each head under the deployed rule, not per window."""

    def raised(self, clip: Clip, decision: FallDecision = DECISION,
               head=None) -> bool:
        report = promotion_gate.judge(decision, Calls.at(), head or Calls.at(0.70),
                                      [clip], [], seq_len=WINDOW)
        s = report.heldout.candidate
        return bool(s.falls + s.false_alarms)

    def test_one_window_called_a_fall_is_averaged_away_by_the_rolling_buffer(self):
        # Down for exactly one window's step, then up again: one window ends
        # on a lying body, once the buffer is full of upright ones.
        rows = [body(0.5, 0.70)] * (3 * HZ)
        rows += descending(0.70)[:WINDOW + HZ // 2]
        rows += [body(0.85, 0.70, lying=1.0)] * HZ
        rows += [body(0.5, 0.70)] * (2 * WINDOW)
        stumble = Clip(frames(rows), fall=False)

        self.assertFalse(self.raised(stumble))
        self.assertTrue(self.raised(stumble, DECISION.replace(buffer=1)))

    def test_someone_lying_throughout_is_held_back_by_the_descent_gate(self):
        # Shuffling across the floor, so not a still subject, but never
        # going down.
        rows = [body(0.85, 0.60 + 0.1 * f / WINDOW, lying=1.0)
                for f in range(WINDOW)]
        rows += [body(0.85, 0.70, lying=1.0)] * (2 * WINDOW)
        lying = Clip(frames(rows), fall=False)

        self.assertFalse(self.raised(lying))
        self.assertTrue(self.raised(lying, DECISION.replace(min_descent=0.0)))

    def test_each_subject_is_decided_on_their_own(self):
        # A fall behind a bystander whose box is larger: the largest-box rule
        # decides on the bystander.
        behind = Clip(frames([body(0.5, 0.20)] * (4 * WINDOW),
                             descending(0.70)), fall=True)

        self.assertTrue(self.raised(behind))
        self.assertFalse(self.raised(behind,
                                     DECISION.replace(subjects="largest")))


if __name__ == "__main__":
    unittest.main()
