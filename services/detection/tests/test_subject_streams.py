"""fallcore.stream.SubjectStreams: one fall decision per tracked person.

The seam is `observe(sightings)`. These tests feed synthetic bodies through the
real buffers, rolling rule and descent gate, with a stub classifier in place of
the checkpoint: it calls a window a fall when the body in its last row is
lying down, which is enough to see whose keypoints were decided on and when.
"""
from __future__ import annotations

import unittest

import numpy as np

from fallcore import config as cfg
from fallcore.extract import Sighting
from fallcore.stream import EnsembleStream, SubjectStreams

HZ = 15          # rows per second, the live rate at frame_stride 1
WINDOW = 60      # rows per window


class LyingIsAFall:
    """Stub classifier: P(fall) 0.99 when the last row is a body lying down."""

    names = ["stub"]
    heads = [None]

    def proba(self, window, aspect=1.0):
        last = window[-1]
        seen = last[:, 2] > 0.3
        lying = seen.any() and np.ptp(last[seen, 0]) < 0.2
        p = 0.99 if lying else 0.01
        return p, [p]


class AlwaysAFall:
    """Stub classifier that scores every window, empty ones too, as a fall."""

    names = ["stub"]
    heads = [None]

    def proba(self, window, aspect=1.0):
        return 0.99, [0.99]


def body(hip_y: float, x: float = 0.5, lying: float = 0.0) -> np.ndarray:
    """A `(17, 3)` cache row: a body with its hips at `hip_y`.

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


def sighting(track_id: int, row: np.ndarray, area: float = 0.05) -> Sighting:
    seen = row[:, 2] > 0
    return Sighting(track_id=track_id, row=row,
                    centre=(float(row[seen, 1].mean()), float(row[seen, 0].mean())),
                    area=area)


def falling(n_before: int = WINDOW, n_after: int = WINDOW, x: float = 0.5):
    """Rows of someone standing, falling over half a second, then lying."""
    rows = [body(0.5, x) for _ in range(n_before)]
    steps = HZ // 2
    rows += [body(0.5 + 0.35 * (i + 1) / steps, x, lying=(i + 1) / steps)
             for i in range(steps)]
    rows += [body(0.85, x, lying=1.0) for _ in range(n_after)]
    return rows


def walking(n: int, x0: float = 0.1, x1: float = 0.5):
    """Rows of someone upright, crossing from `x0` to `x1`."""
    return [body(0.5, x0 + (x1 - x0) * f / n) for f in range(n)]


def streams(classifier=None, min_descent: float = 0.10,
            **kwargs) -> SubjectStreams:
    template = EnsembleStream(
        [], buffer=1, threshold=0.5, clear_below=0.2, clear_windows=1,
        seq_len=WINDOW, frame_stride=1, step=HZ, min_descent=min_descent,
        classifier=classifier or LyingIsAFall())
    return SubjectStreams(template, **kwargs)


def run(s: SubjectStreams, frames) -> list[bool]:
    """Feed frames of sightings; the room's alarm after each one."""
    out = []
    for sightings in frames:
        s.observe(sightings)
        out.append(s.alarm)
    return out


class EachPersonIsDecidedOnTheirOwnTest(unittest.TestCase):
    def test_a_fall_behind_a_closer_bystander_raises_the_alarm(self):
        fall = falling(x=0.3)
        frames = [[sighting(1, r), sighting(2, body(0.5, x=0.7), area=0.2)]
                  for r in fall]
        s = streams()

        alarms = run(s, frames)

        self.assertTrue(alarms[-1])
        self.assertTrue(s.alarm_of(1))
        self.assertFalse(s.alarm_of(2))

    def test_someone_lying_still_throughout_never_raises_the_alarm(self):
        # Scored a fall on every window, but no window holds a descent.
        frames = [[sighting(3, body(0.85, lying=1.0))]] * (3 * WINDOW)

        self.assertFalse(any(run(streams(), frames)))


class ReconnectionTest(unittest.TestCase):
    def switched(self, at: int, new_id: int = 7, gap: int = 0, shift: float = 0.0):
        """The fall, its tracker id changing to `new_id` at row `at`."""
        frames = []
        for f, r in enumerate(falling()):
            if f < at:
                frames.append([sighting(1, r)])
            elif f < at + gap:
                frames.append([])
            else:
                moved = r.copy()
                moved[:, 1] += shift
                frames.append([sighting(new_id, moved)])
        return frames

    def test_a_new_id_mid_fall_continues_the_lost_subjects_stream(self):
        s = streams()

        alarms = run(s, self.switched(at=WINDOW + 4))

        self.assertTrue(alarms[-1])
        self.assertEqual([x.track_ids for x in s.subjects], [[1, 7]])
        self.assertEqual(s.subject_of(7), s.subjects[0].sid)
        self.assertEqual(s.subject_of(1), s.subject_of(7))
        self.assertIsNone(s.subject_of(99))

    def test_with_reconnection_off_the_new_id_is_a_new_subject(self):
        s = streams(reconnect_dist=0.0)

        run(s, self.switched(at=WINDOW + 4)[:WINDOW + 10])

        self.assertEqual(sorted(x.track_ids for x in s.subjects), [[1], [7]])
        self.assertNotEqual(s.subject_of(1), s.subject_of(7))

    def test_a_new_id_far_from_the_lost_subject_is_a_new_subject(self):
        s = streams()

        run(s, self.switched(at=WINDOW + 4, shift=0.4)[:WINDOW + 10])

        self.assertEqual(len(s.subjects), 2)

    def test_a_new_id_after_the_reconnect_window_is_a_new_subject(self):
        s = streams(reconnect_sec=2.0)

        run(s, self.switched(at=WINDOW + 4, gap=2 * HZ + 1)[:WINDOW + 2 * HZ + 10])

        self.assertEqual(len(s.subjects), 2)

    def test_an_id_change_while_lying_keeps_the_one_alarm_up(self):
        s = streams()
        frames = self.switched(at=WINDOW + HZ + 30)  # well after the landing

        alarms = run(s, frames)
        first = alarms.index(True)

        self.assertTrue(all(alarms[first:]))
        self.assertEqual(len(s.subjects), 1)


class LostSubjectTest(unittest.TestCase):
    def test_a_lost_subject_runs_one_window_then_is_dropped(self):
        s = streams()
        run(s, [[sighting(1, body(0.5))]] * 30)

        run(s, [[]] * (WINDOW - 1))
        self.assertEqual(len(s.subjects), 1)
        run(s, [[]])
        self.assertEqual(len(s.subjects), 0)

    def test_a_fall_is_still_raised_when_the_faller_vanishes_on_landing(self):
        # Lost a few rows after landing, before any window was due on them.
        fall = falling(n_after=3)
        frames = [[sighting(1, r)] for r in fall] + [[]] * WINDOW
        s = streams(classifier=LyingWithinTheLastSecond())

        self.assertTrue(any(run(s, frames)))

    def test_an_alarmed_subject_cannot_hold_the_alarm_past_two_windows(self):
        # A classifier that calls an empty window a fall would otherwise keep
        # a vanished subject's alarm up forever, and the recorder never re-arms.
        s = streams(classifier=AlwaysAFall(), min_descent=0.0)
        self.assertTrue(run(s, [[sighting(1, r)] for r in walking(WINDOW)])[-1])

        alarms = run(s, [[]] * (2 * WINDOW))

        self.assertTrue(alarms[WINDOW - 1])
        self.assertFalse(alarms[-1])
        self.assertEqual(len(s.subjects), 0)

    def test_after_an_outage_only_alarmed_subjects_are_kept(self):
        s = streams()
        frames = [[sighting(1, r), sighting(2, body(0.5, x=0.8))]
                  for r in falling(x=0.2)]
        run(s, frames)

        s.reset_buffer()

        self.assertEqual([x.sid for x in s.subjects], [1])
        self.assertTrue(s.alarm)
        self.assertFalse(s.alarm_of(1))  # ids restart with the new connection


class EmptyIsAFall:
    """Stub classifier: P(fall) 0.99 when the window's last row is empty, the
    "vanished" shortcut."""

    names = ["stub"]
    heads = [None]

    def proba(self, window, aspect=1.0):
        p = 0.99 if not (window[-1][:, 2] > 0).any() else 0.01
        return p, [p]


class StillSubjectTest:
    """Furniture the detector takes for a person, scored a fall. The descent
    gate is off (`min_descent=0`): on real keypoints the jitter of a still
    detection passes it."""

    def test_a_still_detection_scored_a_fall(self):
        frames = [[sighting(1, r), sighting(2, body(0.85, x=0.2, lying=1.0))]
                  for r in walking(3 * WINDOW, x0=0.5, x1=0.9)]

        alarms = run(self.streams(LyingIsAFall()), frames)

        self.assertEqual(any(alarms), self.raises)

    def test_a_still_detection_lost_behind_someone(self):
        chair = sighting(2, body(0.5, x=0.2))
        walker = walking(2 * WINDOW, x0=0.9, x1=0.3)
        frames = ([[sighting(1, r), chair] for r in walker[:WINDOW]]
                  + [[sighting(1, r)] for r in walker[WINDOW:]])

        alarms = run(self.streams(EmptyIsAFall()), frames)

        self.assertEqual(any(alarms), self.raises)


class StillSubjectsCannotRaiseTest(StillSubjectTest, unittest.TestCase):
    raises = False

    def streams(self, classifier):
        return streams(classifier=classifier, min_descent=0.0)

    def test_a_still_body_appearing_while_someone_is_missing_can_raise(self):
        # The faller is lost before landing and comes back lying, as a new id
        # out of reconnect range: the new subject never moves, but may be them.
        gap = 2 * HZ + 10
        frames = ([[sighting(1, r)] for r in walking(WINDOW)] + [[]] * gap
                  + [[sighting(9, body(0.85, x=0.9, lying=1.0))]] * WINDOW)
        s = self.streams(LyingIsAFall())

        alarms = run(s, frames)

        self.assertTrue(alarms[-1])
        self.assertTrue(s.alarm_of(9))


class WithoutTheStillnessRuleTest(StillSubjectTest, unittest.TestCase):
    raises = True

    def streams(self, classifier):
        return streams(classifier=classifier, min_descent=0.0, min_travel=0.0)


class LyingWithinTheLastSecond(LyingIsAFall):
    """A fall when the body lay down in the window's last second."""

    def proba(self, window, aspect=1.0):
        for row in window[-HZ:]:
            p = super().proba(row[None], aspect)[0]
            if p > 0.5:
                return p, [p]
        return 0.01, [0.01]


class StateTest(unittest.TestCase):
    def test_the_state_reports_the_alarmed_subject_and_the_rooms_windows(self):
        s = streams()
        frames = [[sighting(1, body(0.5, x=0.8)), sighting(2, r)]
                  for r in falling(x=0.2)]
        run(s, frames)

        st = s.state()

        self.assertTrue(st["alarm"])
        self.assertEqual(st["subject"], 2)
        self.assertEqual(sorted(x["subject"] for x in st["subjects"]), [1, 2])
        self.assertEqual(st["n_windows"],
                         sum(x.stream.decision.n_windows for x in s.subjects))

    def test_an_empty_room_reports_no_subject_and_no_alarm(self):
        st = streams().state()

        self.assertFalse(st["alarm"])
        self.assertIsNone(st["subject"])
        self.assertEqual(st["subjects"], [])
        self.assertEqual(st["n_windows"], 0)


if __name__ == "__main__":
    unittest.main()
