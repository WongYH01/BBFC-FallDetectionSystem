"""fallcore.keypoint_file: the keypoint file sent with a fall alert.

What matters is that it reads back as the frames that were written, in the
layout the module documents, and that the notification service will accept
it: a `.npz` whose first bytes are the zip header it checks.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from fallcore import config as cfg
from fallcore import keypoint_file
from fallcore.extract import Sighting
from fallcore.keypoint_file import Frame, Person


def row(fill: float) -> np.ndarray:
    """A `(17, 3)` cache row with every value `fill`."""
    return np.full((cfg.N_KEYPOINTS, 3), fill, dtype=np.float32)


def person(track_id: int, fill: float, fallen: bool = False,
           subject: int = -1) -> Person:
    return Person(track_id=track_id, keypoints=row(fill),
                  centre=(0.25 * track_id, 0.5), area=0.01 * track_id,
                  fallen=fallen, subject=subject)


class KeypointFileTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "fall_1.npz"

    def tearDown(self):
        self._tmp.cleanup()

    def _frames(self) -> list[Frame]:
        # Two people walking, a frame where the tracker has nobody, then one
        # of them down with the alarm up -- under a new tracker id that a
        # reconnect has joined to the same subject.
        return [
            Frame(t=100.0, aspect=16 / 9, alarm=False, score=0.1,
                  people=(person(1, 0.1, subject=1), person(2, 0.2, subject=2)),
                  lead=2),
            Frame(t=100.1, aspect=16 / 9, alarm=False, score=0.2, lead=1),
            Frame(t=100.2, aspect=16 / 9, alarm=True, score=0.8,
                  people=(person(3, 0.3, fallen=True, subject=1),
                          person(2, 0.4, subject=2)),
                  lead=1),
        ]

    def test_frames_read_back_in_the_documented_layout(self):
        keypoint_file.save(self.path, self._frames(), trigger=2)

        with np.load(self.path) as f:
            self.assertEqual(int(f["version"]), keypoint_file.FORMAT_VERSION)
            self.assertEqual(int(f["trigger"]), 2)
            np.testing.assert_allclose(f["t"], [100.0, 100.1, 100.2])
            self.assertEqual(f["t"].dtype, np.float64)
            np.testing.assert_allclose(f["aspect"], [16 / 9] * 3, rtol=1e-6)
            np.testing.assert_array_equal(f["alarm"], [False, False, True])
            np.testing.assert_allclose(f["score"], [0.1, 0.2, 0.8], rtol=1e-6)
            np.testing.assert_array_equal(f["lead"], [2, 1, 1])
            # Whom the Alert's Outcome is about: the lead when it latched.
            self.assertEqual(int(f["raised_by"]), 1)

            # One row per sighting, frame-major; the empty frame has none.
            np.testing.assert_array_equal(f["frame"], [0, 0, 2, 2])
            np.testing.assert_array_equal(f["track_id"], [1, 2, 3, 2])
            np.testing.assert_array_equal(f["subject"], [1, 2, 1, 2])
            self.assertEqual(f["keypoints"].shape, (4, cfg.N_KEYPOINTS, 3))
            self.assertEqual(f["keypoints"].dtype, np.float32)
            np.testing.assert_allclose(f["keypoints"][:, 0, 0],
                                       [0.1, 0.2, 0.3, 0.4], rtol=1e-6)
            np.testing.assert_allclose(f["centre"],
                                       [[0.25, 0.5], [0.5, 0.5],
                                        [0.75, 0.5], [0.5, 0.5]])
            np.testing.assert_allclose(f["area"], [0.01, 0.02, 0.03, 0.02],
                                       rtol=1e-6)
            np.testing.assert_array_equal(f["fallen"],
                                          [False, False, True, False])

    def test_one_subjects_stream_spans_its_tracker_ids(self):
        # What a trainer does with the file: the faller's rows by subject,
        # zeros where they were not seen.
        keypoint_file.save(self.path, self._frames(), trigger=2)
        with np.load(self.path) as f:
            stream = np.zeros((len(f["t"]), cfg.N_KEYPOINTS, 3), np.float32)
            mine = f["subject"] == 1
            stream[f["frame"][mine]] = f["keypoints"][mine]
        np.testing.assert_allclose(stream[:, 0, 0], [0.1, 0.0, 0.3], rtol=1e-6)

    def test_it_is_a_zip_the_notification_service_accepts(self):
        # KeypointsController takes a .npz by its extension and its first
        # four bytes, PK\x03\x04, and refuses anything else with 415.
        keypoint_file.save(self.path, self._frames(), trigger=2)
        self.assertEqual(self.path.read_bytes()[:4], b"PK\x03\x04")

    def test_loads_without_pickle(self):
        # Nothing in the file is an object array, so a reader never has to
        # trust it enough to unpickle.
        keypoint_file.save(self.path, self._frames(), trigger=2)
        with np.load(self.path, allow_pickle=False) as f:
            for name in f.files:
                self.assertNotEqual(f[name].dtype, object, name)

    def test_a_clip_with_nobody_in_it_still_has_well_shaped_arrays(self):
        frames = [Frame(t=1.0, aspect=1.5, alarm=False, score=0.0),
                  Frame(t=2.0, aspect=1.5, alarm=True, score=0.9)]
        keypoint_file.save(self.path, frames, trigger=1)
        with np.load(self.path) as f:
            self.assertEqual(f["t"].shape, (2,))
            # No per-person decision: nobody in particular raised it.
            self.assertEqual(int(f["raised_by"]), -1)
            np.testing.assert_array_equal(f["lead"], [-1, -1])
            self.assertEqual(f["frame"].shape, (0,))
            self.assertEqual(f["keypoints"].shape, (0, cfg.N_KEYPOINTS, 3))
            self.assertEqual(f["centre"].shape, (0, 2))
            self.assertEqual(f["fallen"].shape, (0,))

    def test_written_under_its_final_name_only_once_complete(self):
        keypoint_file.save(self.path, self._frames(), trigger=2)
        self.assertEqual([p.name for p in self.path.parent.iterdir()],
                         ["fall_1.npz"])

    def test_a_failed_write_leaves_nothing_behind(self):
        with mock.patch.object(np, "savez_compressed",
                               side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                keypoint_file.save(self.path, self._frames(), trigger=2)
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_trigger_must_be_one_of_the_frames(self):
        with self.assertRaises(ValueError):
            keypoint_file.save(self.path, self._frames(), trigger=3)
        with self.assertRaises(ValueError):
            keypoint_file.save(self.path, [], trigger=0)

    def test_people_from_sightings(self):
        seen = [Sighting(track_id=4, row=row(0.5), centre=(0.1, 0.2), area=0.3),
                Sighting(track_id=9, row=row(0.6), centre=(0.7, 0.8), area=0.05)]
        people = keypoint_file.people(seen, fallen_of=lambda tid: tid == 9,
                                      subject_of={4: 2, 9: None}.get)

        self.assertEqual([p.track_id for p in people], [4, 9])
        self.assertEqual([p.fallen for p in people], [False, True])
        self.assertEqual([p.subject for p in people], [2, -1])
        self.assertEqual(people[1].centre, (0.7, 0.8))
        self.assertAlmostEqual(people[1].area, 0.05)
        np.testing.assert_array_equal(people[0].keypoints, row(0.5))

    def test_people_without_per_person_decisions_have_no_subject(self):
        seen = [Sighting(track_id=4, row=row(0.5), centre=(0.1, 0.2), area=0.3)]
        [p] = keypoint_file.people(seen, fallen_of=lambda tid: True)
        self.assertEqual(p.subject, -1)
        self.assertTrue(p.fallen)


if __name__ == "__main__":
    unittest.main()
