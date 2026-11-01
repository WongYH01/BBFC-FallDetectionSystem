"""fallcore.extract.subject_row / sightings / largest_row: pose -> cache rows.

The functions are the seam. These tests hand it fake pose results shaped like
an Ultralytics `Results` (boxes, keypoints, orig_shape), so no pose model runs.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from fallcore.extract import largest_row, sightings, subject_row


class FakeBoxes:
    def __init__(self, xyxy, ids=None):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4)
        self.id = None if ids is None else torch.tensor(ids, dtype=torch.float32)

    def __len__(self):
        return len(self.xyxy)


def fake_result(people, height=200, width=400, ids=None):
    """A pose result for `people`, each `(box_xyxy, (x, y, conf))`.

    Every keypoint of a person sits at the same pixel, which is enough to see
    which person was picked and how the pixel was converted. `ids` are the
    tracker's, as `model.track` attaches them; None is an untracked result.
    """
    data = torch.tensor([[kp] * 17 for _, kp in people],
                        dtype=torch.float32).reshape(-1, 17, 3)
    return SimpleNamespace(boxes=FakeBoxes([box for box, _ in people], ids),
                           keypoints=SimpleNamespace(data=data),
                           orig_shape=(height, width))


class SubjectRowTest(unittest.TestCase):
    def test_the_largest_person_is_the_subject_in_cache_convention(self):
        small = ((0, 0, 10, 10), (40.0, 20.0, 0.3))
        large = ((100, 0, 400, 200), (300.0, 50.0, 0.9))

        row, aspect = subject_row(fake_result([small, large]))

        self.assertEqual(row.shape, (17, 3))
        self.assertEqual(row.dtype, np.float32)
        np.testing.assert_allclose(row[:, 0], 0.25)   # y / height
        np.testing.assert_allclose(row[:, 1], 0.75)   # x / width
        np.testing.assert_allclose(row[:, 2], 0.9)
        self.assertEqual(aspect, 2.0)

    def test_an_empty_frame_is_a_row_of_zeros(self):
        row, aspect = subject_row(fake_result([]))

        np.testing.assert_array_equal(row, np.zeros((17, 3), dtype=np.float32))
        self.assertEqual(aspect, 2.0)

    def test_a_subject_without_keypoints_is_a_row_of_zeros(self):
        result = fake_result([((0, 0, 50, 50), (10.0, 10.0, 0.8))])
        result.keypoints = None

        row, _ = subject_row(result)

        np.testing.assert_array_equal(row, np.zeros((17, 3), dtype=np.float32))

    def test_keypoints_off_the_frame_are_clipped_to_its_edge(self):
        row, _ = subject_row(fake_result([((0, 0, 50, 50), (-20.0, 260.0, 0.4))]))

        np.testing.assert_allclose(row[:, 0], 1.0)
        np.testing.assert_allclose(row[:, 1], 0.0)

    def test_a_zero_height_frame_does_not_divide_by_zero(self):
        _, aspect = subject_row(fake_result([], height=0, width=320))

        self.assertEqual(aspect, 320.0)


class SightingsTest(unittest.TestCase):
    def test_every_tracked_person_is_a_sighting_in_cache_convention(self):
        small = ((0, 0, 40, 20), (40.0, 20.0, 0.3))
        large = ((100, 0, 400, 200), (300.0, 50.0, 0.9))

        found, aspect = sightings(fake_result([small, large], ids=[4, 9]))

        self.assertEqual([s.track_id for s in found], [4, 9])
        np.testing.assert_allclose(found[0].row[:, 0], 0.1)    # y / height
        np.testing.assert_allclose(found[0].row[:, 1], 0.1)    # x / width
        np.testing.assert_allclose(found[1].row[:, 2], 0.9)
        self.assertEqual(found[0].centre, (0.05, 0.05))
        self.assertEqual(found[1].centre, (0.625, 0.5))
        self.assertAlmostEqual(found[1].area, 0.75)
        self.assertEqual(aspect, 2.0)

    def test_an_untracked_result_has_no_sightings(self):
        found, aspect = sightings(fake_result([((0, 0, 50, 50), (10.0, 10.0, 0.8))]))

        self.assertEqual(found, [])
        self.assertEqual(aspect, 2.0)

    def test_an_empty_frame_has_no_sightings(self):
        found, _ = sightings(fake_result([], ids=[]))

        self.assertEqual(found, [])


class LargestRowTest(unittest.TestCase):
    """A frame's sightings -> the row `subject_row` gives for the same frame,
    for a replay that holds sightings rather than a pose result."""

    def test_the_subject_is_the_one_subject_row_picks(self):
        small = ((0, 0, 40, 20), (40.0, 20.0, 0.3))
        large = ((100, 0, 400, 200), (300.0, 50.0, 0.9))
        result = fake_result([small, large], ids=[4, 9])

        np.testing.assert_array_equal(largest_row(sightings(result)[0]),
                                      subject_row(result)[0])

    def test_nobody_in_view_is_a_row_of_zeros(self):
        row = largest_row([])

        np.testing.assert_array_equal(row, np.zeros((17, 3), dtype=np.float32))
        self.assertEqual(row.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
