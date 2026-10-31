"""scripts/fit_calibration_head.py: what a leave-one-subject-out fold fits on.

The seam is `fold_fitting_set`, the windows a fold's camera head is fitted on.
These tests stand in for the two things that need data and weights: the
keypoint cache (`load`) and the frozen backbone (`embed`). Every subject's
keypoints carry its own id, and the stub embedding of a window is the largest
coordinate in it, so an embedding says whose keypoints a window was cut from.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fit_calibration_head import NobodyNegatives, fold_fitting_set

from fallcore import config as cfg

SEQ_LEN = 60
SUBJECTS = (1, 2, 3)


def subject_of(video_id: str) -> int:
    return int(video_id.split("_")[0][1:])


def load(scale, video_id):
    """Keypoint cache stub: 200 rows of every joint at (subject, subject)."""
    kp = np.full((200, cfg.N_KEYPOINTS, 3), float(subject_of(video_id)),
                 dtype=np.float32)
    kp[..., 2] = 0.9
    return kp


def embed(windows):
    """Backbone stub: a window's embedding is its largest coordinate."""
    return windows[..., :2].reshape(len(windows), -1).max(axis=1, keepdims=True)


def room():
    """Three subjects, two windows per clip; subject 1's ADL clips come first.

    Each subject has a fall clip and six ADL clips, so any one fold still has
    six ADL clips of its own to cut partly-zeroed windows from.
    """
    rows = []
    for s in SUBJECTS:
        for k in range(6):
            rows += [(f"s{s}_adl{k}", 0, 0), (f"s{s}_adl{k}", 0, 15)]
    for s in SUBJECTS:
        rows += [(f"s{s}_fall", 1, 0), (f"s{s}_fall", 1, 15)]
    clips = pd.DataFrame(rows, columns=["video_id", "label", "start"])
    subj = clips.video_id.map(subject_of).values
    emb = subj[:, None].astype(np.float32)
    return clips, subj, emb


class LeaveOneSubjectOutTest(unittest.TestCase):
    def test_no_window_a_fold_fits_on_comes_from_the_subject_it_scores(self):
        clips, subj, emb = room()
        negatives = NobodyNegatives(SimpleNamespace(seq_len=SEQ_LEN), embed,
                                    copies=2, load=load)
        keep = np.ones(len(clips), dtype=bool)

        for held_out in SUBJECTS:
            with self.subTest(held_out=held_out):
                e, lab, _ = fold_fitting_set(subj != held_out, clips, emb,
                                             clips.label.values, keep,
                                             negatives=negatives)

                seen = set(np.round(e[:, 0]).astype(int)) - {0}
                self.assertEqual(seen, set(SUBJECTS) - {held_out})


if __name__ == "__main__":
    unittest.main()
