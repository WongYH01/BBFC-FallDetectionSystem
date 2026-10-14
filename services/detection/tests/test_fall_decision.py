"""fallcore.fall_decision: the deployed Fall Decision, from an environment.

The record is the one seam. These tests hand `FallDecision.from_env` a plain
dict in place of the process environment, so no checkpoint, torch or camera is
needed; only the build test at the bottom loads a model, and it skips when the
deployed checkpoint is not on disk.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

from fallcore import config as cfg
from fallcore.fall_decision import FallDecision


class DeployedDefaultsTest(unittest.TestCase):
    def test_an_empty_environment_is_the_deployed_decision(self):
        d = FallDecision.from_env({})

        self.assertEqual([p.name for p in d.checkpoints],
                         ["augnone_ms_coords_hn_s99.pt"])
        self.assertEqual([p.name for p in d.heads],
                         ["probe_augnone_ms_coords_hn_s99.npz"])
        self.assertEqual(d.buffer, 4)
        self.assertEqual(d.threshold, 0.5)
        self.assertEqual(d.clear_below, 0.2)
        self.assertEqual(d.clear_windows, 2)
        self.assertEqual(d.min_descent, 0.10)
        self.assertEqual(d.frame_stride, 2)
        self.assertEqual(d.step, 15)


class EnvironmentOverrideTest(unittest.TestCase):
    def test_each_rule_variable_overrides_only_its_own_field(self):
        cases = [
            ("BUFFER_LEN", "6", "buffer", 6),
            ("FALL_THRESHOLD", "0.65", "threshold", 0.65),
            ("FALL_CLEAR_BELOW", "0.1", "clear_below", 0.1),
            ("FALL_CLEAR_WINDOWS", "3", "clear_windows", 3),
            ("FALL_MIN_DESCENT", "0", "min_descent", 0.0),
            ("STREAM_FRAME_STRIDE", "1", "frame_stride", 1),
        ]
        default = FallDecision.from_env({})
        for var, raw, field, expected in cases:
            with self.subTest(var=var):
                d = FallDecision.from_env({var: raw})

                self.assertEqual(getattr(d, field), expected)
                self.assertEqual(d, FallDecision.from_env({}).replace(
                    **{field: expected}))
                self.assertNotEqual(d, default)

    def test_an_explicit_change_layers_on_top_of_the_environment(self):
        d = FallDecision.from_env({"FALL_MIN_DESCENT": "0"}).replace(threshold=0.7)

        self.assertEqual(d.min_descent, 0.0)
        self.assertEqual(d.threshold, 0.7)
        self.assertEqual(d.buffer, 4)

    def test_an_empty_calibration_head_runs_the_checkpoint_uncalibrated(self):
        d = FallDecision.from_env({"CALIBRATION_HEAD": ""})

        self.assertEqual([p.name for p in d.checkpoints],
                         ["augnone_ms_coords_hn_s99.pt"])
        self.assertEqual(d.heads, (None,))

    def test_ensemble_checkpoints_replace_the_pair(self):
        d = FallDecision.from_env({
            "ENSEMBLE_CKPTS": os.pathsep.join([str(Path("w") / "a.pt"),
                                               str(Path("w") / "b.pt")]),
            "CALIBRATION_HEAD": "",
        })

        self.assertEqual(d.checkpoints, (Path("w") / "a.pt", Path("w") / "b.pt"))
        self.assertEqual(d.heads, (None, None))

    def test_a_calibration_head_path_is_used_as_given(self):
        d = FallDecision.from_env({"CALIBRATION_HEAD": " other/probe.npz "})

        self.assertEqual(d.heads, (Path("other/probe.npz"),))


class SummaryTest(unittest.TestCase):
    def test_the_summary_names_the_pair_and_every_setting(self):
        line = str(FallDecision())

        self.assertNotIn("\n", line)
        for part in ("augnone_ms_coords_hn_s99.pt",
                     "probe_augnone_ms_coords_hn_s99.npz", "buffer=4",
                     "threshold=0.5", "clear_below=0.2", "clear_windows=2",
                     "min_descent=0.1", "frame_stride=2", "step=15"):
            self.assertIn(part, line)

    def test_an_uncalibrated_checkpoint_says_so(self):
        line = str(FallDecision.from_env({"CALIBRATION_HEAD": ""}))

        self.assertIn("augnone_ms_coords_hn_s99.pt (own classifier)", line)


class RefusedOverrideTest(unittest.TestCase):
    def test_a_bad_value_is_refused_and_names_its_variable(self):
        cases = [
            ("FALL_THRESHOLD", "high"),
            ("FALL_THRESHOLD", "1.5"),
            ("BUFFER_LEN", "0"),
            ("BUFFER_LEN", "2.5"),
            ("FALL_CLEAR_BELOW", "-0.1"),
            ("FALL_CLEAR_WINDOWS", "0"),
            ("FALL_MIN_DESCENT", "-1"),
            ("STREAM_FRAME_STRIDE", "0"),
        ]
        for var, raw in cases:
            with self.subTest(var=var, raw=raw):
                with self.assertRaises(ValueError) as caught:
                    FallDecision.from_env({var: raw})

                self.assertIn(var, str(caught.exception))

    def test_more_checkpoints_than_heads_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            FallDecision.from_env({"ENSEMBLE_CKPTS": os.pathsep.join(["a.pt", "b.pt"])})

        self.assertIn("CALIBRATION_HEAD", str(caught.exception))

    def test_a_record_cannot_be_built_out_of_range(self):
        with self.assertRaises(ValueError):
            FallDecision().replace(buffer=0)



class WeightsManifestTest(unittest.TestCase):
    def test_fetch_weights_ships_the_deployed_pair(self):
        sys.path.insert(0, str(cfg.ROOT / "scripts"))
        try:
            from fetch_weights import FILES
        finally:
            sys.path.remove(str(cfg.ROOT / "scripts"))
        shipped = {target for target, _sha in FILES.values()}

        for path in FallDecision().checkpoints + FallDecision().heads:
            with self.subTest(file=path.name):
                self.assertIn(path.relative_to(cfg.ROOT).as_posix(), shipped)


_DEPLOYED = FallDecision()
_HAVE_WEIGHTS = all(p.exists() for p in _DEPLOYED.checkpoints + _DEPLOYED.heads)


@unittest.skipUnless(_HAVE_WEIGHTS, "deployed checkpoint and head not fetched "
                                    "(scripts/fetch_weights.py)")
class BuildTest(unittest.TestCase):
    def test_the_stream_runs_the_decision_it_was_built_from(self):
        d = FallDecision().replace(buffer=6, threshold=0.6, clear_below=0.1,
                                   clear_windows=3, min_descent=0.05,
                                   frame_stride=1)

        stream = d.build()

        self.assertEqual(stream.classifier.names,
                         ["augnone_ms_coords_hn_s99+head"])
        self.assertEqual(stream.decision.buffer, 6)
        self.assertEqual(stream.decision.threshold, 0.6)
        self.assertEqual(stream.decision.clear_below, 0.1)
        self.assertEqual(stream.decision.clear_windows, 3)
        self.assertEqual(stream.min_descent, 0.05)
        self.assertEqual(stream.buffer.frame_stride, 1)

    def test_the_window_length_is_the_checkpoints(self):
        stream = FallDecision().build()

        # The deployed checkpoint was trained on 60-row windows (4 s at
        # frame stride 2), which is also what makes a window due at 120 frames.
        self.assertEqual(stream.buffer.seq_len, 60)
        self.assertEqual(stream.buffer.frames_per_window, 120)

    def test_picam_scores_the_deployed_decision_unless_told_otherwise(self):
        from fallcore import picam

        det = picam.load_detector("ms coords + room head")

        self.assertEqual([p.name for p in det.checkpoints],
                         ["augnone_ms_coords_hn_s99.pt"])
        self.assertEqual([p.name for p in det.heads],
                         ["probe_augnone_ms_coords_hn_s99.npz"])
        self.assertEqual(det.stream.min_descent, 0.10)
        self.assertEqual(det.stream.decision.buffer, 4)
        self.assertEqual(det.stream.decision.threshold, 0.5)
        self.assertEqual(det.stream.decision.clear_below, 0.2)
        self.assertEqual(det.stream.decision.clear_windows, 2)

    def test_picam_can_still_score_the_ungated_rule(self):
        from fallcore import picam

        det = picam.load_detector("ms coords + room head", min_descent=0.0)

        self.assertEqual(det.stream.min_descent, 0.0)


class PicamConstantsTest(unittest.TestCase):
    def test_the_v2_constants_notebooks_read_are_the_deployed_values(self):
        from fallcore import picam

        self.assertEqual((picam.V2_BUFFER, picam.V2_CLEAR_BELOW,
                          picam.V2_CLEAR_WINDOWS, picam.V2_MIN_DESCENT),
                         (4, 0.2, 2, 0.10))


if __name__ == "__main__":
    unittest.main()
