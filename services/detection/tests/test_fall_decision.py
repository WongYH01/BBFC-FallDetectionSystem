"""fallcore.fall_decision: the deployed Fall Decision, from an environment.

The record is the one seam. These tests hand `FallDecision.from_env` a plain
dict in place of the process environment, so no checkpoint, torch or camera is
needed; the given-classifier tests build streams on a stub, and only the build
test at the bottom loads a model, skipping when the deployed checkpoint is not
on disk.
"""
from __future__ import annotations

import sys
import unittest

from fallcore import config as cfg
from fallcore.fall_decision import FallDecision


class DeployedDefaultsTest(unittest.TestCase):
    def test_an_empty_environment_is_the_deployed_decision(self):
        d = FallDecision.from_env({})

        self.assertEqual([p.name for p in d.checkpoints],
                         ["augnone_ms_coords_hn_s99.pt"])
        self.assertEqual([p.name for p in d.heads],
                         ["probe_augnone_ms_coords_hn_s99_yolo26m-int8.npz"])
        self.assertEqual(d.pose, "yolo26m_r640_ortint8rr")
        self.assertEqual(d.buffer, 4)
        self.assertEqual(d.threshold, 0.5)
        self.assertEqual(d.clear_below, 0.2)
        self.assertEqual(d.clear_windows, 2)
        self.assertEqual(d.min_descent, 0.10)
        self.assertEqual(d.frame_stride, 2)
        self.assertEqual(d.step, 15)
        self.assertEqual(d.subjects, "tracked")
        self.assertEqual(d.reconnect_sec, 2.0)
        self.assertEqual(d.reconnect_dist, 0.35)


class EnvironmentOverrideTest(unittest.TestCase):
    def test_each_rule_variable_overrides_only_its_own_field(self):
        cases = [
            ("BUFFER_LEN", "6", "buffer", 6),
            ("FALL_THRESHOLD", "0.65", "threshold", 0.65),
            ("FALL_CLEAR_BELOW", "0.1", "clear_below", 0.1),
            ("FALL_CLEAR_WINDOWS", "3", "clear_windows", 3),
            ("FALL_MIN_DESCENT", "0", "min_descent", 0.0),
            ("STREAM_FRAME_STRIDE", "1", "frame_stride", 1),
            ("FALL_SUBJECTS", "largest", "subjects", "largest"),
            ("FALL_RECONNECT_SEC", "3", "reconnect_sec", 3.0),
            ("FALL_RECONNECT_DIST", "0", "reconnect_dist", 0.0),
        ]
        default = FallDecision.from_env({})
        for var, raw, field, expected in cases:
            with self.subTest(var=var):
                d = FallDecision.from_env({var: raw})

                self.assertEqual(getattr(d, field), expected)
                self.assertEqual(d, FallDecision.from_env({}).replace(
                    **{field: expected}))
                self.assertNotEqual(d, default)

    def test_the_subject_rule_ignores_case_and_spaces(self):
        d = FallDecision.from_env({"FALL_SUBJECTS": " Largest "})

        self.assertEqual(d.subjects, "largest")

    def test_an_explicit_change_layers_on_top_of_the_environment(self):
        d = FallDecision.from_env({"FALL_MIN_DESCENT": "0"}).replace(threshold=0.7)

        self.assertEqual(d.min_descent, 0.0)
        self.assertEqual(d.threshold, 0.7)
        self.assertEqual(d.buffer, 4)

    def test_the_variables_that_swapped_the_models_are_refused(self):
        # Even empty: `CALIBRATION_HEAD=` used to mean "uncalibrated", and an
        # old .env must not be silently ignored.
        for var in ("POSE_SCALE", "POSE_MODEL", "POSE_IMGSZ", "ENSEMBLE_CKPTS",
                    "CALIBRATION_HEAD"):
            with self.subTest(var=var):
                with self.assertRaises(ValueError) as caught:
                    FallDecision.from_env({var: ""})

                self.assertIn(var, str(caught.exception))


class SummaryTest(unittest.TestCase):
    def test_the_summary_names_the_pair_and_every_setting(self):
        line = str(FallDecision())

        self.assertNotIn("\n", line)
        for part in ("augnone_ms_coords_hn_s99.pt",
                     "probe_augnone_ms_coords_hn_s99_yolo26m-int8.npz",
                     "on yolo26m_r640_ortint8rr", "buffer=4",
                     "threshold=0.5", "clear_below=0.2", "clear_windows=2",
                     "min_descent=0.1", "frame_stride=2", "step=15",
                     "subjects=tracked", "reconnect_sec=2.0",
                     "reconnect_dist=0.35"):
            self.assertIn(part, line)

    def test_an_uncalibrated_checkpoint_says_so(self):
        line = str(FallDecision().replace(heads=(None,)))

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
            ("FALL_SUBJECTS", "everyone"),
            ("FALL_RECONNECT_SEC", "-1"),
            ("FALL_RECONNECT_DIST", "far"),
        ]
        for var, raw in cases:
            with self.subTest(var=var, raw=raw):
                with self.assertRaises(ValueError) as caught:
                    FallDecision.from_env({var: raw})

                self.assertIn(var, str(caught.exception))

    def test_more_checkpoints_than_heads_is_refused(self):
        with self.assertRaises(ValueError):
            FallDecision().replace(checkpoints=(cfg.CKPT_DIR / "a.pt",
                                                cfg.CKPT_DIR / "b.pt"))

    def test_a_record_cannot_be_built_out_of_range(self):
        with self.assertRaises(ValueError):
            FallDecision().replace(buffer=0)


class GivenClassifierTest(unittest.TestCase):
    """`build` on a classifier already built, as the Promotion Gate runs a
    Candidate Head: the rule is the record's, and nothing is loaded."""

    class Stub:
        names = ["stub"]

        def proba(self, window, aspect=1.0):
            return 0.5, [0.5]

    def test_the_rule_is_the_records_on_the_given_classifier(self):
        stub = self.Stub()
        d = FallDecision().replace(buffer=6, min_descent=0.05, frame_stride=1,
                                   reconnect_dist=0.2)

        streams = d.build_camera(classifier=stub, seq_len=45)

        self.assertIs(streams.classifier, stub)
        self.assertEqual(streams.reconnect_dist, 0.2)
        self.assertEqual(streams.template.decision.buffer, 6)
        self.assertEqual(streams.template.min_descent, 0.05)
        self.assertEqual(streams.template.buffer.seq_len, 45)

    def test_a_given_classifier_needs_its_window_length(self):
        with self.assertRaisesRegex(ValueError, "seq_len"):
            FallDecision().build(classifier=self.Stub())



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

    def test_fetch_weights_ships_the_pose_backbone_and_nothing_else(self):
        from fallcore.pose_backbone import DEPLOYED

        sys.path.insert(0, str(cfg.ROOT / "scripts"))
        try:
            from fetch_weights import FILES
        finally:
            sys.path.remove(str(cfg.ROOT / "scripts"))
        shipped = {target for target, _sha in FILES.values()}
        d = FallDecision()

        self.assertEqual(shipped, {p.relative_to(cfg.ROOT).as_posix()
                                   for p in (DEPLOYED.model, *d.checkpoints,
                                             *d.heads)})


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

    def test_tracked_subjects_share_one_classifier_and_the_decisions_rule(self):
        d = FallDecision().replace(frame_stride=1, reconnect_sec=3.0,
                                   reconnect_dist=0.2)

        streams = d.build_camera()

        self.assertEqual(type(streams).__name__, "SubjectStreams")
        self.assertEqual(streams.reconnect, 3 * 15)
        self.assertEqual(streams.reconnect_dist, 0.2)
        self.assertEqual(streams.keep, 60)
        first, second = streams.template.fresh(), streams.template.fresh()
        self.assertIs(first.classifier, streams.classifier)
        self.assertIs(second.classifier, streams.classifier)
        self.assertEqual(first.min_descent, 0.10)
        self.assertEqual(first.decision.buffer, 4)

    def test_the_largest_rule_builds_one_stream(self):
        stream = FallDecision().replace(subjects="largest").build_camera()

        self.assertEqual(type(stream).__name__, "EnsembleStream")

    def test_a_head_fitted_on_another_pose_backbone_is_refused(self):
        import tempfile
        from pathlib import Path

        from fallcore.calibrate import ProbeHead

        head = ProbeHead.load(_DEPLOYED.heads[0])
        head.pose = "yolo26n"
        other = head.save(Path(tempfile.mkdtemp()) / "n_head.npz")

        with self.assertRaisesRegex(ValueError, "yolo26n keypoints"):
            FallDecision().replace(heads=(other,)).build()

    def test_picam_scores_the_yolo26n_pair_its_caches_hold(self):
        from fallcore import picam

        if not (cfg.CKPT_DIR / "probe_augnone_ms_coords_hn_s99.npz").exists():
            self.skipTest("the retired yolo26n head is not on disk")
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

        if not (cfg.CKPT_DIR / "probe_augnone_ms_coords_hn_s99.npz").exists():
            self.skipTest("the retired yolo26n head is not on disk")
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
