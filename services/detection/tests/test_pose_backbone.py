"""fallcore.pose_backbone: which pose backbone and head the detector runs.

Two seams: `from_env` takes a plain dict in place of the process environment,
and `open_first` takes the loader, so no model file, ONNX Runtime or camera is
needed.
"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fallcore import pose_backbone as pb


class FromEnvTest(unittest.TestCase):
    def test_an_empty_environment_runs_m_with_n_as_the_backup(self):
        choices = pb.from_env({})

        self.assertEqual([c.scale for c in choices], ["yolo26m", "yolo26n"])
        self.assertEqual(choices[0].imgsz, (384, 640))
        self.assertEqual(choices[0].head.name,
                         "probe_augnone_ms_coords_hn_s99_yolo26m-int8.npz")
        self.assertEqual(choices[1].head.name, "probe_augnone_ms_coords_hn_s99.npz")

    def test_pose_scale_n_runs_n_alone(self):
        self.assertEqual([c.scale for c in pb.from_env({"POSE_SCALE": "yolo26n"})],
                         ["yolo26n"])

    def test_an_unpaired_scale_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            pb.from_env({"POSE_SCALE": "yolo26s"})
        self.assertIn("POSE_SCALE", str(caught.exception))

    def test_a_pinned_model_replaces_the_file_and_drops_the_backup(self):
        choices = pb.from_env({"POSE_MODEL": "elsewhere/m.onnx"})

        self.assertEqual(len(choices), 1)
        self.assertEqual(choices[0].model, Path("elsewhere/m.onnx"))
        self.assertEqual(choices[0].head, pb.BACKBONES["yolo26m"].head)

    def test_a_pinned_head_replaces_the_head_and_drops_the_backup(self):
        named = pb.from_env({"CALIBRATION_HEAD": " other/probe.npz "})
        empty = pb.from_env({"CALIBRATION_HEAD": ""})

        self.assertEqual([c.head for c in named], [Path("other/probe.npz")])
        self.assertEqual([c.head for c in empty], [None])

    def test_imgsz_takes_one_size_or_height_and_width(self):
        self.assertEqual(pb.from_env({"POSE_IMGSZ": "480"})[0].imgsz, (480, 480))
        self.assertEqual(pb.parse_imgsz("384,640"), (384, 640))
        self.assertEqual(pb.parse_imgsz("384x640"), (384, 640))
        for bad in ("", "big", "1,2,3", "0"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pb.parse_imgsz(bad)


class OpenFirstTest(unittest.TestCase):
    def setUp(self):
        tmp = Path(tempfile.mkdtemp())
        for name in ("m.onnx", "m.npz", "n.onnx", "n.npz"):
            (tmp / name).write_bytes(b"")
        self.m = replace(pb.BACKBONES["yolo26m"], model=tmp / "m.onnx",
                         head=tmp / "m.npz")
        self.n = replace(pb.BACKBONES["yolo26n"], model=tmp / "n.onnx",
                         head=tmp / "n.npz")

    def test_the_first_choice_that_loads_runs(self):
        chosen, loaded, skipped = pb.open_first([self.m, self.n], lambda c: c.scale)

        self.assertEqual((chosen, loaded, skipped), (self.m, "yolo26m", []))

    def test_a_missing_file_falls_back_without_loading_it(self):
        m = replace(self.m, head=self.m.head.with_name("gone.npz"))
        loads = []

        chosen, _, skipped = pb.open_first(
            [m, self.n], lambda c: loads.append(c.scale) or c.scale)

        self.assertIs(chosen, self.n)
        self.assertEqual(loads, ["yolo26n"])
        self.assertIn("gone.npz", skipped[0])

    def test_a_model_that_will_not_load_falls_back(self):
        def load(c):
            if c is self.m:
                raise RuntimeError("bad graph")
            return c.scale

        chosen, _, skipped = pb.open_first([self.m, self.n], load)

        self.assertIs(chosen, self.n)
        self.assertEqual(skipped, ["yolo26m: RuntimeError: bad graph"])

    def test_when_nothing_loads_the_asked_for_backbone_s_error_is_raised(self):
        def load(c):
            raise RuntimeError(c.scale)

        with self.assertRaisesRegex(RuntimeError, "yolo26m"):
            pb.open_first([self.m, self.n], load)

    def test_no_head_is_not_a_missing_file(self):
        bare = replace(self.m, head=None)

        self.assertIs(pb.open_first([bare], lambda c: c)[0], bare)


if __name__ == "__main__":
    unittest.main()
