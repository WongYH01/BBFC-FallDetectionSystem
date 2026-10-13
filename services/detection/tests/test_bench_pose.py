"""scripts/bench_pose.py: the stride rule and the verdicts, against a fake timer.

The timer is the one seam. These tests hand `measure_and_recommend` fixed
ms-per-frame values, so no model file, ONNX Runtime or GPU is needed.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bench_pose import measure_and_recommend


def fixed_timer(ms_by_name: dict[str, float], device: str = "cpu"):
    """A timer that reports `ms_by_name[<file name>]` and records its calls."""
    calls = []

    def time_model(path, imgsz, threads):
        calls.append((Path(path).name, imgsz, threads))
        ms = ms_by_name[Path(path).name]
        if isinstance(ms, Exception):
            raise ms
        return {"mean_ms": ms, "p95_ms": ms * 1.1, "device": device}

    time_model.calls = calls
    return time_model


class StrideRuleTest(unittest.TestCase):
    def test_30fps_fast_model_fits_at_stride_2_1(self):
        [row] = measure_and_recommend(
            ["yolo26n-pose.onnx"], camera_fps=30,
            time_model=fixed_timer({"yolo26n-pose.onnx": 20.0}))

        self.assertEqual(row["vid_stride"], 2)
        self.assertEqual(row["stream_frame_stride"], 1)
        self.assertAlmostEqual(row["spacing_ms"], 66.7, places=1)
        self.assertAlmostEqual(row["fps"], 50.0)
        self.assertAlmostEqual(row["headroom"], 50.0 / 15.0)
        self.assertEqual(row["verdict"], "fits")
        self.assertIsNone(row["warning"])

    def test_15fps_camera_poses_every_frame(self):
        [row] = measure_and_recommend(
            ["yolo26n-pose.onnx"], camera_fps=15,
            time_model=fixed_timer({"yolo26n-pose.onnx": 20.0}))

        self.assertEqual((row["vid_stride"], row["stream_frame_stride"]), (1, 1))
        self.assertAlmostEqual(row["spacing_ms"], 66.7, places=1)
        self.assertIsNone(row["warning"])

    def test_25fps_camera_warns_that_no_stride_lands_near_67ms(self):
        [row] = measure_and_recommend(
            ["yolo26n-pose.onnx"], camera_fps=25,
            time_model=fixed_timer({"yolo26n-pose.onnx": 20.0}))

        self.assertEqual((row["vid_stride"], row["stream_frame_stride"]), (2, 1))
        self.assertAlmostEqual(row["spacing_ms"], 80.0)
        self.assertIsNotNone(row["warning"])
        self.assertIn("80", row["warning"])


class VerdictTest(unittest.TestCase):
    def verdict(self, ms, camera_fps=30):
        [row] = measure_and_recommend(
            ["m.onnx"], camera_fps=camera_fps,
            time_model=fixed_timer({"m.onnx": ms}))
        return row["verdict"]

    def test_verdict_thresholds_at_30fps(self):
        self.assertEqual(self.verdict(20.0), "fits")        # 50 fps, 3.3x
        self.assertEqual(self.verdict(60.0), "marginal")    # 16.7 fps, 1.1x
        self.assertEqual(self.verdict(80.0), "too slow")    # 12.5 fps, 0.83x

    def test_fits_needs_30_percent_headroom(self):
        # 15 rows/s * 1.3 = 19.5 fps -> 51.3 ms is the edge
        self.assertEqual(self.verdict(51.0), "fits")
        self.assertEqual(self.verdict(52.0), "marginal")


class RowsTest(unittest.TestCase):
    def test_a_model_that_fails_to_load_does_not_stop_the_run(self):
        timer = fixed_timer({
            "yolo26n-pose.onnx": 20.0,
            "broken.onnx": RuntimeError("bad export"),
            "yolo26s-pose.pt": 30.0,
        })
        rows = measure_and_recommend(
            ["yolo26n-pose.onnx", "broken.onnx", "yolo26s-pose.pt"],
            time_model=timer)

        self.assertEqual([r["model"] for r in rows],
                         ["yolo26n-pose.onnx", "broken.onnx", "yolo26s-pose.pt"])
        broken = rows[1]
        self.assertEqual(broken["verdict"], "failed")
        self.assertIn("bad export", broken["error"])
        self.assertIsNone(broken["fps"])
        self.assertEqual(rows[0]["verdict"], "fits")
        self.assertEqual(rows[2]["verdict"], "fits")

    def test_each_row_carries_the_settings_it_was_timed_under(self):
        timer = fixed_timer({"yolo26n-pose.onnx": 20.0, "yolo26m-pose.pt": 40.0},
                            device="cuda")
        rows = measure_and_recommend(
            ["yolo26n-pose.onnx", "yolo26m-pose.pt"], imgsz=480, threads=4,
            time_model=timer)

        self.assertEqual([(r["backend"], r["imgsz"], r["threads"], r["device"])
                          for r in rows],
                         [("onnx", 480, 4, "cuda"), ("torch", 480, 4, "cuda")])
        self.assertEqual(timer.calls, [("yolo26n-pose.onnx", 480, 4),
                                       ("yolo26m-pose.pt", 480, 4)])
        self.assertAlmostEqual(rows[1]["p95_ms"], 44.0)
        self.assertAlmostEqual(rows[1]["mean_ms"], 40.0)


if __name__ == "__main__":
    unittest.main()
