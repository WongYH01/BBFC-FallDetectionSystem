"""fallcore.pose_backbone: the one pose backbone the detector runs.

Reads constants only, so no model file, ONNX Runtime or camera is needed.
"""
from __future__ import annotations

import unittest

from fallcore import pose_backbone as pb
from fallcore.fall_decision import FallDecision


class DeployedTest(unittest.TestCase):
    def test_it_is_the_yolo26m_int8_rect_export(self):
        self.assertEqual(pb.DEPLOYED.model.name, "yolo26m-pose-384x640-int8.onnx")
        self.assertEqual(pb.DEPLOYED.imgsz, (384, 640))

    def test_it_is_the_backbone_the_deployed_head_reads(self):
        self.assertEqual(pb.DEPLOYED.name, FallDecision().pose)


if __name__ == "__main__":
    unittest.main()
