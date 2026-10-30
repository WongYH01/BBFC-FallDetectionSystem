"""The pose backbone the detector runs: the one its camera head was fitted on.

A camera head reads the keypoints of the pose backbone it was fitted on: the
yolo26n head on yolo26m keypoints loses two picam-3 falls that a head fitted on
yolo26m keypoints catches. So there is one backbone, and the Fall Decision's
head is the one fitted on its keypoints (`fallcore.fall_decision.DEPLOYED_POSE`).
Nothing in the environment swaps either half on its own.

yolo26m runs as an INT8 ONNX export at 384x640 that ONNX Runtime runs on any
x86 or ARM CPU. On the two picam sessions neither head was fitted on it catches
the same 50 of 52 falls as yolo26n with 4 false alarms against 7, and its whole
loop runs 24.8 fps on 4 threads of a Ryzen 7 9800X3D. When its file is missing
or will not load the detector stops: there is no backup to fall back to.

Light on purpose: importing it pulls in only `fallcore.config`, so a test or a
CLI can read the choice without torch or ONNX Runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config as cfg
from .fall_decision import DEPLOYED_POSE


@dataclass(frozen=True)
class PoseBackbone:
    #: The keypoint cache its keypoints are stored as, which is what a head
    #: records it was fitted on (`fallcore.calibrate.ProbeHead.pose`).
    name: str
    model: Path
    #: (height, width) the export's input is fixed at. A static ONNX graph
    #: runs at that size only, and `track()` refuses any other on a graph that
    #: is not square.
    imgsz: tuple[int, int]


#: 1280x720 frames letterbox to 384x640, so a square 640 input is 44% padding;
#: the rect export is 1.6x faster and reproduces the .pt keypoints. Static INT8
#: with 7-bit weights (reduce_range), so CPUs without VNNI cannot saturate it.
DEPLOYED = PoseBackbone(DEPLOYED_POSE,
                        cfg.ROOT / "models" / "yolo26m-pose-384x640-int8.onnx",
                        (384, 640))
