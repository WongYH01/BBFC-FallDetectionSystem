"""fallcore -- YOLO26-Pose + Transformer fall detection.

A replication of Benabdennour et al., "Real-Time Human Fall Detection From
Video Using YOLOv11 With Pose Estimation" (IEEE Access, vol. 14, 2026) with the
pose backbone swapped from YOLOv11 to YOLO26, plus a size sweep over the
backbone that the paper only samples at two points.

The notebooks in notebooks/ are the interface; this package holds the logic so
the eight of them do not each carry their own copy of a training loop.
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import (bench, calibrate, caucafall, config, data, decide, evaluate, extract,
               gmdcsa, infer, interpret, le2i, manifest, model, omnifall, picam,
               rulebased, stream, train, viz)

__all__ = [
    "bench", "calibrate", "caucafall", "config", "data", "decide", "evaluate",
    "extract", "gmdcsa", "infer", "interpret", "le2i", "manifest", "model",
    "omnifall", "picam", "rulebased", "stream", "train", "viz",
]
