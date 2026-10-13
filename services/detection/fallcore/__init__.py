"""fallcore -- YOLO26-Pose + Transformer fall detection.

A replication of Benabdennour et al., "Real-Time Human Fall Detection From
Video Using YOLOv11 With Pose Estimation" (IEEE Access, vol. 14, 2026) with the
pose backbone swapped from YOLOv11 to YOLO26, plus a size sweep over the
backbone that the paper only samples at two points.

The notebooks in notebooks/ are the interface; this package holds the logic so
the eight of them do not each carry their own copy of a training loop.

Submodules load lazily (PEP 562). Eagerly importing them pulled in pandas,
scikit-learn and the rest of the research stack on every `from fallcore import
config`, which the deployment services do not have installed and do not need;
`from fallcore import X` and `import fallcore.X` behave exactly as before, the
import just happens on first use.
"""
from __future__ import annotations

import importlib

__version__ = "0.1.0"

__all__ = [
    "bench", "calibrate", "caucafall", "config", "data", "decide", "evaluate",
    "extract", "gmdcsa", "infer", "interpret", "le2i", "manifest", "model",
    "omnifall", "picam", "rulebased", "stream", "train", "viz",
]


def __getattr__(name: str):
    if name in __all__:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
