"""The pose backbones the detector can run, each paired with its camera head.

A camera head reads the keypoints of the pose backbone it was fitted on: the
yolo26n head on yolo26m keypoints loses two picam-3 falls that a head fitted on
yolo26m keypoints catches. So a backbone and its head are chosen together, by
`POSE_SCALE`, and the backbone the detector ends up running decides the head.

yolo26m is deployed, as an INT8 ONNX export at 384x640 that ONNX Runtime runs
on any x86 or ARM CPU. On the two picam sessions neither head was fitted on it
catches the same 50 of 52 falls as yolo26n with 4 false alarms against 7, and
its whole loop runs 24.8 fps on 4 threads of a Ryzen 7 9800X3D (yolo26n ~43 on
2). yolo26n is the backup: fetched alongside it, picked by `POSE_SCALE=yolo26n`,
and run automatically when the yolo26m files are missing or its model will not
load (`open_first`).

Light on purpose: importing it pulls in only `fallcore.config`, so a test or a
CLI can read the choice without torch or ONNX Runtime.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from . import config as cfg
from .fall_decision import DEPLOYED_HEAD


@dataclass(frozen=True)
class PoseBackbone:
    scale: str
    model: Path
    #: (height, width) the export's input is fixed at. A static ONNX graph
    #: runs at that size only, and `track()` refuses any other on a graph that
    #: is not square.
    imgsz: tuple[int, int]
    #: The camera head fitted on this backbone's keypoints; None runs the
    #: checkpoint's own classifier (`CALIBRATION_HEAD=` empty).
    head: Path | None


BACKBONES: dict[str, PoseBackbone] = {
    # 1280x720 frames letterbox to 384x640, so a square 640 input is 44%
    # padding; the rect export is 1.6x faster and reproduces the .pt keypoints.
    # Static INT8 with 7-bit weights (reduce_range), so CPUs without VNNI
    # cannot saturate it. Its head is fitted with segment labels
    # (scripts/fit_calibration_head.py --labels segment).
    "yolo26m": PoseBackbone(
        "yolo26m", cfg.ROOT / "models" / "yolo26m-pose-384x640-int8.onnx",
        (384, 640), cfg.CKPT_DIR / "probe_augnone_ms_coords_hn_s99_yolo26m-int8.npz"),
    "yolo26n": PoseBackbone(
        "yolo26n", cfg.ROOT / "models" / "yolo26n-pose.onnx",
        (640, 640), cfg.CKPT_DIR / DEPLOYED_HEAD),
}
DEPLOYED = "yolo26m"
BACKUP = "yolo26n"


def parse_imgsz(text: str) -> tuple[int, int]:
    """`"640"` -> (640, 640); `"384,640"` -> (384, 640), height first."""
    parts = [p for p in text.replace("x", ",").split(",") if p.strip()]
    try:
        sizes = [int(p) for p in parts]
    except ValueError:
        sizes = []
    if len(sizes) not in (1, 2) or min(sizes) < 32:
        raise ValueError(f"POSE_IMGSZ must be a size like 640 or 384,640, "
                         f"not {text!r}")
    return (sizes[0], sizes[-1])


def from_env(env: Mapping[str, str]) -> list[PoseBackbone]:
    """The backbones to try, in order: `POSE_SCALE`'s, then the backup.

    `POSE_MODEL` replaces the chosen backbone's file, `POSE_IMGSZ` its input
    size and `CALIBRATION_HEAD` its head (empty for none). The backup is only
    offered while nothing is pinned: with `POSE_MODEL` or `CALIBRATION_HEAD`
    set, the operator chose the files, and a missing one is an error rather
    than a reason to run something else.
    """
    scale = (env.get("POSE_SCALE") or DEPLOYED).strip()
    if scale not in BACKBONES:
        raise ValueError(f"POSE_SCALE must be one of {', '.join(BACKBONES)}, "
                         f"not {scale!r}: a head is fitted per backbone, so "
                         f"another one needs POSE_MODEL and CALIBRATION_HEAD")
    chosen = BACKBONES[scale]
    if env.get("POSE_MODEL"):
        chosen = replace(chosen, model=Path(env["POSE_MODEL"]))
    if env.get("POSE_IMGSZ"):
        chosen = replace(chosen, imgsz=parse_imgsz(env["POSE_IMGSZ"]))
    if "CALIBRATION_HEAD" in env:
        head = env["CALIBRATION_HEAD"].strip()
        chosen = replace(chosen, head=Path(head) if head else None)
    pinned = bool(env.get("POSE_MODEL")) or "CALIBRATION_HEAD" in env
    if pinned or scale == BACKUP:
        return [chosen]
    return [chosen, BACKBONES[BACKUP]]


def open_first(choices: list[PoseBackbone], load: Callable[[PoseBackbone], object]):
    """`(backbone, loaded, skipped)` for the first choice `load` accepts.

    `load` returns the opened model or raises. A choice whose model or head is
    missing is skipped without being loaded. `skipped` says why each earlier
    choice was passed over; when every choice fails, the first one's error is
    raised, since that is the backbone that was asked for.
    """
    skipped, first_error = [], None
    for choice in choices:
        missing = [str(p) for p in (choice.model, choice.head)
                   if p is not None and not p.exists()]
        try:
            if missing:
                raise FileNotFoundError("missing " + ", ".join(missing))
            return choice, load(choice), skipped
        except Exception as exc:  # noqa: BLE001 - any failure moves to the backup
            first_error = first_error or exc
            skipped.append(f"{choice.scale}: {type(exc).__name__}: {exc}")
    raise first_error
