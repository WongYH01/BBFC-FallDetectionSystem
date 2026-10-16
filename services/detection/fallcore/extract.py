"""Frozen YOLO26-Pose over video -> cached (F, 17, 3) keypoint arrays.

This is the expensive half of the pipeline and the only place the pose backbone
is touched. The backbone is never trained: the paper uses COCO-pretrained
weights as a fixed feature extractor, and the size sweep asks how much keypoint
quality (not detector fitness) improves with capacity.

Storage convention follows the paper, Section III-A-3: per frame, per keypoint,
the triple is (y_norm, x_norm, confidence) -- y BEFORE x, both divided by the
frame height/width so the representation is resolution-invariant. Getting that
order wrong transposes every pose and still trains to a plausible-looking
number, so notebook 01 draws a cached skeleton back onto its source frames.

Full-length sequences are cached rather than the paper's pre-truncated 60
frames. One extraction pass then serves every seq_len the ablations ask for,
and cropping stays a decision made at training time.
"""
from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import config as cfg


def resolve_weights(scale: str) -> str:
    """Path or name to hand Ultralytics for a given scale.

    Prefers a checkpoint already on disk (models/ may hold yolo26n-pose.pt);
    otherwise returns the bare asset name and lets Ultralytics download it.
    """
    if scale not in cfg.POSE_BACKBONES:
        raise KeyError(f"unknown scale {scale!r}; known: {list(cfg.POSE_BACKBONES)}")
    name = cfg.POSE_BACKBONES[scale]["weights"]
    local = cfg.LOCAL_WEIGHTS_DIR / name
    return str(local) if local.exists() else name


def load_pose_model(scale: str, device: str = "cuda"):
    """Load a frozen YOLO26 pose model in eval mode.

    YOLO26 pose is end-to-end (end2end: True, reg_max: 1 in its yaml), so there
    is no NMS threshold to tune here -- unlike the YOLOv11 backbone the paper
    used. One less knob to hold constant across the sweep.
    """
    from ultralytics import YOLO

    model = YOLO(resolve_weights(scale))
    model.to(device)
    for p in model.model.parameters():
        p.requires_grad_(False)
    model.model.eval()
    return model


def keypoints_path(scale: str, video_id: str) -> Path:
    return cfg.KEYPOINTS_DIR / scale / f"{video_id}.npy"


def _largest_person(result) -> int | None:
    """Index of the detection with the biggest box, or None if the frame is empty.

    Section III-A-3: "the individual with the largest bounding box area is
    selected as the primary subject". A blunt rule, and one reason the paper
    lists crowded scenes as a limitation.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return int(np.argmax(areas))


def _cache_row(k: np.ndarray, h: int, w: int) -> np.ndarray:
    """Ultralytics' `(17, 3)` `(x, y, conf)` in pixels -> the cache's row.

    The paper stores `(y_norm, x_norm, conf)`: swapped, and divided by the
    frame height and width.
    """
    kp = np.zeros((cfg.N_KEYPOINTS, 3), dtype=np.float32)
    kp[:, 0] = np.clip(k[:, 1] / max(h, 1), 0.0, 1.0)   # y first
    kp[:, 1] = np.clip(k[:, 0] / max(w, 1), 0.0, 1.0)   # then x
    kp[:, 2] = k[:, 2]
    return kp


def subject_row(result) -> tuple[np.ndarray, float]:
    """One pose result -> the subject's `(17, 3)` row and the frame's width/height.

    The cache convention, defined here once for the cache builder and every
    live or offline caller: the largest box is the subject, and its keypoints
    are stored `(y_norm, x_norm, conf)` -- swapped and divided by the frame
    height and width. A frame with no detection is a row of zeros, exactly as
    the classifier was trained on it. The aspect ratio comes back alongside
    because the window kinematics need x and y in the same units, and a caller
    that feeds a stream rows without it scores a squashed body.
    """
    h, w = result.orig_shape
    kp = np.zeros((cfg.N_KEYPOINTS, 3), dtype=np.float32)
    idx = _largest_person(result)
    keypoints = getattr(result, "keypoints", None)
    if idx is not None and keypoints is not None and keypoints.data is not None \
            and len(keypoints.data) > idx:
        kp = _cache_row(keypoints.data[idx].cpu().numpy(), h, w)
    return kp, w / max(h, 1)


@dataclass(frozen=True)
class Sighting:
    """One tracked person in one frame.

    `row` is the cache convention (`subject_row`'s); `centre` is the box
    centre as `(x, y)` fractions of the frame width and height, and `area` the
    box's share of the frame.
    """
    track_id: int
    row: np.ndarray
    centre: tuple[float, float]
    area: float


def sightings(result) -> tuple[list[Sighting], float]:
    """One tracked pose result -> every tracked person and the frame's width/height.

    The per-subject counterpart of `subject_row`, for a result from
    `model.track`: every box with a tracker id becomes a `Sighting`. A box the
    tracker has not confirmed carries no id and is left out, since it cannot
    be followed from one frame to the next.
    """
    h, w = result.orig_shape
    out: list[Sighting] = []
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or len(boxes) == 0 or getattr(boxes, "id", None) is None \
            or keypoints is None or keypoints.data is None:
        return out, w / max(h, 1)
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.int().cpu().tolist()
    data = keypoints.data.cpu().numpy()
    for i, tid in enumerate(ids):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        out.append(Sighting(
            track_id=int(tid), row=_cache_row(data[i], h, w),
            centre=((x1 + x2) / 2 / max(w, 1), (y1 + y2) / 2 / max(h, 1)),
            area=(x2 - x1) * (y2 - y1) / max(w * h, 1)))
    return out, w / max(h, 1)


def pose_device(model, device: str = "cuda") -> str:
    """The device to run an Ultralytics model on, given what that model is.

    An ONNX file is executed by ONNX Runtime, and Ultralytics re-binds the input
    tensor onto whatever `device` says -- so `cuda` for a CPU-only ONNX session
    fails with "no data transfer registered for copying tensors from
    Device:[DeviceType:1] to Device:[DeviceType:0]".

    Always CPU for ONNX here, deliberately. Whether ONNX Runtime even advertises
    `CUDAExecutionProvider` depends on import order in the same process (it is
    absent if `onnxruntime` is imported before torch has loaded its CUDA DLLs),
    so "is CUDA available" is not a safe question to ask at this point -- and
    the exports in `runs/onnx/` are CPU graphs running on a CPU deployment. A
    GPU deployment should pass the device explicitly.
    """
    names = [getattr(model, a, None) for a in ("ckpt", "ckpt_path", "model_name", "pt_path")]
    if any(str(n).lower().endswith(".onnx") for n in names if n):
        return "cpu"
    return device


def extract_video(
    model,
    video_path: "str | Path | Sequence[str | Path]",
    imgsz: int = cfg.PREPROCESS.imgsz,
    device: str = "cuda",
    max_frames: int | None = None,
) -> np.ndarray:
    """Run pose over one video, returning (F, 17, 3) float32 as (y, x, conf).

    Frames with no detection become all-zero rows. That is deliberate: the
    confidence channel carries the zero forward, so the classifier can learn
    that a missing subject is itself a signal, and a video with too many of them
    is caught by the mean-confidence filter.

    `video_path` may also be an explicit ordered sequence of image paths, for
    datasets distributed as frame folders rather than as video files. The order
    given is the order returned, which matters when the frame labels live in
    sibling files keyed by filename -- see fallcore.caucafall, where the .avi
    copies disagree with the frame count by as much as 23 frames and the PNGs
    are the authoritative source.
    """
    frames: list[np.ndarray] = []
    source = ([str(p) for p in video_path]
              if isinstance(video_path, (list, tuple))
              else str(video_path))
    stream = model.predict(
        source=source,
        stream=True,
        imgsz=imgsz,
        device=pose_device(model, device),
        verbose=False,
    )

    for result in stream:
        kp, _aspect = subject_row(result)
        frames.append(kp)
        if max_frames is not None and len(frames) >= max_frames:
            break

    if not frames:
        return np.zeros((0, cfg.N_KEYPOINTS, 3), dtype=np.float32)
    return np.stack(frames).astype(np.float32)


def extract_dataset(
    manifest,
    scale: str,
    device: str = "cuda",
    imgsz: int = cfg.PREPROCESS.imgsz,
    limit: int | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> dict:
    """Extract keypoints for every video in `manifest` at one backbone scale.

    Idempotent: videos whose .npy already exists are skipped unless
    `overwrite`, so an interrupted run costs nothing to resume. Returns a
    summary including a CUDA-synchronised frames-per-second figure covering
    decode plus inference -- the number that actually bounds the pipeline.
    """
    from tqdm.auto import tqdm

    out_dir = cfg.KEYPOINTS_DIR / scale
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = manifest.to_dict("records")
    if limit is not None:
        rows = rows[:limit]

    model = load_pose_model(scale, device=device)
    stats: dict[str, dict] = {}
    n_frames_done = 0
    n_extracted = 0
    t_start = time.perf_counter()

    iterator = tqdm(rows, desc=f"extract {scale}", unit="vid") if progress else rows
    for row in iterator:
        vid, path = row["video_id"], row["path"]
        dest = out_dir / f"{vid}.npy"

        if dest.exists() and not overwrite:
            arr = np.load(dest)
        else:
            try:
                arr = extract_video(model, path, imgsz=imgsz, device=device)
            except Exception as exc:  # a corrupt file should not kill the pass
                stats[vid] = {"error": repr(exc), "n_frames": 0, "mean_conf": 0.0}
                continue
            np.save(dest, arr)
            n_frames_done += len(arr)
            n_extracted += 1

        # Mean over every keypoint of every frame -- the quantity the paper
        # thresholds at 0.2 to drop a video.
        mean_conf = float(arr[:, :, 2].mean()) if arr.size else 0.0
        stats[vid] = {"n_frames": int(len(arr)), "mean_conf": mean_conf}

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

    return {
        "scale": scale,
        "n_videos": len(rows),
        "n_extracted": n_extracted,
        "n_cached": len(rows) - n_extracted,
        "frames_extracted": n_frames_done,
        "elapsed_sec": elapsed,
        # Only meaningful when something was actually extracted; a fully-cached
        # re-run measures disk reads, not the model.
        "fps": (n_frames_done / elapsed) if n_frames_done and elapsed > 0 else None,
        "per_video": stats,
    }


def load_keypoints(scale: str, video_id: str) -> np.ndarray:
    return np.load(keypoints_path(scale, video_id))


def confidence_table(scale: str, video_ids: list[str]) -> dict[str, float]:
    """Mean keypoint confidence per video, for the quality filter."""
    out = {}
    for vid in video_ids:
        p = keypoints_path(scale, vid)
        if not p.exists():
            continue
        arr = np.load(p)
        out[vid] = float(arr[:, :, 2].mean()) if arr.size else 0.0
    return out


def surviving_ids(
    scales: list[str],
    video_ids: list[str],
    min_conf: float = cfg.PREPROCESS.min_mean_confidence,
) -> tuple[list[str], dict]:
    """Videos that clear the confidence filter under *every* scale.

    The filter is backbone-dependent -- a bigger pose model rescues clips a
    smaller one gives up on -- so applying it per scale would hand each arm of
    the sweep a different dataset, and the resulting F1 gap would partly measure
    which videos got dropped. Taking the intersection costs a few extra clips
    and buys an apples-to-apples comparison.
    """
    per_scale = {s: confidence_table(s, video_ids) for s in scales}

    keep, detail = [], {}
    for vid in video_ids:
        confs = {s: per_scale[s].get(vid) for s in scales}
        if any(c is None for c in confs.values()):
            detail[vid] = {"kept": False, "reason": "missing cache", "conf": confs}
            continue
        if all(c >= min_conf for c in confs.values()):
            keep.append(vid)
        else:
            detail[vid] = {"kept": False, "reason": "low confidence", "conf": confs}

    summary = {
        "min_conf": min_conf,
        "n_input": len(video_ids),
        "n_kept": len(keep),
        "n_dropped": len(video_ids) - len(keep),
        "dropped_per_scale": {
            s: sum(1 for v in video_ids
                   if per_scale[s].get(v) is not None and per_scale[s][v] < min_conf)
            for s in scales
        },
        "mean_conf_per_scale": {
            s: float(np.mean([c for c in per_scale[s].values()])) if per_scale[s] else 0.0
            for s in scales
        },
    }
    return keep, {"summary": summary, "dropped": detail}


def save_extraction_report(report: dict, scale: str) -> Path:
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.METRICS_DIR / f"extraction_{scale}.json"
    slim = {k: v for k, v in report.items() if k != "per_video"}
    slim["per_video_count"] = len(report.get("per_video", {}))
    path.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Person bounding boxes (notebook 11 -- the rule-based baseline)
# ---------------------------------------------------------------------------
def boxes_path(scale: str, video_id: str) -> Path:
    return cfg.BOXES_DIR / scale / f"{video_id}.npy"


def extract_boxes_video(
    model,
    video_path: "str | Path | Sequence[str | Path]",
    imgsz: int = cfg.PREPROCESS.imgsz,
    device: str = "cuda",
) -> np.ndarray:
    """Largest person's box per frame as (F, 5): x1, y1, x2, y2, conf.

    A frame with no detection is an all-zero row, matching the convention in
    `extract_video`. The same `_largest_person` rule picks the subject, so the
    box cached here belongs to the same person whose keypoints were cached --
    without that, the rule-based baseline and the transformer would be scoring
    different people in a crowded frame.

    demo-cam's detector runs Ultralytics *tracking* and scores every box in the
    frame. These datasets are single-subject, so largest-box-per-frame is the
    faithful analogue and keeps the comparison against the keypoint pipeline
    honest. What it does not reproduce is the tracker's identity handling, which
    only matters with more than one person in shot.
    """
    rows: list[np.ndarray] = []
    source = ([str(p) for p in video_path]
              if isinstance(video_path, (list, tuple))
              else str(video_path))
    stream = model.predict(source=source, stream=True, imgsz=imgsz,
                           device=device, verbose=False)

    for result in stream:
        row = np.zeros(5, dtype=np.float32)
        idx = _largest_person(result)
        boxes = getattr(result, "boxes", None)
        if idx is not None and boxes is not None and len(boxes) > idx:
            row[:4] = boxes.xyxy[idx].cpu().numpy()
            conf = getattr(boxes, "conf", None)
            row[4] = float(conf[idx].cpu()) if conf is not None else 1.0
        rows.append(row)

    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


def extract_boxes_dataset(
    manifest,
    scale: str,
    device: str = "cuda",
    imgsz: int = cfg.PREPROCESS.imgsz,
    overwrite: bool = False,
    progress: bool = True,
) -> dict:
    """Cache person boxes for every video in `manifest`. Idempotent."""
    from tqdm.auto import tqdm

    out_dir = cfg.BOXES_DIR / scale
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = manifest.to_dict("records")
    model = load_pose_model(scale, device=device)
    n_extracted = 0
    t0 = time.perf_counter()

    it = tqdm(rows, desc=f"boxes {scale}", unit="vid") if progress else rows
    for row in it:
        dest = out_dir / f"{row['video_id']}.npy"
        if dest.exists() and not overwrite:
            continue
        try:
            arr = extract_boxes_video(model, row["path"], imgsz=imgsz, device=device)
        except Exception:
            continue
        np.save(dest, arr)
        n_extracted += 1

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"scale": scale, "n_videos": len(rows), "n_extracted": n_extracted,
            "n_cached": len(rows) - n_extracted,
            "elapsed_sec": time.perf_counter() - t0}


def load_boxes(scale: str, video_id: str) -> np.ndarray:
    return np.load(boxes_path(scale, video_id))
