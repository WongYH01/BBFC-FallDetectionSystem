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
from pathlib import Path

import numpy as np
import torch

from . import config as cfg


def resolve_weights(scale: str) -> str:
    """Path or name to hand Ultralytics for a given scale.

    Prefers a checkpoint already on disk (demo-cam/models has yolo26n-pose.pt);
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
        device=device,
        verbose=False,
    )

    for result in stream:
        h, w = result.orig_shape
        kp = np.zeros((cfg.N_KEYPOINTS, 3), dtype=np.float32)

        idx = _largest_person(result)
        keypoints = getattr(result, "keypoints", None)
        if idx is not None and keypoints is not None and keypoints.data is not None \
                and len(keypoints.data) > idx:
            # Ultralytics gives (x, y, conf) in pixels; the paper stores
            # (y_norm, x_norm, conf). Swap and normalise in one step.
            k = keypoints.data[idx].cpu().numpy()  # (17, 3)
            kp[:, 0] = np.clip(k[:, 1] / max(h, 1), 0.0, 1.0)   # y first
            kp[:, 1] = np.clip(k[:, 0] / max(w, 1), 0.0, 1.0)   # then x
            kp[:, 2] = k[:, 2]

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
