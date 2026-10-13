"""Latency, throughput and VRAM, measured the way Section IV-D describes.

The protocol matters more than it looks: 3 trials of 200 runs, 50 warmup
iterations discarded, outliers removed at 1.5x IQR, and torch.cuda.synchronize()
around every measurement. Without the synchronise you time the queueing of CUDA
kernels rather than their execution, which on a modern GPU reads as an
implausibly fast model.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import config as cfg


def _iqr_filter(values: np.ndarray, k: float = cfg.BENCH.iqr_multiplier) -> np.ndarray:
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return values[(values >= q1 - k * iqr) & (values <= q3 + k * iqr)]


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_module(
    model,
    sample: torch.Tensor,
    device: str = "cuda",
    bench: cfg.BenchConfig = cfg.BENCH,
) -> dict:
    """Per-call latency of any nn.Module on a fixed input.

    Returns milliseconds. `sample` should already carry the batch dimension you
    want measured -- the paper quotes the transformer at batch size 1.
    """
    device = device if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    sample = sample.to(device)

    for _ in range(bench.warmup):
        model(sample)
    _sync(device)

    timings: list[float] = []
    for _ in range(bench.trials):
        for _ in range(bench.runs_per_trial):
            _sync(device)
            t0 = time.perf_counter()
            model(sample)
            _sync(device)
            timings.append((time.perf_counter() - t0) * 1000.0)

    raw = np.asarray(timings)
    clean = _iqr_filter(raw)
    return {
        "device": device,
        "n_raw": int(raw.size),
        "n_after_iqr": int(clean.size),
        "mean_ms": float(clean.mean()),
        "std_ms": float(clean.std()),
        "median_ms": float(np.median(clean)),
        "p95_ms": float(np.percentile(clean, 95)),
        "throughput_per_sec": float(1000.0 / clean.mean()) if clean.mean() > 0 else None,
    }


@torch.no_grad()
def benchmark_pose(
    scale: str,
    device: str = "cuda",
    imgsz: int = cfg.PREPROCESS.imgsz,
    bench: cfg.BenchConfig = cfg.BENCH,
    runs: int | None = None,
) -> dict:
    """Per-frame latency of a pose backbone on a synthetic frame.

    A blank frame is enough: YOLO26 is dense and end-to-end, so its cost does
    not depend on how many people are in the image the way an NMS-bound
    detector's would. Uses fewer runs than the classifier benchmark by default
    because each one is three orders of magnitude more expensive.
    """
    from .extract import load_pose_model

    device = device if torch.cuda.is_available() else "cpu"
    model = load_pose_model(scale, device=device)
    frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    n_runs = runs if runs is not None else 100

    for _ in range(min(bench.warmup, 20)):
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)
    _sync(device)

    timings = []
    for _ in range(n_runs):
        _sync(device)
        t0 = time.perf_counter()
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)
        _sync(device)
        timings.append((time.perf_counter() - t0) * 1000.0)

    clean = _iqr_filter(np.asarray(timings))
    return {
        "scale": scale,
        "params": cfg.POSE_BACKBONES[scale]["params"],
        "gflops": cfg.POSE_BACKBONES[scale]["gflops"],
        "mean_ms_per_frame": float(clean.mean()),
        "std_ms": float(clean.std()),
        "fps": float(1000.0 / clean.mean()) if clean.mean() > 0 else None,
        "n_after_iqr": int(clean.size),
        "device": device,
    }


@torch.no_grad()
def peak_vram_mb(model, sample: torch.Tensor, device: str = "cuda") -> float | None:
    """Peak allocated VRAM for one forward pass, in MB. None on CPU."""
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return None
    model = model.to(device).eval()
    sample = sample.to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model(sample)
    _sync(device)
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def end_to_end_profile(
    pose_bench: dict,
    clf_bench: dict,
    frames_per_clip: int = cfg.PREPROCESS.max_frames,
) -> dict:
    """Combine the two stages into the numbers that describe deployment.

    The pipeline is sequential, so throughput is bounded by the pose stage: the
    classifier runs once per clip while the backbone runs once per frame. In the
    paper that is 13.12 ms/frame against 3.54 ms/clip, which is why they call the
    classifier's contribution negligible.
    """
    pose_ms = pose_bench["mean_ms_per_frame"]
    clf_ms = clf_bench["mean_ms"]
    clip_ms = pose_ms * frames_per_clip + clf_ms
    return {
        "scale": pose_bench["scale"],
        "pose_ms_per_frame": pose_ms,
        "pose_fps": pose_bench["fps"],
        "classifier_ms_per_clip": clf_ms,
        "clip_ms_total": clip_ms,
        "classifier_share": clf_ms / clip_ms if clip_ms else None,
        "realtime_margin_at_30fps": (1000.0 / 30.0) / pose_ms if pose_ms else None,
    }
