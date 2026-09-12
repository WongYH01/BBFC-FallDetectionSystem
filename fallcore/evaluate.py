"""The two evaluation protocols from Section III-C-3, and the metrics they share.

Random clip: one window per video -- a standard, cheap benchmark.
Full video:  Algorithm 2's sliding window with early exit -- what deployment
             actually looks like, and measurably more recall-hungry because of
             it. The paper sees FN drop 7 -> 4 and FP rise 7 -> 9 when switching.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from . import config as cfg
from .data import video_windows


def binary_metrics(y_true, y_prob, threshold: float = cfg.EVAL.threshold) -> dict:
    """Accuracy, precision, recall, F1, specificity and AUC for the Fall class.

    `zero_division=0` throughout: a degenerate model that predicts one class for
    everything should score 0 on the class it never predicts, not raise.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)),
    }
    # AUC is undefined when the labels are all one class -- report None rather
    # than letting sklearn raise mid-sweep.
    out["auc"] = (float(roc_auc_score(y_true, y_prob))
                  if len(np.unique(y_true)) > 1 else None)
    return out


@torch.no_grad()
def evaluate_random_clip(
    model,
    loader,
    device: str = "cuda",
    threshold: float = cfg.EVAL.threshold,
) -> dict:
    """Score one fixed window per video. Returns metrics plus raw probabilities."""
    device = device if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    probs, targets = [], []
    for x, y in loader:
        p = torch.sigmoid(model(x.to(device))).cpu().numpy()
        probs.append(p)
        targets.append(y.numpy())

    probs = np.concatenate(probs)
    targets = np.concatenate(targets)
    return {"protocol": "random_clip",
            **binary_metrics(targets, probs, threshold),
            "probs": probs.tolist(), "targets": targets.tolist()}


@torch.no_grad()
def predict_video(
    model,
    video_id: str,
    scale: str = cfg.DEFAULT_SCALE,
    seq_len: int = cfg.FINAL_TRAIN.seq_len,
    stride: int = cfg.EVAL.sliding_stride,
    threshold: float = cfg.EVAL.threshold,
    features: str = "full",
    device: str = "cuda",
    early_exit: bool = cfg.EVAL.early_exit,
) -> dict:
    """Algorithm 2: slide, score, and call it a fall on the first window over tau.

    With early_exit the reported probability is the first window to cross the
    threshold (or the max, when none does), which is what the paper's "classify
    as Fall if any clip is Fall" rule amounts to. All windows are scored in a
    single batch regardless -- the early exit is a decision rule, not a
    throughput trick, and short-circuiting the GPU work would only muddy the
    latency numbers.
    """
    device = device if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    windows = video_windows(video_id, scale=scale, seq_len=seq_len,
                            stride=stride, features=features)
    probs = torch.sigmoid(model(windows.to(device))).cpu().numpy()

    over = np.nonzero(probs >= threshold)[0]
    if early_exit and len(over):
        first = int(over[0])
        return {"video_id": video_id, "pred": 1, "prob": float(probs[first]),
                "trigger_window": first, "n_windows": len(probs),
                "window_probs": probs.tolist()}

    return {"video_id": video_id, "pred": int(len(over) > 0),
            "prob": float(probs.max()), "trigger_window": None,
            "n_windows": len(probs), "window_probs": probs.tolist()}


def evaluate_full_video(
    model,
    video_ids: list[str],
    labels: dict[str, int],
    scale: str = cfg.DEFAULT_SCALE,
    seq_len: int = cfg.FINAL_TRAIN.seq_len,
    stride: int = cfg.EVAL.sliding_stride,
    threshold: float = cfg.EVAL.threshold,
    features: str = "full",
    device: str = "cuda",
    progress: bool = True,
) -> dict:
    """Run the sliding-window protocol over a set of videos."""
    iterator = video_ids
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(video_ids, desc="sliding-window eval", unit="vid")

    probs, targets, details = [], [], []
    for vid in iterator:
        r = predict_video(model, vid, scale=scale, seq_len=seq_len, stride=stride,
                          threshold=threshold, features=features, device=device)
        probs.append(r["prob"])
        targets.append(labels[vid])
        details.append({k: r[k] for k in ("video_id", "pred", "prob",
                                          "trigger_window", "n_windows")})

    return {"protocol": "full_video_sliding_window",
            "stride": stride,
            **binary_metrics(targets, probs, threshold),
            "probs": probs, "targets": targets, "details": details}


def compare_protocols(random_clip: dict, full_video: dict) -> str:
    """Side-by-side table of the two protocols, with the error trade spelled out."""
    rows = [
        ("Accuracy",    random_clip["accuracy"],  full_video["accuracy"]),
        ("Precision",   random_clip["precision"], full_video["precision"]),
        ("Recall",      random_clip["recall"],    full_video["recall"]),
        ("F1",          random_clip["f1"],        full_video["f1"]),
        ("Specificity", random_clip["specificity"], full_video["specificity"]),
    ]
    lines = [f"{'metric':<13} {'random clip':>12} {'full video':>12} {'delta':>9}",
             "-" * 49]
    for name, a, b in rows:
        lines.append(f"{name:<13} {a:>11.2%} {b:>12.2%} {b - a:>+9.2%}")
    lines.append("-" * 49)
    lines.append(f"{'false neg':<13} {random_clip['fn']:>11} {full_video['fn']:>12} "
                 f"{full_video['fn'] - random_clip['fn']:>+9}")
    lines.append(f"{'false pos':<13} {random_clip['fp']:>11} {full_video['fp']:>12} "
                 f"{full_video['fp'] - random_clip['fp']:>+9}")
    return "\n".join(lines)


def strip_arrays(result: dict) -> dict:
    """Drop the per-sample arrays so a result is small enough to keep in JSON."""
    return {k: v for k, v in result.items()
            if k not in ("probs", "targets", "details", "window_probs")}


def window_dropout(
    clips,
    scale: str = cfg.DEFAULT_SCALE,
    seq_len: int = cfg.FINAL_TRAIN.seq_len,
    frame_stride: int | None = None,
) -> np.ndarray:
    """Fraction of each window's frames in which the pose model found nobody.

    The companion to `score_clips`: same clips table, same index space, same
    windowing -- including `take_window`'s start clamp, so row *i* here
    describes exactly the window row *i* of `score_clips` was scored on. Without
    that alignment the two would silently disagree near the end of a video.

    Measured over the *real* rows only. A short sequence is padded by repeating
    its last frame, and repeated frames are not evidence about whether anybody
    was present.

    This is what `fallcore.infer`'s detection gate thresholds, lifted out so the
    benchmark notebooks can ask what that gate would cost them. Notebook 13 uses
    it; nothing else does, and no evaluation applies it by default.
    """
    from .data import prepare_sequence
    from .extract import load_keypoints

    fs = cfg.PREPROCESS.frame_stride if frame_stride is None else frame_stride
    out = np.zeros(len(clips), dtype=float)

    for vid, group in clips.groupby("video_id"):
        seq = prepare_sequence(load_keypoints(scale, vid), None, fs)
        present = seq[:, :, 2].sum(axis=1) > 0
        T = len(present)
        vals = []
        for s in group.start:
            s = int(s)
            # Mirror take_window: clamp into range, then read seq_len rows.
            s = 0 if T <= seq_len else max(0, min(s, T - seq_len))
            seen = present[s:s + seq_len]
            vals.append(1.0 - seen.mean() if len(seen) else 1.0)
        out[clips.index.get_indexer(group.index)] = vals
    return out


def score_clips(
    model,
    model_cfg,
    train_cfg,
    clips,
    scale: str = cfg.DEFAULT_SCALE,
    frame_stride: int | None = None,
    fill_gaps: bool = False,
    device: str | None = None,
    aspects: dict[str, float] | None = None,
) -> np.ndarray:
    """P(fall) for every window in a clips table, aligned to its row order.

    `clips` is the frame produced by the cross-dataset adapters: one row per
    window, with `video_id` and `start` in the strided index space. Windows are
    grouped by video so each keypoint file is read and strided once rather than
    once per window.

    Returned probabilities are placed by positional index, so the array lines up
    with `clips` as given even when the frame is not sorted by video.
    """
    import torch

    from .data import build_features, fill_detection_gaps, prepare_sequence, take_window
    from .extract import load_keypoints

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fs = cfg.PREPROCESS.frame_stride if frame_stride is None else frame_stride
    probs = np.zeros(len(clips), dtype=np.float64)

    model.to(device).eval()
    with torch.no_grad():
        for vid, group in clips.groupby("video_id"):
            seq = prepare_sequence(load_keypoints(scale, vid), None, fs)
            if fill_gaps:
                seq = fill_detection_gaps(seq)
            a = 1.0 if aspects is None else aspects.get(vid, 1.0)
            windows = np.stack([
                build_features(take_window(seq, train_cfg.seq_len, start=int(s)),
                               model_cfg.features, a)
                for s in group.start
            ])
            out = torch.sigmoid(
                model(torch.from_numpy(windows).float().to(device))).cpu().numpy()
            probs[clips.index.get_indexer(group.index)] = out
    return probs
