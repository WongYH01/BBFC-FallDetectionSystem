"""The deployment head: a logistic probe on the frozen transformer's pool.

Why this exists, in one paragraph. On the deployment camera the pretrained
classifier's *decision surface* is what fails, not the features: a two-parameter
logistic recalibration of its logit changes nothing (window AUC 0.8714 ->
0.8708), while a logistic probe on the same frozen 256-d pool -- fitted on four
of the five recorded subjects and scored on the fifth -- lifts window AUC to
0.9764 and the causal rolling-b8 clip F1 from 0.836 to 0.947 with no ADL false
alarms (`runs/metrics/calibration_head.csv`). So the shipped artefact is the
frozen backbone plus a ~1 KB head, refitted per room from a short commissioning
recording. Nothing about the backbone or the training pipeline changes.

What that costs, stated plainly: the head is fitted on the room's own clips,
both classes included. A site that can record its ADLs but never a fall cannot
fit it as-is (it degenerates to the threshold-only recalibration, which the
numbers above say buys nothing); the honest alternatives are a staged session,
fall footage from that room, or pseudo-labelling the stream -- none of which
this module does for you.

The head is applied after `model.encoder(model.embed(x))` and `model._pool`,
which is exactly the forward path `FallDetectorTransformer.forward` takes; the
only change to inference economics is one 256-d dot product per window.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import config as cfg

# The floor below which a fitted head is not worth shipping: the pretrained
# head's own cross-subject AUC on picam. Refuse rather than silently degrade.
MIN_USEFUL_AUC = 0.90


@dataclass
class ProbeHead:
    """Standardise a pooled embedding, then a logistic model on top of it."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    backbone: str = ""
    features: str = "full"
    seq_len: int = 60

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def proba(self, embeddings: np.ndarray) -> np.ndarray:
        """P(fall) for an `(n, dim)` block of pooled embeddings."""
        e = np.asarray(embeddings, dtype=np.float64)
        if e.ndim == 1:
            e = e[None, :]
        z = (e - self.mean) / self.scale
        logit = z @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-logit))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, scale=self.scale, coef=self.coef,
                 intercept=self.intercept, backbone=self.backbone,
                 features=self.features, seq_len=self.seq_len)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ProbeHead":
        blob = np.load(path, allow_pickle=False)
        return cls(mean=blob["mean"], scale=blob["scale"], coef=blob["coef"],
                   intercept=np.atleast_1d(blob["intercept"]),
                   backbone=str(blob["backbone"]), features=str(blob["features"]),
                   seq_len=int(blob["seq_len"]))


def fit(embeddings: np.ndarray, labels, C: float = 1.0,
        backbone: str = "", features: str = "full",
        seq_len: int = 60) -> ProbeHead:
    """Fit a head on pooled embeddings. `labels` is per-window, not per-clip."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    e = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("a probe needs both classes; one is missing here")
    scaler = StandardScaler().fit(e)
    head = LogisticRegression(max_iter=2000, C=C).fit(scaler.transform(e), y)
    return ProbeHead(mean=scaler.mean_, scale=scaler.scale_,
                     coef=head.coef_.ravel().astype(np.float64),
                     intercept=head.intercept_.astype(np.float64),
                     backbone=backbone, features=features, seq_len=seq_len)


def validate(head: "ProbeHead", checkpoint: str | Path, features: str,
             seq_len: int) -> None:
    """Refuse a head that was not fitted for this backbone, rather than apply it.

    A head is a function of one embedding space. On a different checkpoint it is
    a random linear map that still returns confident-looking numbers, and on a
    different feature variant the embedding does not even mean the same thing.
    """
    name = Path(checkpoint).name
    if head.backbone and Path(head.backbone).name != name:
        raise ValueError(
            f"head was fitted for {Path(head.backbone).name}, not {name} -- "
            f"refit it (scripts/fit_calibration_head.py)")
    if head.features != features:
        raise ValueError(
            f"head reads {head.features!r} features but {name} emits "
            f"{features!r}")
    if head.seq_len != seq_len:
        raise ValueError(
            f"head was fitted on {head.seq_len}-row windows, but {name} is "
            f"scored on {seq_len}")


def embeddings(model, model_cfg, train_cfg, clips, scale: str = cfg.DEFAULT_SCALE,
               device: str = "cpu") -> np.ndarray:
    """Pooled `(n_windows, d_model)` embeddings, backbone frozen.

    Same windowing as `evaluate.score_clips` (`prepare_sequence` on the cached
    keypoints, then `take_window` per row's `start`), so a head fitted here is
    applied to exactly the windows the rest of the harness scores. Reads the
    `.npy` cache only; no video is touched.
    """
    from .data import build_features, prepare_sequence, take_window
    from .extract import load_keypoints

    device = device if torch.cuda.is_available() else "cpu"
    fs = cfg.PREPROCESS.frame_stride
    out = np.zeros((len(clips), model_cfg.d_model), dtype=np.float32)
    model.to(device).eval()
    with torch.no_grad():
        for vid, group in clips.groupby("video_id"):
            seq = prepare_sequence(load_keypoints(scale, vid), None, fs)
            windows = np.stack([
                build_features(take_window(seq, train_cfg.seq_len, start=int(s)),
                               model_cfg.features)
                for s in group.start
            ])
            x = torch.from_numpy(windows).float().to(device)
            h = model.encoder(model.embed(x))
            out[clips.index.get_indexer(group.index)] = (
                model._pool(h).cpu().numpy())
    return out


def score_clips(model, model_cfg, train_cfg, clips, head: ProbeHead,
                scale: str = cfg.DEFAULT_SCALE, device: str = "cpu") -> np.ndarray:
    """P(fall) per clip window: frozen backbone, then `head`."""
    return head.proba(embeddings(model, model_cfg, train_cfg, clips,
                                 scale=scale, device=device))
