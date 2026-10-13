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


#: Extra per-window inputs the head can take alongside the pooled embedding.
#: All three are measured in units of the subject's own body, never of the
#: frame, which is the point: picam-1's head separated falls from lying by
#: height *in frame*, and that cue is the furniture seen from one camera
#: position, so it weakened as soon as the camera was re-aimed
#: (`runs/metrics/picam2_what_fires.csv`).
#: "hip_drop" rather than "drop": these names become DataFrame columns,
#: and `df.drop` is a method, which silently shadows the column.
KINEMATIC_NAMES = ("descent_speed", "hip_drop", "extent_ratio")

#: Rows inside a trained window are `frame_stride` frames apart at the nominal
#: capture rate, so this is the rate the kinematics are differenced at.
ROW_HZ = cfg.TRAIN_NOMINAL_FPS / cfg.PREPROCESS.frame_stride


def window_kinematics(window: np.ndarray, aspect: float = 1.0) -> np.ndarray:
    """`(3,)` scale-free descent summary of one raw `(T, 17, 3)` window.

    * `descent_speed` -- the fastest the hips fall over any half second, in
      body lengths per second. On picam-2 this separates falls (0.75 +/- 0.19)
      from deliberate lying down (0.20 +/- 0.07) with no overlap, in a room
      neither the backbone nor the head had seen.
    * `hip_drop` -- how far the hips descend within the window, body lengths, taking
      the largest later-minus-earlier difference so a bob does not count.
    * `extent_ratio` -- the body's vertical extent at the end of the window over
      its largest extent in the window: ~1 upright, well under 1 lying down.

    Undetected frames are bridged rather than read as position (0, 0), and a
    window with almost nothing detected returns zeros -- no claim.

    **Everything is divided by the body's longest dimension**, not by its
    height: a lying body is barely tall, so scaling by height alone turns
    estimator jitter into an apparent sprint (measured: a motionless lying
    window read 0.19 body lengths per second). `aspect` is the frame's
    width/height, needed because x and y are normalised by different numbers of
    pixels -- without it a body lying across a 16:9 frame measures 1.8x shorter
    than the same body standing.
    """
    w = np.asarray(window, dtype=np.float64)
    seen = w[:, :, 2] > 0.3
    present = seen.any(axis=1)
    if present.sum() < 5:
        return np.zeros(len(KINEMATIC_NAMES))

    with np.errstate(all="ignore"):
        ys = np.where(seen, w[:, :, 0], np.nan)
        xs = np.where(seen, w[:, :, 1] * float(aspect), np.nan)
        extent = np.nanmax(ys, axis=1) - np.nanmin(ys, axis=1)
        across = np.nanmax(xs, axis=1) - np.nanmin(xs, axis=1)
    # The longest dimension is ~the body's length whether it is upright or flat.
    stand = float(np.nanmax(np.fmax(extent, across)))
    if not np.isfinite(stand) or stand < 1e-3:
        return np.zeros(len(KINEMATIC_NAMES))

    hips = w[:, [cfg.LEFT_HIP, cfg.RIGHT_HIP], :]
    ok = (hips[:, :, 2] > 0.3).all(axis=1)
    if ok.sum() < 5:
        return np.zeros(len(KINEMATIC_NAMES))
    idx = np.flatnonzero(ok)
    y = np.interp(np.arange(len(w)), idx, hips[idx, :, 0].mean(axis=1)) / stand

    k = max(1, round(0.5 * ROW_HZ))
    speed = float(np.max((y[k:] - y[:-k]) / (k / ROW_HZ))) if len(y) > k else 0.0

    # Largest later-minus-earlier drop: a running minimum makes it one pass.
    drop = float(np.max(y - np.minimum.accumulate(y)))

    # Upright ~1 (the body's height is its longest dimension), lying ~0.2.
    with np.errstate(all="ignore"):
        tail = float(np.nanmean(extent[-max(1, len(w) // 4):]))
    ratio = tail / stand if np.isfinite(tail) else 1.0
    return np.array([max(speed, 0.0), max(drop, 0.0), ratio])


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
    #: Names of the extra per-window inputs appended after the embedding, in
    #: order. Empty for a head fitted on the embedding alone -- which is what
    #: every head saved before 2026-09-24 is, so they keep loading unchanged.
    extra: tuple[str, ...] = ()

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def n_extra(self) -> int:
        return len(self.extra)

    def inputs(self, embeddings: np.ndarray,
               extras: np.ndarray | None = None) -> np.ndarray:
        """Embedding, with the extra inputs appended when this head takes them."""
        e = np.asarray(embeddings, dtype=np.float64)
        if e.ndim == 1:
            e = e[None, :]
        if not self.n_extra:
            return e
        if extras is None:
            raise ValueError(
                f"this head takes {self.extra} alongside the embedding; pass "
                f"extras (fallcore.calibrate.window_kinematics)")
        x = np.asarray(extras, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != self.n_extra:
            raise ValueError(f"head takes {self.n_extra} extra input(s) "
                             f"{self.extra}, got {x.shape[1]}")
        return np.concatenate([e, x], axis=1)

    def proba(self, embeddings: np.ndarray,
              extras: np.ndarray | None = None) -> np.ndarray:
        """P(fall) for an `(n, d_model)` block of pooled embeddings."""
        z = (self.inputs(embeddings, extras) - self.mean) / self.scale
        logit = z @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-logit))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, scale=self.scale, coef=self.coef,
                 intercept=self.intercept, backbone=self.backbone,
                 features=self.features, seq_len=self.seq_len,
                 # A plain unicode array, so loading never needs allow_pickle.
                 extra=np.array(self.extra, dtype="<U32"))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ProbeHead":
        blob = np.load(path, allow_pickle=False)
        # `extra` is absent from every head saved before 2026-09-24; those are
        # embedding-only and must keep loading untouched.
        extra = tuple(str(s) for s in blob["extra"]) if "extra" in blob else ()
        return cls(mean=blob["mean"], scale=blob["scale"], coef=blob["coef"],
                   intercept=np.atleast_1d(blob["intercept"]),
                   backbone=str(blob["backbone"]), features=str(blob["features"]),
                   seq_len=int(blob["seq_len"]), extra=extra)


def fit(embeddings: np.ndarray, labels, C: float = 1.0,
        backbone: str = "", features: str = "full",
        seq_len: int = 60, extras: np.ndarray | None = None,
        extra_names: tuple[str, ...] = ()) -> ProbeHead:
    """Fit a head on pooled embeddings. `labels` is per-window, not per-clip.

    `extras` appends per-window inputs (see `window_kinematics`) to every
    embedding before standardising, so the fitted head weighs them alongside the
    learned features rather than replacing them.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    e = np.asarray(embeddings, dtype=np.float64)
    if extras is not None:
        if not extra_names:
            extra_names = KINEMATIC_NAMES
        x = np.asarray(extras, dtype=np.float64)
        if x.shape[1] != len(extra_names):
            raise ValueError(f"{x.shape[1]} extra column(s) for names "
                             f"{extra_names}")
        e = np.concatenate([e, x], axis=1)
    else:
        extra_names = ()

    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("a probe needs both classes; one is missing here")
    scaler = StandardScaler().fit(e)
    head = LogisticRegression(max_iter=2000, C=C).fit(scaler.transform(e), y)
    return ProbeHead(mean=scaler.mean_, scale=scaler.scale_,
                     coef=head.coef_.ravel().astype(np.float64),
                     intercept=head.intercept_.astype(np.float64),
                     backbone=backbone, features=features, seq_len=seq_len,
                     extra=tuple(extra_names))


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


def window_extras(train_cfg, clips, scale: str = cfg.DEFAULT_SCALE,
                  aspects: dict | None = None) -> np.ndarray:
    """`(n_windows, 3)` kinematics for the same windows `embeddings` scores.

    Windowed identically -- `prepare_sequence` on the cached keypoints, then
    `take_window` per row's `start` -- so column i here belongs to row i there.
    Reads the `.npy` cache only.
    """
    from .data import prepare_sequence, take_window
    from .extract import load_keypoints

    fs = cfg.PREPROCESS.frame_stride
    out = np.zeros((len(clips), len(KINEMATIC_NAMES)), dtype=np.float64)
    for vid, group in clips.groupby("video_id"):
        seq = prepare_sequence(load_keypoints(scale, vid), None, fs)
        ar = 1.0 if aspects is None else float(aspects.get(vid, 1.0))
        rows = np.stack([
            window_kinematics(take_window(seq, train_cfg.seq_len, start=int(s)), ar)
            for s in group.start
        ])
        out[clips.index.get_indexer(group.index)] = rows
    return out


def score_clips(model, model_cfg, train_cfg, clips, head: ProbeHead,
                scale: str = cfg.DEFAULT_SCALE, device: str = "cpu",
                aspects: dict | None = None) -> np.ndarray:
    """P(fall) per clip window: frozen backbone, then `head`."""
    emb = embeddings(model, model_cfg, train_cfg, clips, scale=scale,
                     device=device)
    extras = (window_extras(train_cfg, clips, scale=scale, aspects=aspects)
              if head.n_extra else None)
    return head.proba(emb, extras)
