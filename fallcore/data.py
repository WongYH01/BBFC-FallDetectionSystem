"""Datasets over the cached keypoints: windowing, padding, feature variants.

Everything reads the .npy cache written by fallcore.extract, so nothing here
touches video or the GPU. A full split loads into RAM in a couple of hundred
megabytes, which is why the training loop is fast enough to run a 54-config
ablation sweep.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import config as cfg
from .extract import keypoints_path


# ---------------------------------------------------------------------------
# Feature variants (ablated in Section IV-A-3)
# ---------------------------------------------------------------------------
def build_features(seq: np.ndarray, variant: str = "full",
                   aspect: float = 1.0) -> np.ndarray:
    """Map a (T, 17, 3) keypoint window to a (T, D) feature matrix.

    "full"       D=51   flatten (y, x, conf)                      [paper default]
    "coords"     D=34   drop the confidence channel
    "velocity"   D=85   full, plus frame-to-frame deltas of (y, x)
    "centered"   D=51   translate to the hip midpoint and rescale
    "accel"      D=119  full, plus per-second velocity and acceleration
    "angles"     D=59   full, plus knee/hip articulation angles and their rates
    "kinematic"  D=127  full + velocity + acceleration + angles + angular rates

    `aspect` is the source frame's width/height. Coordinates reach here divided
    by the frame's own height and width separately (see fallcore.extract), so a
    16:9 clip and a 1:1 clip encode the same physical limb at different apparent
    slopes. That does not matter for the four original variants -- they never
    mix the two axes -- but an articulation angle is exactly such a mixture, so
    the last three rescale x into units of frame height before measuring. The
    Kaggle corpus really does mix aspects (3,877 clips at 16:9, 2,384 at ~1:1,
    76 portrait), which is why this is a correction and not a formality.
    """
    T = seq.shape[0]

    if variant == "full":
        return seq.reshape(T, -1)

    if variant == "coords":
        return seq[:, :, :2].reshape(T, -1)

    if variant == "velocity":
        coords = seq[:, :, :2]
        # Prepend a zero row so the velocity block is the same length as the
        # sequence; the first frame has no predecessor to difference against.
        vel = np.diff(coords, axis=0, prepend=coords[:1])
        return np.concatenate([seq.reshape(T, -1), vel.reshape(T, -1)], axis=1)

    if variant == "centered":
        out = seq.copy()
        hips = seq[:, [cfg.LEFT_HIP, cfg.RIGHT_HIP], :2]
        centre = hips.mean(axis=1, keepdims=True)             # (T, 1, 2)
        out[:, :, :2] = seq[:, :, :2] - centre
        # Rescale by the per-frame spread so the pose is size-invariant. The
        # paper reports this variant diverging during training; the guard below
        # keeps it from being an outright division by zero when a frame has no
        # detection, so the failure is the method's and not an arithmetic one.
        scale = np.abs(out[:, :, :2]).max(axis=(1, 2), keepdims=True)
        out[:, :, :2] = out[:, :, :2] / np.maximum(scale, 1e-6)
        return out.reshape(T, -1)

    # The three kinematic variants below are the proposal in the 2026 dataset
    # review: hand the attention heads explicit derivatives and joint
    # articulation instead of making them infer both from raw coordinates.
    #
    # **Tested and rejected, 2026-09-12.** Five arms x three seeds at the
    # paper's final config, scored on the 60-clip picam set
    # (`runs/metrics/kinematic_features.csv`). Every arm cost picam AUC:
    # velocity -0.017, kinematic -0.025, accel -0.039, angles -0.056 against a
    # 0.9415 baseline, and only `accel` clears its own seed noise -- in the
    # wrong direction, reproducibly, at 0.903 on all three seeds. False alarms
    # at 100% recall went 16.0 -> 19.7 (accel) and -> 20.0 (kinematic).
    #
    # The reason is in `runs/metrics/braking_probe.csv`: deceleration divided by
    # descent speed separates falls from controlled descents with Cohen's
    # d = -0.05. The "impact spike" is entirely explained by a faster descent
    # needing a harder stop, so the second derivative carries nothing the first
    # does not already carry, and 34 extra mostly-zero channels dilute the
    # position block that does the work. `angles` posts the best in-domain
    # number in the whole table (Kaggle F1 0.9788 at seed 99) and the worst
    # out-of-domain AUC, which is the usual shape of an overfit feature.
    #
    # Kept because the implementation is correct and the negative result is
    # worth being able to re-run; do not reach for these without new evidence.
    if variant in ("accel", "angles", "kinematic"):
        blocks = [seq.reshape(T, -1)]
        if variant in ("accel", "kinematic"):
            vel, acc = _derivatives(seq, aspect)
            blocks += [vel.reshape(T, -1), acc.reshape(T, -1)]
        if variant in ("angles", "kinematic"):
            ang, dang = _articulation(seq, aspect)
            blocks += [ang, dang]
        return np.concatenate(blocks, axis=1)

    raise ValueError(f"unknown feature variant {variant!r}")


# Seconds between consecutive cached frames: the corpus is nominally 30 fps and
# extract keeps every frame_stride-th one. Derivatives are expressed per second
# rather than per frame so acceleration lands at the same order of magnitude as
# a normalised coordinate (~0.5) instead of ~0.002, where the input projection
# would see it as noise against the position block. A single nominal constant is
# enough: it is a uniform rescaling, so only the ratio between blocks matters.
_DT = cfg.PREPROCESS.frame_stride / cfg.TRAIN_NOMINAL_FPS


def _geometric(seq: np.ndarray, aspect: float) -> np.ndarray:
    """(T, 17, 2) coordinates in units of frame height, x undistorted."""
    geo = seq[:, :, :2].copy()
    geo[:, :, 1] *= aspect
    return geo


def _derivatives(seq: np.ndarray, aspect: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-second first and second time derivatives of every keypoint.

    Both are prepended with a zero row so they stay the same length as the
    sequence. No smoothing is applied: a Savitzky-Golay filter would tame the
    jitter that differencing twice amplifies, but it would equally blunt the
    impact spike that is the whole reason to compute acceleration, so the raw
    form is the one that tests the claim being made.

    Derivatives are masked to frames where the keypoint was actually detected at
    both ends of the difference. Without that, a keypoint blinking out lands at
    exactly (0, 0) -- the image corner -- and differencing reads the jump as a
    velocity of order 10 and an acceleration of order 400, swamping every real
    movement in the window. That is not a scaling nuisance: detection dropout is
    correlated with the Fall label in this corpus, so an unmasked acceleration
    block would mostly re-encode the dropout shortcut and the experiment would
    measure that instead of the braking signature it is meant to test.
    """
    geo = _geometric(seq, aspect)
    present = seq[:, :, 2] > 0                                  # (T, 17)

    vel = np.diff(geo, axis=0, prepend=geo[:1]) / _DT
    v_ok = np.zeros_like(present)
    v_ok[1:] = present[1:] & present[:-1]
    vel[~v_ok] = 0.0

    acc = np.diff(vel, axis=0, prepend=vel[:1]) / _DT
    a_ok = np.zeros_like(v_ok)
    a_ok[1:] = v_ok[1:] & v_ok[:-1]
    acc[~a_ok] = 0.0

    # Both blocks are still an order of magnitude above the position block after
    # masking, simply because a second per-second derivative is a large number.
    # Squash rather than rescale: tanh keeps small differences linear, where the
    # braking-versus-impact distinction lives, and saturates the residual
    # single-frame jitter spikes instead of letting them set the layer's scale.
    return np.tanh(vel / 4.0), np.tanh(acc / 40.0)


# (joint, neighbour_a, neighbour_b) -- the angle is measured *at* `joint`.
_ARTICULATIONS = [
    (13, 11, 15),   # left knee:  hip - knee - ankle
    (14, 12, 16),   # right knee
    (11,  5, 13),   # left hip:   shoulder - hip - knee
    (12,  6, 14),   # right hip
]


def _articulation(seq: np.ndarray, aspect: float) -> tuple[np.ndarray, np.ndarray]:
    """Knee and hip flexion angles in [0, 1] (radians / pi), and their rates.

    Undetected keypoints arrive as exact zeros, for which an angle is undefined
    rather than small; those frames get 0.0 and are excluded from the rate, so a
    detection dropping in and out cannot manufacture a hyperflexion spike. The
    confidence channel in the `full` block still reports the dropout, so nothing
    is hidden -- it is just no longer disguised as motion.
    """
    geo = _geometric(seq, aspect)
    conf = seq[:, :, 2]
    T = seq.shape[0]
    ang = np.zeros((T, len(_ARTICULATIONS)), dtype=np.float32)
    valid = np.zeros((T, len(_ARTICULATIONS)), dtype=bool)

    for i, (j, a, b) in enumerate(_ARTICULATIONS):
        ok = (conf[:, j] > 0) & (conf[:, a] > 0) & (conf[:, b] > 0)
        u = geo[:, a] - geo[:, j]
        v = geo[:, b] - geo[:, j]
        nu = np.linalg.norm(u, axis=1)
        nv = np.linalg.norm(v, axis=1)
        ok &= (nu > 1e-6) & (nv > 1e-6)
        cos = np.clip((u * v).sum(axis=1) / np.maximum(nu * nv, 1e-6), -1.0, 1.0)
        ang[ok, i] = np.arccos(cos[ok]) / np.pi
        valid[:, i] = ok

    dang = np.diff(ang, axis=0, prepend=ang[:1]) / _DT
    # Zero any rate whose two endpoints are not both real measurements.
    both = np.zeros_like(valid)
    both[1:] = valid[1:] & valid[:-1]
    dang[~both] = 0.0
    # Squashed on the same argument as the linear derivatives: a knee flexing
    # through a right angle in half a second reads 1.0 here and stays linear,
    # while the residual per-frame estimator jitter saturates instead of
    # setting the block's scale.
    return ang, np.tanh(dang / 2.0).astype(np.float32)


def feature_dim(variant: str = "full") -> int:
    return {"full": 51, "coords": 34, "velocity": 85, "centered": 51,
            "accel": 119, "angles": 59, "kinematic": 127}[variant]


def fill_detection_gaps(seq: np.ndarray, max_gap: int | None = None) -> np.ndarray:
    """Interpolate keypoint positions across frames where nothing was detected.

    `extract.extract_video` writes an all-zero row when the pose model finds
    nobody. Those zeros are not a neutral placeholder: (0, 0) is the top-left
    corner, a position no hip ever occupies, so a dropped frame enters the
    network as an *impossible pose* rather than as missing data.

    That matters because dropout is correlated with the label in the training
    corpus -- Fall clips lose the subject in 17% of frames against 1% for
    No-Fall -- so the classifier learns "the detector lost the person" as
    evidence of a fall. The shortcut survives dropping the confidence channel,
    because the zeroed coordinates carry it too. On CAUCAFall, where ordinary
    kneeling loses detections six times as often as the training negatives did,
    it misfires: 26% of the false positives there trace to it.

    Filling the gaps removes the artefact without hiding anything -- the
    confidence channel still reads 0 on an interpolated frame, so the
    information remains available as information rather than as geometry.

    `max_gap` leaves runs longer than that many frames untouched: a stretch
    where the subject was genuinely absent for seconds should not be bridged
    with a fabricated trajectory.
    """
    out = seq.copy()
    present = seq[:, :, 2].sum(axis=1) > 0
    if present.all() or not present.any():
        return out

    idx = np.flatnonzero(present)
    missing = np.flatnonzero(~present)

    if max_gap is not None:
        # Split the missing frames into consecutive runs and drop the long ones.
        keep = []
        for run in np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1):
            if run.size and run.size <= max_gap:
                keep.append(run)
        missing = np.concatenate(keep) if keep else np.array([], dtype=int)
        if missing.size == 0:
            return out

    for k in range(seq.shape[1]):
        for c in (0, 1):
            out[missing, k, c] = np.interp(missing, idx, seq[idx, k, c])
    return out


# ---------------------------------------------------------------------------
# Sequence preparation (Algorithm 1, lines 6 and 11)
# ---------------------------------------------------------------------------
def prepare_sequence(
    arr: np.ndarray,
    max_frames: int | None = cfg.PREPROCESS.max_frames,
    frame_stride: int = cfg.PREPROCESS.frame_stride,
) -> np.ndarray:
    """Take the head of a video and subsample it, per Algorithm 1 line 6.

    With the paper's defaults a clip becomes at most 60 frames (first 120, every
    2nd). Raising max_frames leaves more sequence for the random window to move
    within -- see the note in take_window.
    """
    if max_frames is not None:
        arr = arr[:max_frames]
    return arr[::frame_stride]


def take_window(
    seq: np.ndarray,
    seq_len: int,
    rng: np.random.Generator | None = None,
    start: int | None = None,
) -> np.ndarray:
    """Return exactly `seq_len` frames, padding by repeating the last frame.

    A note on the paper's own numbers: Algorithm 1 builds a 60-frame sequence
    and then samples a random window of length T from it. At the ablation
    baseline (T=45) that is a real augmentation; at the final config (T=60) the
    window is the whole sequence and the "random" crop is a no-op. We reproduce
    that faithfully by default. Raising PREPROCESS.max_frames restores genuine
    random cropping at T=60, which is a documented departure, not the paper.
    """
    T = seq.shape[0]

    if T == 0:
        return np.zeros((seq_len, cfg.N_KEYPOINTS, 3), dtype=np.float32)

    if T < seq_len:
        pad = np.repeat(seq[-1:], seq_len - T, axis=0)
        return np.concatenate([seq, pad], axis=0)

    if start is None:
        start = 0 if rng is None else int(rng.integers(0, T - seq_len + 1))
    start = max(0, min(start, T - seq_len))
    return seq[start:start + seq_len]


def sliding_starts(n_frames: int, seq_len: int, stride: int) -> list[int]:
    """Window start offsets for Algorithm 2, always yielding at least one."""
    if n_frames <= seq_len:
        return [0]
    return list(range(0, n_frames - seq_len + 1, stride))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class KeypointClipDataset(Dataset):
    """One random (or fixed) window per video -- the random-clip protocol.

    Sequences are loaded once into memory at construction. `train=True` draws a
    fresh random window each epoch; `train=False` pins the window so validation
    and test numbers do not jitter between evaluations.
    """

    def __init__(
        self,
        video_ids: list[str],
        labels: dict[str, int],
        scale: str = cfg.DEFAULT_SCALE,
        seq_len: int = cfg.FINAL_TRAIN.seq_len,
        features: str = "full",
        train: bool = True,
        seed: int = cfg.SPLIT_SEED,
        max_frames: int | None = cfg.PREPROCESS.max_frames,
        frame_stride: int = cfg.PREPROCESS.frame_stride,
        fill_gaps: bool = False,
        frame_dropout: float = 0.0,
        aspects: dict[str, float] | None = None,
    ):
        self.video_ids = list(video_ids)
        self.labels = labels
        self.seq_len = seq_len
        self.features = features
        self.train = train
        self.fill_gaps = fill_gaps
        # Frame width/height per video, for the angle-bearing feature variants.
        # Absent means 1.0, which reproduces every earlier run exactly.
        self.aspects = [1.0 if aspects is None else aspects.get(v, 1.0)
                        for v in self.video_ids]
        # Randomly blanking frames during training decorrelates detection
        # dropout from the label, so the network cannot use "the detector lost
        # the person" as a shortcut. Applied to training draws only -- an
        # augmentation, not a preprocessing step.
        self.frame_dropout = frame_dropout
        self.rng = np.random.default_rng(seed)

        self.sequences: list[np.ndarray] = []
        self.targets: list[float] = []
        missing = []
        for vid in self.video_ids:
            p = keypoints_path(scale, vid)
            if not p.exists():
                missing.append(vid)
                continue
            arr = prepare_sequence(np.load(p), max_frames, frame_stride)
            if fill_gaps:
                arr = fill_detection_gaps(arr)
            self.sequences.append(arr)
            self.targets.append(float(labels[vid]))

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} videos have no cached keypoints for scale "
                f"{scale!r} (first: {missing[:3]}). Run notebook 01 for this scale."
            )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        # A deterministic per-item start in eval mode: centred if there is room
        # to move, otherwise 0. Same window every call.
        start = None if self.train else max(0, (len(seq) - self.seq_len) // 2)
        window = take_window(seq, self.seq_len, rng=self.rng, start=start)
        if self.train and self.frame_dropout > 0:
            drop = self.rng.random(len(window)) < self.frame_dropout
            if drop.any():
                window = window.copy()
                window[drop] = 0.0
        x = build_features(window, self.features, self.aspects[idx])
        return torch.from_numpy(np.ascontiguousarray(x)).float(), \
            torch.tensor(self.targets[idx], dtype=torch.float32)


def make_loaders(
    splits: dict[str, list[str]],
    labels: dict[str, int],
    scale: str = cfg.DEFAULT_SCALE,
    seq_len: int = cfg.FINAL_TRAIN.seq_len,
    batch_size: int = cfg.FINAL_TRAIN.batch_size,
    features: str = "full",
    seed: int = cfg.SPLIT_SEED,
    num_workers: int = 0,
    max_frames: int | None = cfg.PREPROCESS.max_frames,
    frame_stride: int = cfg.PREPROCESS.frame_stride,
    fill_gaps: bool = False,
    frame_dropout: float = 0.0,
    aspects: dict[str, float] | None = None,
) -> dict[str, DataLoader]:
    """Train/val/test loaders over one backbone scale's keypoint cache."""
    loaders = {}
    for name in ("train", "val", "test"):
        ds = KeypointClipDataset(
            splits[name], labels, scale=scale, seq_len=seq_len,
            features=features, train=(name == "train"), seed=seed,
            max_frames=max_frames, frame_stride=frame_stride,
            fill_gaps=fill_gaps, frame_dropout=frame_dropout, aspects=aspects,
        )
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


class KeypointWindowDataset(Dataset):
    """One item per *labelled window*, for densely annotated corpora.

    `KeypointClipDataset` carries one label per video, which is what the Kaggle
    compilation is: short clips that are entirely a fall or entirely not. That
    shape does not survive contact with OmniFall, where a single MCFD recording
    runs for minutes and contains a fall, the lying that follows it, and a great
    deal of ordinary activity. Collapsing such a video to one label would throw
    away every negative window in it and mislabel most of the positives.

    So the unit here is the window, not the video: `clips` supplies `video_id`,
    `start` and `label`, and the label is the one OmniFall's segments imply for
    that exact span. Sequences are loaded once per video and shared across all
    windows that index into them, which matters when one video contributes
    dozens of rows.

    `start` is in *strided* index space, the same space `omnifall.build_clips`
    and `evaluate.score_clips` use, so rows line up positionally across all
    three. `max_frames` therefore defaults to None: truncating to the paper's
    first 120 frames would put most windows past the end of their own sequence.

    `reverse=True` plays every window backwards, which turns a `stand_up` into
    a synthetic controlled descent -- the class the Kaggle corpus is short of
    (238 clips against 1,814 falls carrying the same upright-to-horizontal
    transition).

    **Tested and rejected; do not reach for this without new evidence.** Adding
    2,272 reversed stand-ups as No-Fall looked like a large win at threshold
    0.5 -- picam false alarms fell 15 -> 3 and F1 rose 0.800 -> 0.842 -- but the
    AUC *fell*, 0.9533 -> 0.9411, and at the threshold that catches every fall
    the false alarms rose 12 -> 16. The F1 gain was the decision boundary
    moving, not the model separating the classes better. The damage lands
    exactly where the trick is weakest: a reversed stand-up ends lying down, so
    it teaches "descends and ends horizontal = not a fall", and `lie down fall`
    -- a fall that starts from lying -- collapsed from 0.999 to 0.09-0.41 on
    four of five clips. Kept as an option because the mechanism is sound and a
    corpus with real fall-from-lying examples might change the result.
    """

    def __init__(
        self,
        clips,
        scale: str = cfg.DEFAULT_SCALE,
        seq_len: int = cfg.FINAL_TRAIN.seq_len,
        features: str = "full",
        train: bool = True,
        seed: int = cfg.SPLIT_SEED,
        max_frames: int | None = None,
        frame_stride: int = cfg.PREPROCESS.frame_stride,
        fill_gaps: bool = False,
        frame_dropout: float = 0.0,
        multiclass: bool = False,
        reverse: bool = False,
        aspects: dict[str, float] | None = None,
    ):
        self.seq_len = seq_len
        self.features = features
        self.train = train
        self.frame_dropout = frame_dropout
        self.multiclass = multiclass
        self.reverse = reverse
        self.aspect_map = aspects or {}
        self.rng = np.random.default_rng(seed)

        clips = clips.reset_index(drop=True)
        self.starts = clips["start"].to_numpy(dtype=np.int64)
        # CrossEntropyLoss wants an integer class index; BCEWithLogitsLoss wants
        # a float in {0, 1}. The dtype is the only difference between the two
        # modes as far as this dataset is concerned.
        self.targets = clips["label"].to_numpy(
            dtype=np.int64 if multiclass else np.float32)
        self.video_ids = clips["video_id"].tolist()

        # One load per video, however many windows index into it.
        cache: dict[str, np.ndarray] = {}
        missing = []
        for vid in dict.fromkeys(self.video_ids):
            p = keypoints_path(scale, vid)
            if not p.exists():
                missing.append(vid)
                continue
            arr = prepare_sequence(np.load(p), max_frames, frame_stride)
            cache[vid] = fill_detection_gaps(arr) if fill_gaps else arr

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} videos have no cached keypoints for scale "
                f"{scale!r} (first: {missing[:3]}). Run omnifall.ingest().")

        self.cache = cache

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        seq = self.cache[self.video_ids[idx]]
        # The start is pinned even in training mode. Re-drawing it at random,
        # as the random-clip protocol does, would move the window away from the
        # span its label was computed for -- the label and the start are one
        # annotation here, not two independent choices.
        window = take_window(seq, self.seq_len, rng=self.rng,
                             start=int(self.starts[idx]))
        if self.reverse:
            window = window[::-1]
        if self.train and self.frame_dropout > 0:
            drop = self.rng.random(len(window)) < self.frame_dropout
            if drop.any():
                window = window.copy()
                window[drop] = 0.0
        x = build_features(window, self.features,
                           self.aspect_map.get(self.video_ids[idx], 1.0))
        y = torch.tensor(self.targets[idx],
                         dtype=torch.long if self.multiclass else torch.float32)
        return torch.from_numpy(np.ascontiguousarray(x)).float(), y


def make_clip_loaders(
    parts: dict,
    scale: str = cfg.DEFAULT_SCALE,
    seq_len: int = cfg.FINAL_TRAIN.seq_len,
    batch_size: int = cfg.FINAL_TRAIN.batch_size,
    features: str = "full",
    seed: int = cfg.SPLIT_SEED,
    num_workers: int = 0,
    max_frames: int | None = None,
    frame_stride: int = cfg.PREPROCESS.frame_stride,
    fill_gaps: bool = False,
    frame_dropout: float = 0.0,
    multiclass: bool = False,
) -> dict[str, DataLoader]:
    """Train/val/test loaders over window tables, for `train_model(loaders=...)`.

    `parts` maps each split name to its own clips DataFrame, already partitioned
    by video so no video contributes windows to two splits. With `multiclass`,
    `label` is an OmniFall activity id rather than a 0/1 fall flag.
    """
    loaders = {}
    for name in ("train", "val", "test"):
        ds = KeypointWindowDataset(
            parts[name], scale=scale, seq_len=seq_len, features=features,
            train=(name == "train"), seed=seed, max_frames=max_frames,
            frame_stride=frame_stride, fill_gaps=fill_gaps,
            frame_dropout=frame_dropout, multiclass=multiclass,
        )
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def video_windows(
    video_id: str,
    scale: str = cfg.DEFAULT_SCALE,
    seq_len: int = cfg.FINAL_TRAIN.seq_len,
    stride: int = cfg.EVAL.sliding_stride,
    features: str = "full",
    max_frames: int | None = None,
    frame_stride: int = cfg.PREPROCESS.frame_stride,
    fill_gaps: bool = False,
    aspect: float = 1.0,
) -> torch.Tensor:
    """Every sliding window of one video as a (W, seq_len, D) batch.

    Used by the full-video protocol in fallcore.evaluate.

    `max_frames` defaults to None -- the *whole* video -- unlike training, which
    reads only the first 120 frames. That asymmetry is the paper's: Algorithm 1
    trains on the head of each clip, while Algorithm 2 begins "run pose on all
    frames of v". Inheriting the training truncation here would cap every clip
    at exactly `seq_len` frames, leave one window per video, and silently turn
    the sliding-window protocol into the random-clip protocol wearing a hat.
    """
    seq = prepare_sequence(np.load(keypoints_path(scale, video_id)),
                           max_frames, frame_stride)
    if fill_gaps:
        seq = fill_detection_gaps(seq)
    starts = sliding_starts(len(seq), seq_len, stride)
    windows = [build_features(take_window(seq, seq_len, start=s), features, aspect)
               for s in starts]
    return torch.from_numpy(np.stack(windows)).float()


def labels_from_manifest(manifest) -> dict[str, int]:
    """video_id -> label, dropping anything unlabelled."""
    df = manifest[manifest.label >= 0]
    return dict(zip(df.video_id, df.label.astype(int)))


def aspects_from_manifest(*manifests) -> dict[str, float]:
    """video_id -> frame width/height, merged across any number of manifests.

    Only the angle-bearing feature variants consult this; pass the result as
    `aspects=` to a dataset or `aspect=` to `video_windows`. Rows without usable
    dimensions are omitted, and a missing key falls back to 1.0 at lookup.
    """
    out: dict[str, float] = {}
    for m in manifests:
        if m is None or "width" not in m or "height" not in m:
            continue
        ok = m[(m.width > 0) & (m.height > 0)]
        out.update(zip(ok.video_id, (ok.width / ok.height).astype(float)))
    return out


# ---------------------------------------------------------------------------
# Trajectory extraction for plotting and inspection
# ---------------------------------------------------------------------------
def masked_joint_trajectory(
    arr: np.ndarray,
    joint: int = cfg.RIGHT_HIP,
    channel: int = 0,
    min_conf: float = 0.0,
) -> np.ndarray:
    """One joint's coordinate over time, with undetected frames set to NaN.

    Frames where the pose model found nobody are stored as all-zero rows by
    `extract.extract_video`, so a naive plot of `arr[:, joint, 0]` renders a
    missing detection as a hip pinned to the very top of the frame. Masking
    those to NaN makes matplotlib break the line instead of drawing a spike,
    which is the difference between showing a gap and inventing a measurement.
    """
    y = arr[:, joint, channel].astype(float).copy()
    y[arr[:, joint, 2] <= min_conf] = np.nan
    return y


def resample_trajectory(y: np.ndarray, n: int = 100,
                        max_gap: float | None = None) -> np.ndarray:
    """Interpolate a trajectory onto `n` points of normalised time.

    Clips differ in length, so they must share a time axis before per-frame
    statistics across clips mean anything. A trajectory with fewer than two
    valid points is returned as all-NaN rather than fabricated.

    `max_gap` (a fraction of clip duration) controls honesty about dropouts:
    left as None every gap is bridged, which is what you want before taking a
    median across many clips. Set to, say, 0.03 and any output point further
    than that from a real observation stays NaN, so a long stretch of missing
    detections draws as a break instead of a straight line the subject never
    travelled.
    """
    y = np.asarray(y, dtype=float)
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return np.full(n, np.nan)

    t_src = np.linspace(0.0, 1.0, len(y))[valid]
    t_out = np.linspace(0.0, 1.0, n)
    out = np.interp(t_out, t_src, y[valid])

    if max_gap is not None:
        idx = np.searchsorted(t_src, t_out)
        left = t_src[np.clip(idx - 1, 0, len(t_src) - 1)]
        right = t_src[np.clip(idx, 0, len(t_src) - 1)]
        nearest = np.minimum(np.abs(t_out - left), np.abs(t_out - right))
        out[nearest > max_gap] = np.nan

    return out


def detection_dropout(arr: np.ndarray) -> float:
    """Fraction of frames in which the pose model detected nobody at all."""
    if len(arr) == 0:
        return 1.0
    return float((arr[:, :, 2].sum(axis=1) == 0).mean())
