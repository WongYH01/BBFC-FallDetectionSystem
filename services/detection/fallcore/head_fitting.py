"""The camera-head fitting recipe: which windows a Camera Head is fitted on.

Shared by `scripts/fit_calibration_head.py`, which fits and prices a head on
the picam caches by hand, and the weekly Refit (`fallcore.refit`), which fits
a Candidate Head on the Commissioning Recording plus the camera's triaged
Alerts. One recipe, so a head the Refit promotes is fitted the way the shipped
one was: segment labels on the Commissioning Recording's fall clips, and the
synthetic nobody-visible windows labelled ADL.

Everything that reads keypoints takes `load(pose, video_id)`, `pose` naming
the pose backbone whose keypoints they are. It defaults to the keypoint cache
on disk, so a caller with the clips somewhere else (the Refit's mirrored
bucket, a test's stubs) passes its own.
"""
from __future__ import annotations

import logging

import numpy as np

from . import calibrate
from . import config as cfg
from .fall_decision import DEPLOYED_POSE

log = logging.getLogger(__name__)

#: Rows in half a second, at the row rate every window is cut at.
HALF_S = round(0.5 * calibrate.ROW_HZ)


def fall_row(pose: str, video_id: str, load=None) -> int:
    """Start row of the fastest half second of hip descent in a whole clip.

    The fall itself, as opposed to the standing before it or the lying after
    it. Undetected rows are bridged, as `calibrate.window_kinematics` does.
    """
    if load is None:
        from .extract import load_keypoints as load

    kp = load(pose, video_id)[::cfg.PREPROCESS.frame_stride]
    hips = kp[:, [cfg.LEFT_HIP, cfg.RIGHT_HIP], :]
    ok = (hips[:, :, 2] > 0.3).all(axis=1)
    idx = np.flatnonzero(ok)
    y = np.interp(np.arange(len(kp)), idx, hips[idx, :, 0].mean(axis=1))
    return int(np.argmax(y[HALF_S:] - y[:-HALF_S]))


def segment_labels(clips, pose: str, seq_len: int, load=None):
    """`(keep, label)` per window when a fall clip's label follows its fall.

    A fall clip is a few seconds of standing, about one second of falling, then
    lying still. Clip labels call every one of its windows Fall; here a window
    that ends before the fall is ADL (the same picture as walking), one that
    holds the whole fastest half second or starts after it is Fall, and one
    that cuts the fall in part is dropped. ADL clips' windows stay ADL. On the
    picam sessions this is what lets a yolo26m head keep the falls a clip-label
    head misses in an unseen room (research/scripts/picam3_segments.py).

    `load(pose, video_id)` reads a fall clip's keypoints, as in
    `nobody_windows`.
    """
    fall = clips.label.values == 1
    rows = {v: fall_row(pose, v, load) for v in clips.video_id[fall].unique()}
    e = clips.video_id.map(rows)
    end = clips.start + seq_len - 1
    pre = (end < e).values
    holds = ((clips.start <= e) & (e + HALF_S <= end)).values
    post = (clips.start > e + HALF_S).values
    label = np.where(fall & pre, 0, clips.label.values)
    keep = ~fall | pre | holds | post
    return keep, label


def nobody_windows(clips, seq_len: int, copies: int = 10,
                   pose: str = DEPLOYED_POSE, load=None):
    """Synthetic "nobody visible" windows, to be fitted as ADL.

    The head otherwise inherits -- and amplifies -- the room's Dropout
    artifact: in picam's own fall clips the pose backbone loses the subject
    for about half of a `fall forward` window against ~1% of an ADL window, so
    "sixty frames of zeros" is a *fall* cue in the fitting data. A live camera produces that input
    every time the room empties, and the head scored it 0.9999, i.e. a
    self-sustaining false alarm once the rolling mean latches.

    Three flavours, all unambiguously not a fall:

    * all-zero windows (nobody detected at all);
    * sparse-detection windows (two joints found, near the origin -- what a
      backlit or half-occluded frame leaves behind);
    * real ADL windows with the tail 25/50/75% zeroed, which is what a person
      leaving frame or walking behind furniture actually produces. Built from
      `clips`, so they carry this room's own pose statistics.

    `load(pose, video_id)` reads a clip's keypoints (default: the cache on
    disk). Fitted into the training folds only; the held-out subject never sees
    them. Measured effect on the LOSO score of the multi-scale Checkpoint: AUC
    0.9923 -> 0.9909, rolling b4/b8 F1 unchanged at 0.967 (1 FP, 1 FN), while
    P(fall | all-zero window) falls 0.9999 -> 0.0077.
    """
    from .data import prepare_sequence, take_window

    if load is None:
        from .extract import load_keypoints as load

    T_, K = seq_len, cfg.N_KEYPOINTS
    rng = np.random.default_rng(0)
    out = [np.zeros((T_, K, 3), dtype=np.float32)]

    sparse = np.zeros((T_, K, 3), dtype=np.float32)
    lo = 1 + int(0.8 * T_)
    sparse[lo:, [0, 5], 0] = rng.uniform(0.0, 0.08, (T_ - lo, 2))
    sparse[lo:, [0, 5], 1] = rng.uniform(0.0, 0.08, (T_ - lo, 2))
    sparse[lo:, [0, 5], 2] = 0.4
    out.append(sparse)

    adl = clips[clips.label == 0].video_id.drop_duplicates().head(6).tolist()
    for vid in adl:
        seq = prepare_sequence(load(pose, vid), None,
                               cfg.PREPROCESS.frame_stride)
        ref = take_window(seq, T_, start=0)
        for frac in (0.25, 0.50, 0.75):
            cut = int(T_ * (1 - frac))
            w = ref.copy()
            w[cut:] = 0.0
            out.append(w.astype(np.float32))

    windows = []
    for w in out:
        windows.extend([w] * copies)
    return np.stack(windows)


def embed_windows(model, mcfg, windows, device: str = "cpu"):
    """Pooled embeddings for raw (n, T, 17, 3) windows, the Checkpoint frozen."""
    import torch

    from .data import build_features

    x = np.stack([build_features(w, mcfg.features) for w in windows])
    model.to(device).eval()
    with torch.no_grad():
        h = model.encoder(model.embed(torch.from_numpy(x).float().to(device)))
        return model._pool(h).cpu().numpy()


class NobodyNegatives:
    """A fold's nobody-visible windows, embedded, from that fold's own clips.

    Calling it with the clips a fold trains on returns `(embeddings,
    kinematics)` for the windows `nobody_windows` cuts from them (`kinematics`
    is None unless asked for). `embed(windows)` is the frozen Checkpoint's pooled
    embedding. The windows depend only on which ADL clips they are cut from, so
    folds that share those clips share one build.
    """

    def __init__(self, seq_len: int, embed, copies: int = 10,
                 pose: str = DEPLOYED_POSE, kinematics: bool = False,
                 load=None):
        self.seq_len, self.embed, self.copies = seq_len, embed, copies
        self.pose, self.kinematics, self.load = pose, kinematics, load
        self._built = {}

    def __call__(self, clips):
        key = tuple(clips[clips.label == 0].video_id.drop_duplicates().head(6))
        if key not in self._built:
            windows = nobody_windows(clips, self.seq_len, copies=self.copies,
                                     pose=self.pose, load=self.load)
            # A window nobody is visible in has no measurable descent; zeros
            # are the honest value and are what `window_kinematics` returns.
            kin = (np.stack([calibrate.window_kinematics(w) for w in windows])
                   if self.kinematics else None)
            self._built[key] = (self.embed(windows), kin)
            log.info("nobody-visible windows: %d labelled ADL (%d "
                     "flavours x %d copies, from %d ADL clips)", len(windows),
                     len(windows) // self.copies, self.copies, len(key))
        return self._built[key]


def fold_fitting_set(train, clips, emb, y, keep, ext=None, negatives=None):
    """`(embeddings, labels, extras)` a head fitted on the `train` rows sees.

    `train` masks the rows of `clips` (one per window) the fold may learn from;
    `keep` drops the windows the label scheme leaves out. `negatives`, a
    `NobodyNegatives`, adds the synthetic nobody-visible windows, cut from the
    training rows' clips only: a left-out subject's own ADL clips never reach
    the head that scores it.
    """
    real = train & keep
    e, lab = emb[real], y[real]
    x = ext[real] if ext is not None else None
    if negatives is not None:
        syn_emb, syn_ext = negatives(clips[train])
        e = np.concatenate([e, syn_emb])
        lab = np.concatenate([lab, np.zeros(len(syn_emb), dtype=lab.dtype)])
        if x is not None:
            x = np.concatenate([x, syn_ext])
    return e, lab, x
