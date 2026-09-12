"""The demo-cam aspect-ratio rule, ported for offline evaluation.

`demo-cam/v1/detector.py` decides someone has fallen when their person box is
wider than it is tall, debounced by a latch. This module reproduces that
decision exactly, minus the parts that only make sense on a live stream, so the
rule can be scored on the same clips as the learned models.

The whole rule, from `detector.py`:

    ratio = box_width / box_height
    ratio > FALL_RATIO (1.0)         -> a "hit"
    FALL_MIN_FRAMES   (3) hits       -> latch fallen
    FALL_CLEAR_FRAMES (5) misses     -> clear

The asymmetric latch/clear counts are deliberate there and preserved here: a
single count strobes at the boundary, which is the flicker the debounce exists
to kill.

**What is deliberately not reproduced.** demo-cam runs Ultralytics *tracking*
and keeps a latch per track ID, because a live room can hold several people.
These datasets are single-subject, so the cached box is the largest person per
frame and one latch runs over the clip. The wall-clock staleness logic
(`_STALE_SECONDS`, `_FRESH_FRAMES`) is dropped too: it exists to stop a dead
video feed reading as a safe room, which cannot happen against a file.

Neither omission favours the rule or the learned models -- they are stream
plumbing, not the decision.

The rule's own docstring in `detector.py` is candid that it detects **lying
down, not falling**: no velocity term, so a person already on the floor scores
the same as one who just went down, and sitting or crouching fires it. Notebook
11 is the measurement of that honesty.
"""
from __future__ import annotations

import numpy as np

from . import config as cfg

# Defaults copied from demo-cam/v1/detector.py, where they are env-overridable.
FALL_RATIO = 1.0
FALL_MIN_FRAMES = 3
FALL_CLEAR_FRAMES = 5


def aspect_ratios(boxes: np.ndarray) -> np.ndarray:
    """width / height per frame; NaN where nothing was detected.

    `extract.extract_boxes_video` writes an all-zero row for an undetected
    frame. Those become NaN rather than 0.0, because a ratio of zero is a very
    tall box -- a confident "standing" -- and would let a detection failure
    silently clear a latched fall.
    """
    boxes = np.asarray(boxes, dtype=float)
    if boxes.size == 0:
        return np.zeros(0)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(h > 0, w / np.maximum(h, 1e-9), np.nan)
    ratio[(w <= 0) | (h <= 0)] = np.nan
    return ratio


def latch(
    ratios: np.ndarray,
    threshold: float = FALL_RATIO,
    min_frames: int = FALL_MIN_FRAMES,
    clear_frames: int = FALL_CLEAR_FRAMES,
) -> np.ndarray:
    """Run the debounce over a ratio sequence; returns the latch state per frame.

    A NaN frame (no detection) is neither a hit nor a miss: the counters hold.
    That matches `detector.py`, where a frame with no box simply contributes no
    box to score, and the track ages out on `last_seen` rather than being
    counted as evidence of standing.
    """
    ratios = np.asarray(ratios, dtype=float)
    out = np.zeros(len(ratios), dtype=bool)
    hits = misses = 0
    fallen = False

    for i, r in enumerate(ratios):
        if not np.isnan(r):
            if r > threshold:
                hits += 1
                misses = 0
            else:
                misses += 1
                hits = 0
            if not fallen and hits >= min_frames:
                fallen = True
            elif fallen and misses >= clear_frames:
                fallen = False
        out[i] = fallen
    return out


def window_scores(
    boxes: np.ndarray,
    starts,
    seq_len: int = cfg.FINAL_TRAIN.seq_len,
    frame_stride: int = cfg.PREPROCESS.frame_stride,
    threshold: float = FALL_RATIO,
    min_frames: int = FALL_MIN_FRAMES,
    clear_frames: int = FALL_CLEAR_FRAMES,
    full_rate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Score sliding windows of one video: (fired, peak_ratio) per window.

    `starts` are window offsets in the *strided* index space the clip tables
    use, so a window covers original frames
    `[start * frame_stride, (start + seq_len) * frame_stride)`.

    `full_rate=True` runs the rule over every original frame in that span, which
    is what demo-cam does -- the rule is cheap enough to see every frame, and the
    latch counts frames, so subsampling would change what `FALL_MIN_FRAMES`
    means. Setting it False restricts the rule to exactly the frames the
    transformer sees, which is the stricter like-for-like comparison; notebook 11
    reports both.

    `fired` is True when the latch is ever set inside the window -- the same
    early-exit decision Algorithm 2 makes for the learned model, so the two are
    compared under one rule for turning a sequence into a verdict.

    `peak_ratio` is the largest observed ratio, a continuous score that makes an
    ROC possible. Without it the rule has a single operating point and cannot be
    compared against a model that has a whole curve.
    """
    ratios_full = aspect_ratios(boxes)
    fired = np.zeros(len(starts), dtype=bool)
    peak = np.zeros(len(starts), dtype=float)

    for i, s in enumerate(starts):
        s = int(s)
        if full_rate:
            lo, hi = s * frame_stride, (s + seq_len) * frame_stride
            seg = ratios_full[lo:hi]
        else:
            seg = ratios_full[::frame_stride][s:s + seq_len]

        if seg.size == 0 or np.all(np.isnan(seg)):
            peak[i] = 0.0
            continue
        fired[i] = bool(latch(seg, threshold, min_frames, clear_frames).any())
        peak[i] = float(np.nanmax(seg))
    return fired, peak


def score_clips(
    clips,
    scale: str = cfg.DEFAULT_SCALE,
    seq_len: int = cfg.FINAL_TRAIN.seq_len,
    frame_stride: int = cfg.PREPROCESS.frame_stride,
    threshold: float = FALL_RATIO,
    min_frames: int = FALL_MIN_FRAMES,
    clear_frames: int = FALL_CLEAR_FRAMES,
    full_rate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the rule to a clips table, aligned to its row order.

    Mirrors `evaluate.score_clips` so the rule and the learned models can be
    dropped into the same reporting code. Returns (fired, peak_ratio).
    """
    from .extract import load_boxes

    fired = np.zeros(len(clips), dtype=bool)
    peak = np.zeros(len(clips), dtype=float)

    for vid, group in clips.groupby("video_id"):
        boxes = load_boxes(scale, vid)
        f, p = window_scores(boxes, group.start.to_numpy(), seq_len=seq_len,
                             frame_stride=frame_stride, threshold=threshold,
                             min_frames=min_frames, clear_frames=clear_frames,
                             full_rate=full_rate)
        idx = clips.index.get_indexer(group.index)
        fired[idx] = f
        peak[idx] = p
    return fired, peak
