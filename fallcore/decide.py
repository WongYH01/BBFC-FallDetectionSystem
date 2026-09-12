"""Turning per-window scores into a clip verdict.

Every other notebook makes a clip Fall the way Algorithm 2 does: as soon as one
window clears the threshold. That is the decision of the per-clip maximum, so a
single noisy window decides the clip. On the deployment camera that rule is the
largest source of ADL false alarms -- several clips contain one or two high
windows and nothing else -- and it is also the one part of the pipeline that can
be changed without retraining.

Two levers, usable separately or together:

* an **aggregate** over the clip's windows (mean, median, top-3 mean, ...),
  which makes the verdict a property of the whole clip rather than its noisiest
  moment;
* **persistence** -- `min_consecutive` adjacent over-threshold windows -- which
  is also what a live monitor needs: a latch that only engages on sustained
  evidence and then stays engaged.

`clip_scores` computes every quantity the rules need for a window table in one
pass; `decide`, `fire_start`, `rolling_alarm` and `rule_metrics` turn those into
clip verdicts, alarm times and confusion counts. Nothing here trains or scores a
model.

Honesty about the aggregates: they are not causal. A mean over the whole clip
uses windows that had not happened yet at the moment an alarm would have been
raised, so an aggregate rule labels finished clips; it is not something a live
stream can evaluate until the clip is over. `fire_start` exists for the causal
comparison: persistence is causal by construction, and its latency against the
peak rule is directly measurable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

#: Aggregate columns `clip_scores` computes and `decide` accepts.
AGGREGATES = ("peak", "second", "mean", "median", "top3_mean")


def max_consecutive(over) -> int:
    """Longest run of truth in a boolean sequence; 0 when empty or all-False."""
    best = run = 0
    for v in np.asarray(over, dtype=bool):
        if v:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def first_true(over) -> int | None:
    """Index of the first True, or None. `argmax` alone cannot say "never"."""
    idx = np.flatnonzero(np.asarray(over, dtype=bool))
    return int(idx[0]) if idx.size else None


def clip_scores(
    windows: pd.DataFrame,
    score: str = "prob",
    threshold: float = cfg.EVAL.threshold,
) -> pd.DataFrame:
    """One row per clip: aggregates, persistence and the first-fire index.

    `windows` needs `video_id`, `start` and a numeric `score` column -- the
    tables produced by `evaluate.score_clips`, `picam.build_clips` and the
    cross-dataset adapters all have that shape. `label`, `action` and `subject`
    are carried through when present, so a caller can score without re-merging.

    Windows are read in `start` order for the run statistics; the aggregates are
    order-free. `first_fire_start` is the `start` of the first over-threshold
    window (`NaN` when none), i.e. Algorithm 2's early exit in the strided index
    space the clip tables use; the alarm itself is raised at that window's end.
    """
    if score not in windows.columns:
        raise KeyError(f"score {score!r} is not a column of the window table")

    rows = []
    for vid, d in windows.groupby("video_id", sort=False):
        d = d.sort_values("start")
        p = d[score].to_numpy(dtype=float)
        over = p >= threshold
        sorted_p = np.sort(p) if p.size else np.zeros(1)
        first = first_true(over)

        row = {
            "video_id": vid,
            "n_windows": int(p.size),
            "peak": float(sorted_p[-1]),
            "second": float(sorted_p[-2]) if p.size > 1 else float(sorted_p[-1]),
            "mean": float(p.mean()) if p.size else 0.0,
            "median": float(np.median(p)) if p.size else 0.0,
            "top3_mean": float(sorted_p[-3:].mean()) if p.size else 0.0,
            "n_over": int(over.sum()),
            "max_run": max_consecutive(over),
            "first_fire_start": (float(d["start"].iloc[first])
                                 if first is not None else np.nan),
        }
        for name in ("label", "action", "subject"):
            if name in d.columns:
                row[name] = d[name].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows).set_index("video_id")


def decide(
    clips: pd.DataFrame,
    threshold: float = cfg.EVAL.threshold,
    aggregate: str = "mean",
    min_consecutive: int = 1,
) -> pd.Series:
    """Clip verdict under one rule, as a boolean Series indexed like `clips`.

    A clip fires when `clips[aggregate]` clears `threshold` *and* its longest
    run of over-threshold windows is at least `min_consecutive`. The rule the
    existing notebooks use is exactly `aggregate="peak", min_consecutive=1`,
    which reproduces Algorithm 2's early exit -- so this is a strict
    generalisation of the reported protocol, not a different one.

    `clips` must come from `clip_scores(..., threshold=...)` with the same
    threshold: `max_run` was counted at that threshold and is not recomputed.
    """
    if aggregate not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}, not {aggregate!r}")
    return (clips[aggregate] >= threshold) & (clips["max_run"] >= min_consecutive)


def fire_start(
    windows: pd.DataFrame,
    score: str = "prob",
    threshold: float = cfg.EVAL.threshold,
    min_consecutive: int = 1,
) -> pd.Series:
    """First window `start` at which a causal rule engages, per clip.

    The causal twin of `decide`: walks the clip in `start` order and returns the
    `start` of the window that completes a run of `min_consecutive` windows over
    `threshold`, `NaN` when the clip never gets there. `min_consecutive=1` is
    Algorithm 2's early exit; larger values price the persistence rule's latency
    directly. The alarm would be raised at that window's end, not its start.
    """
    out = {}
    for vid, d in windows.groupby("video_id", sort=False):
        d = d.sort_values("start")
        run, hit = 0, np.nan
        over = d[score].to_numpy(dtype=float) >= threshold
        for start, value in zip(d["start"].to_numpy(), over):
            run = run + 1 if value else 0
            if run >= min_consecutive:
                hit = float(start)
                break
        out[vid] = hit
    return pd.Series(out, name="fire_start")


def _rolling_mean(values: np.ndarray, buffer: int) -> np.ndarray:
    """Mean of the last `buffer` values at each step, fewer while it fills."""
    out = np.empty(len(values), dtype=float)
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= buffer:
            total -= values[i - buffer]
        out[i] = total / min(i + 1, buffer)
    return out


def rolling_scores(
    windows: pd.DataFrame,
    score: str = "prob",
    buffer: int = 4,
) -> pd.Series:
    """Causal rolling mean of a window table's scores, aligned to its rows.

    The score at each window is the mean of the last `buffer` window scores in
    `start` order -- the quantity a live monitor can compute the moment that
    window arrives. `buffer=1` is the raw score; while the buffer fills at the
    start of a recording, the mean is over the windows available so far.

    This is the causal counterpart of `clip_scores`' `mean` column: same idea,
    no future windows.
    """
    if buffer < 1:
        raise ValueError(f"buffer must be >= 1, not {buffer}")
    if score not in windows.columns:
        raise KeyError(f"score {score!r} is not a column of the window table")

    out = np.empty(len(windows), dtype=float)
    for _, d in windows.groupby("video_id", sort=False):
        d = d.sort_values("start")
        out[windows.index.get_indexer(d.index)] = _rolling_mean(
            d[score].to_numpy(dtype=float), buffer)
    return pd.Series(out, index=windows.index)


def rolling_alarm(
    windows: pd.DataFrame,
    score: str = "prob",
    buffer: int = 4,
    threshold: float = cfg.EVAL.threshold,
) -> pd.DataFrame:
    """Replay a window table as a live stream; one row per recording.

    The alarm engages at the first window whose rolling mean clears `threshold`
    and, as in a real monitor, stays engaged once it has: `fired` is "ever
    engaged", `fire_start` is that window's strided start (`NaN` when never) and
    the alarm would be raised at its end. `rolling_peak` is the largest rolling
    score, which is what a threshold sweep ranks on.

    `decide` reads all windows at once; this only ever reads the past.
    """
    rows = []
    for vid, d in windows.groupby("video_id", sort=False):
        d = d.sort_values("start")
        v = _rolling_mean(d[score].to_numpy(dtype=float), buffer)
        first = first_true(v >= threshold)
        rows.append({
            "video_id": vid,
            "fired": first is not None,
            "fire_start": (float(d["start"].iloc[first])
                           if first is not None else np.nan),
            "rolling_peak": float(v.max()) if v.size else 0.0,
            "n_windows": int(v.size),
        })
    return pd.DataFrame(rows).set_index("video_id")


def rule_metrics(labels, fired) -> dict:
    """Confusion counts and P/R/F1 for a boolean decision.

    `evaluate.binary_metrics` scores a probability and reports an ROC; a rule
    output is already a decision at one operating point, so it has no ranking
    left to measure and needs its own count. Names match `binary_metrics` so a
    table of the two can be read side by side at the same threshold.
    """
    labels = np.asarray(labels, dtype=int)
    fired = np.asarray(fired, dtype=bool)
    tp = int((fired & (labels == 1)).sum())
    fp = int((fired & (labels == 0)).sum())
    fn = int((~fired & (labels == 1)).sum())
    tn = int((~fired & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {
        "accuracy": (tp + tn) / len(labels) if len(labels) else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
