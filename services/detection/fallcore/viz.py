"""Plots the notebooks share: confusion matrices, curves, skeletons, sweeps.

Every function takes an optional `ax` so figures can be composed, and returns
the axes. Saving is the caller's business except for `save`, which is here only
to keep the figures directory consistent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as cfg


def save(fig, name: str, dpi: int = 150) -> Path:
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_confusion(result: dict, ax=None, title: str = "", cmap: str = "Blues"):
    """2x2 confusion matrix from a metrics dict, annotated with counts."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3.6))
    cm = np.array([[result["tn"], result["fp"]],
                   [result["fn"], result["tp"]]])
    ax.imshow(cm, cmap=cmap)
    for (i, j), v in np.ndenumerate(cm):
        # Flip the label colour on the dark cells so it stays legible.
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13,
                color="white" if v > cm.max() * 0.6 else "black")
    ax.set_xticks([0, 1], ["No-Fall", "Fall"])
    ax.set_yticks([0, 1], ["No-Fall", "Fall"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title or f"acc {result['accuracy']:.2%}  F1 {result['f1']:.2%}")
    return ax


def plot_training(history: list[dict], best_epoch: int | None = None, axes=None):
    """Loss, F1 and accuracy against epoch -- the paper's Figure 6."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    epochs = [h["epoch"] for h in history]

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("loss")
    axes[0].legend()

    axes[1].plot(epochs, [h["val_f1"] for h in history], color="tab:green")
    axes[1].set_title("validation F1")

    axes[2].plot(epochs, [h["val_accuracy"] for h in history], color="tab:orange")
    axes[2].set_title("validation accuracy")

    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if best_epoch is not None:
            ax.axvline(best_epoch, ls="--", color="green", alpha=0.7)
    return axes


def plot_size_sweep(df, ax=None, x: str = "pose_params", y: str = "f1_mean",
                    yerr: str | None = "f1_std", annotate: bool = True):
    """Downstream F1 against pose-model cost, with across-seed error bars.

    The error bars are the point of this plot. A gap between two scales that is
    smaller than the spread across seeds is not evidence that the bigger
    backbone is better.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.2))
    xs = df[x].to_numpy(dtype=float)
    ys = df[y].to_numpy(dtype=float)
    errs = df[yerr].to_numpy(dtype=float) if yerr and yerr in df else None

    ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=4, lw=1.6, ms=8)
    if annotate:
        for _, row in df.iterrows():
            ax.annotate(row["scale"], (float(row[x]), float(row[y])),
                        textcoords="offset points", xytext=(8, -4), fontsize=9)
    ax.set_xlabel({"pose_params": "pose backbone parameters",
                   "pose_gflops": "pose backbone GFLOPs",
                   "pose_fps": "pose throughput (FPS)"}.get(x, x))
    ax.set_ylabel("downstream F1 (fall class)")
    ax.grid(alpha=0.3)
    return ax


def plot_sensitivity(df, ax=None, top: int | None = None,
                     label_col: str | None = None, value_col: str = "delta_f1"):
    """Horizontal bars of delta-F1 per ablated parameter -- the paper's Table 8.

    The label column is auto-detected ("dimension" or "parameter") so this works
    with either naming; pass `label_col` to be explicit.
    """
    import matplotlib.pyplot as plt

    if label_col is None:
        for candidate in ("dimension", "parameter", "name"):
            if candidate in df.columns:
                label_col = candidate
                break
        else:
            raise KeyError(
                f"no label column found in {list(df.columns)}; "
                f"pass label_col= explicitly"
            )

    d = df.sort_values(value_col)
    if top:
        d = d.tail(top)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.42 * len(d) + 1.2))
    ax.barh(d[label_col].astype(str), d[value_col] * 100, color="tab:blue")
    ax.set_xlabel("F1 range across the swept values (percentage points)")
    ax.grid(axis="x", alpha=0.3)
    return ax


def draw_skeleton(ax, keypoints_yxc: np.ndarray, width: int, height: int,
                  conf_threshold: float = 0.3, color: str = "lime"):
    """Draw one cached (17, 3) frame in (y, x, conf) order onto pixel axes.

    This is the check that the y/x ordering in the cache is right: pass a real
    frame as the axes background and the skeleton should land on the person.
    If it lands transposed, extract.py has the channels backwards.
    """
    kp = np.asarray(keypoints_yxc)
    ys = kp[:, 0] * height
    xs = kp[:, 1] * width
    conf = kp[:, 2]

    for a, b in cfg.SKELETON_EDGES:
        if conf[a] >= conf_threshold and conf[b] >= conf_threshold:
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], "-", color=color, lw=2, alpha=0.9)
    visible = conf >= conf_threshold
    ax.scatter(xs[visible], ys[visible], s=22, c="red", zorder=3)
    return ax


def plot_hip_trajectories(traces: dict, axes=None, joint_name: str = "right hip",
                          band: tuple[float, float] = (25, 75),
                          individual: dict | None = None, n_individual: int = 14):
    """Individual trajectories and their per-class median band, side by side.

    `traces` maps a class name to a list of resampled trajectories on a shared
    normalised-time axis, and drives the median band on the right -- the
    consolidated view the paper's Figure 3 presents.

    `individual` optionally supplies gap-preserving versions (resampled with a
    `max_gap`) for the left panel, so a stretch where the pose model lost the
    subject draws as a break rather than as a straight line. Falls back to
    `traces` when not given.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    colours = {"Fall": "tab:orange", "No-Fall": "tab:blue"}
    individual = individual or traces
    for name, series in traces.items():
        colour = colours.get(name, None)
        arr = np.asarray(series, dtype=float)
        t = np.linspace(0, 1, arr.shape[1])

        shown = np.asarray(individual.get(name, series), dtype=float)
        t_shown = np.linspace(0, 1, shown.shape[1])
        for row in shown[:n_individual]:
            axes[0].plot(t_shown, row, color=colour, alpha=0.4, lw=1.1)
        axes[0].plot([], [], color=colour, label=name)

        # nan-aware, because a clip can be entirely undetected in places.
        with np.errstate(all="ignore"):
            median = np.nanmedian(arr, axis=0)
            lo = np.nanpercentile(arr, band[0], axis=0)
            hi = np.nanpercentile(arr, band[1], axis=0)
        axes[1].plot(t, median, color=colour, lw=2.4, label=f"{name} (median)")
        axes[1].fill_between(t, lo, hi, color=colour, alpha=0.2)

    for ax, title in zip(axes, [f"individual clips ({joint_name})",
                                f"median with {band[0]:.0f}-{band[1]:.0f}% band"]):
        ax.set_xlabel("normalised time")
        ax.set_ylabel(f"{joint_name} y  (0 = top of frame)")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        ax.legend()
    return axes


def plot_keypoint_importance(scores: np.ndarray, ax=None):
    """Gradient-saliency per keypoint as a labelled bar chart (Figure 8)."""
    import matplotlib.pyplot as plt

    order = np.argsort(scores)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6))
    ax.barh([cfg.KEYPOINT_NAMES[i] for i in order], scores[order], color="tab:purple")
    ax.set_xlabel("mean |gradient| (normalised)")
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_skeleton_importance(scores: np.ndarray, ax=None, cmap: str = "viridis"):
    """The same saliency mapped onto a schematic body, as in the paper."""
    import matplotlib.pyplot as plt

    # A neutral standing pose in normalised (x, y); only the layout matters,
    # the colour carries the information.
    layout = np.array([
        [0.50, 0.06], [0.47, 0.04], [0.53, 0.04], [0.44, 0.05], [0.56, 0.05],
        [0.40, 0.18], [0.60, 0.18], [0.35, 0.33], [0.65, 0.33],
        [0.32, 0.47], [0.68, 0.47], [0.44, 0.50], [0.56, 0.50],
        [0.43, 0.72], [0.57, 0.72], [0.42, 0.93], [0.58, 0.93],
    ])
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 6.5))

    for a, b in cfg.SKELETON_EDGES:
        ax.plot(layout[[a, b], 0], layout[[a, b], 1], "-", color="lightgray", lw=2, zorder=1)
    sc = ax.scatter(layout[:, 0], layout[:, 1], c=scores, s=260, cmap=cmap,
                    zorder=2, edgecolors="black", linewidths=0.6)
    for i, name in enumerate(cfg.KEYPOINT_NAMES):
        ax.annotate(name, layout[i], textcoords="offset points", xytext=(11, 0),
                    fontsize=7.5, va="center")
    ax.invert_yaxis()
    ax.set_xlim(0.15, 0.95)
    ax.axis("off")
    ax.figure.colorbar(sc, ax=ax, label="importance", fraction=0.04)
    return ax


def plot_attention(frame_scores: np.ndarray, ax=None):
    """Mean attention received per frame position (Figure 9)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(np.arange(len(frame_scores)), frame_scores, color="tab:blue")
    peak = int(np.argmax(frame_scores))
    ax.axvline(peak, color="red", ls="--", alpha=0.8, label=f"peak @ frame {peak}")
    ax.set_xlabel("frame position in window")
    ax.set_ylabel("mean attention received")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return ax
