"""Training loop: AdamW, BCE-with-logits, early stopping on validation F1.

Table 7 and Algorithm 1. Deliberately plain -- no scheduler, no AMP, no grad
clipping, because the paper uses none of those and adding them would make our
numbers incomparable to theirs. The model is small enough that it does not need
them.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config as cfg
from .data import feature_dim, make_loaders
from .model import build_model, count_parameters


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG the training path touches, and pin cuDNN.

    Seeding the RNGs alone is not enough on CUDA. cuDNN picks convolution
    algorithms by benchmarking them, and several of its convolution backward
    kernels accumulate in a non-deterministic order, so the 1D-CNN baseline in
    notebook 05 moved by 0.34 F1 points across three runs at an identical seed
    while the transformer, LSTM and MLP reproduced exactly. A comparison
    against a model that will not sit still is not a comparison.

    `deterministic=False` restores the faster benchmarking path for runs where
    throughput matters more than reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        # Needed by the deterministic CUDA reduction kernels; harmless elsewhere.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only, not strict: PyTorch's memory-efficient attention backward
        # has no deterministic implementation, so strict mode would raise and
        # the transformer could not train at all. It warns instead. Measured
        # consequence: across two independent five-seed sweeps the transformer
        # reproduced to four decimals on every seed, so the residual
        # non-determinism does not reach the predictions here -- but it is not
        # a guarantee. For a strict run, disable the fused attention kernels
        # (torch.backends.cuda.enable_mem_efficient_sdp(False) and
        # enable_flash_sdp(False)); that changes which kernel runs, so it also
        # changes the numbers, and every transformer result in this project was
        # produced with them enabled.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older builds lack warn_only; skip rather than raise on attention.
            pass


class FocalLoss(nn.Module):
    """Binary focal loss, ablated against BCE in Section IV-A-4 (Table 20).

    Computed from logits via the numerically stable BCE path rather than from
    an explicit sigmoid, so it does not lose the property that made the paper
    prefer BCEWithLogits in the first place.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t).pow(self.gamma) * bce).mean()


def make_loss(train_cfg: cfg.TrainConfig,
              class_counts: "np.ndarray | None" = None) -> nn.Module:
    if train_cfg.loss == "bce":
        return nn.BCEWithLogitsLoss()
    if train_cfg.loss == "focal":
        return FocalLoss(train_cfg.focal_gamma, train_cfg.focal_alpha)
    if train_cfg.loss == "ce":
        weight = None
        if train_cfg.class_weight == "balanced":
            if class_counts is None:
                raise ValueError("class_weight='balanced' needs class_counts")
            counts = np.asarray(class_counts, dtype=np.float64)
            # Inverse frequency, normalised to mean 1 so the loss stays on the
            # same scale as the unweighted case and the learning rate carries
            # over. Empty classes get zero weight rather than an infinity.
            with np.errstate(divide="ignore"):
                w = np.where(counts > 0, counts.sum() / (len(counts) * counts), 0.0)
            w = w / w[w > 0].mean()
            weight = torch.tensor(w, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"unknown loss {train_cfg.loss!r}")


@torch.no_grad()
def _epoch_eval(model, loader, criterion, device, threshold: float,
                fall_ids: "tuple[int, ...] | None" = None):
    """Loss and the four headline metrics over one loader.

    Multi-class runs are still scored as a *fall detector*: the alarm
    probability is the summed softmax mass on `fall_ids`, and the reported
    metrics are the same binary ones the single-logit models produce. That is
    deliberate -- the whole point of the multi-class variant is to compare it
    against the binary models on identical terms, which a class-accuracy number
    would not do. Per-class accuracy is reported alongside, not instead.
    """
    from .evaluate import binary_metrics

    model.eval()
    multiclass = fall_ids is not None
    total_loss, n = 0.0, 0
    probs, targets, cls_hit = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        n += len(y)
        if multiclass:
            soft = torch.softmax(logits, dim=-1)
            probs.append(soft[:, list(fall_ids)].sum(dim=-1).cpu().numpy())
            targets.append(torch.isin(
                y, torch.tensor(fall_ids, device=y.device)).float().cpu().numpy())
            cls_hit.append((logits.argmax(-1) == y).float().cpu().numpy())
        else:
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y.cpu().numpy())

    probs = np.concatenate(probs) if probs else np.zeros(0)
    targets = np.concatenate(targets) if targets else np.zeros(0)
    metrics = binary_metrics(targets, probs, threshold)
    metrics["loss"] = total_loss / max(n, 1)
    if multiclass and cls_hit:
        metrics["class_accuracy"] = float(np.concatenate(cls_hit).mean())
    return metrics


def train_model(
    splits: dict[str, list[str]],
    labels: dict[str, int],
    model_cfg: cfg.ModelConfig | None = None,
    train_cfg: cfg.TrainConfig | None = None,
    scale: str = cfg.DEFAULT_SCALE,
    run_name: str | None = None,
    save_checkpoint: bool = True,
    verbose: bool = True,
    progress: bool = False,
    loaders: dict | None = None,
    monitor: dict | None = None,
    aspects: dict[str, float] | None = None,
) -> dict:
    """Train one model to convergence and return its history plus best metrics.

    Early stopping tracks validation F1, not loss: F1 is the paper's primary
    indicator, and for a safety-critical binary problem a model can improve its
    loss while getting worse at the thing it exists to do.

    `monitor` is an optional `{name: loader}` map evaluated every epoch and
    recorded in `history` as `<name>_<metric>`. It never touches selection --
    that is the whole point. Notebook 14 uses it to plot the out-of-distribution
    curve against the validation curve it is actually stopping on, which is a
    question that cannot be asked without measuring a set you must not select
    on. Costs one extra forward pass per epoch per loader.
    """
    model_cfg = model_cfg or cfg.FINAL_MODEL
    train_cfg = train_cfg or cfg.FINAL_TRAIN
    run_name = run_name or f"{scale}_d{model_cfg.d_model}_seed{train_cfg.seed}"

    device = train_cfg.device if torch.cuda.is_available() else "cpu"
    set_seed(train_cfg.seed)

    # A densely labelled corpus supplies its own window-level loaders, because
    # its unit of supervision is the window rather than the video; see
    # `data.make_clip_loaders`. Everything after this point is identical, which
    # is the point -- the training loop should not know which corpus it is on.
    if loaders is None:
        loaders = make_loaders(
            splits, labels, scale=scale, seq_len=train_cfg.seq_len,
            batch_size=train_cfg.batch_size, features=model_cfg.features,
            seed=train_cfg.seed, num_workers=train_cfg.num_workers,
            fill_gaps=train_cfg.fill_gaps, frame_dropout=train_cfg.frame_dropout,
            aspects=aspects,
        )

    model = build_model(model_cfg, input_dim=feature_dim(model_cfg.features)).to(device)

    # Multi-class scoring needs to know which output indices mean "alarm".
    # Output index i is OmniFall's label id i, so 1 is `fall` and 2 is `fallen`
    # -- the same pair the binary label map collapses into positive.
    fall_ids = (1, 2) if model_cfg.n_classes > 1 else None

    counts = None
    if train_cfg.loss == "ce" and train_cfg.class_weight == "balanced":
        ds = loaders["train"].dataset
        counts = np.bincount(np.asarray(ds.targets, dtype=np.int64),
                             minlength=model_cfg.n_classes)
    # .to(device) matters here: CrossEntropyLoss keeps its class weights as a
    # buffer, and a CPU weight against CUDA logits is a runtime error.
    criterion = make_loss(train_cfg, counts).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    history: list[dict] = []
    best = {"val_f1": -1.0, "epoch": -1}
    best_state = None
    epochs_without_improvement = 0
    t0 = time.perf_counter()

    epoch_iter = range(1, train_cfg.max_epochs + 1)
    if progress:
        from tqdm.auto import tqdm
        epoch_iter = tqdm(epoch_iter, desc=run_name, unit="ep")

    for epoch in epoch_iter:
        model.train()
        running, seen = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            seen += len(y)

        train_loss = running / max(seen, 1)
        val = _epoch_eval(model, loaders["val"], criterion, device,
                          cfg.EVAL.threshold, fall_ids)
        row = {"epoch": epoch, "train_loss": train_loss,
               **{f"val_{k}": v for k, v in val.items()}}
        for name, loader in (monitor or {}).items():
            watched = _epoch_eval(model, loader, criterion, device,
                                  cfg.EVAL.threshold, fall_ids)
            row.update({f"{name}_{k}": v for k, v in watched.items()})
        history.append(row)

        if val["f1"] > best["val_f1"]:
            best = {"val_f1": val["f1"], "epoch": epoch,
                    **{f"val_{k}": v for k, v in val.items()}}
            # Detach to CPU so the kept copy does not pin GPU memory for the
            # rest of training -- it matters when a sweep runs 54 of these.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  epoch {epoch:>3}  train_loss {train_loss:.4f}  "
                  f"val_loss {val['loss']:.4f}  val_f1 {val['f1']:.4f}  "
                  f"val_acc {val['accuracy']:.4f}")

        if epochs_without_improvement >= train_cfg.patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} "
                      f"(no val F1 improvement for {train_cfg.patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test = _epoch_eval(model, loaders["test"], criterion, device,
                       cfg.EVAL.threshold, fall_ids)
    elapsed = time.perf_counter() - t0

    result = {
        "run_name": run_name,
        "scale": scale,
        "model_config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "n_params": count_parameters(model),
        "best_epoch": best["epoch"],
        "best_val_f1": best["val_f1"],
        "test": test,
        "history": history,
        "epochs_run": len(history),
        "train_seconds": elapsed,
        "device": device,
    }

    if save_checkpoint:
        cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt = cfg.CKPT_DIR / f"{run_name}.pt"
        torch.save({"state_dict": model.state_dict(),
                    "model_config": model_cfg.to_dict(),
                    "train_config": train_cfg.to_dict(),
                    "scale": scale}, ckpt)
        result["checkpoint"] = str(ckpt)

    if verbose:
        print(f"  done: best val F1 {best['val_f1']:.4f} @ epoch {best['epoch']}, "
              f"test F1 {test['f1']:.4f}, {elapsed:.0f}s")

    return result


def load_checkpoint(path: str | Path, device: str = "cuda"):
    """Rebuild a model from a checkpoint written by train_model."""
    device = device if torch.cuda.is_available() else "cpu"
    blob = torch.load(path, map_location=device, weights_only=False)
    model_cfg = cfg.ModelConfig(**blob["model_config"])
    model = build_model(model_cfg, input_dim=feature_dim(model_cfg.features)).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, model_cfg, cfg.TrainConfig(**blob["train_config"]), blob["scale"]


def save_result(result: dict, name: str | None = None) -> Path:
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.METRICS_DIR / f"{name or result['run_name']}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def with_overrides(base: cfg.ModelConfig | cfg.TrainConfig, **kwargs):
    """Copy a config with fields replaced -- the ablation sweep's workhorse."""
    return replace(base, **kwargs)
