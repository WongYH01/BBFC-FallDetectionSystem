"""The Fall Decision demo-cam v2 deploys, defined once.

A Fall Decision is the calibrated pair (one checkpoint plus the room head fitted
on its embedding), the rule that turns its window scores into an alarm (rolling
buffer, threshold, hysteresis, descent gate) and the sampling the windows are
cut with. The live detector, `replay.py` and picam scoring all build their
`EnsembleStream` from one of these, so the rule a replay or a rendered clip
scores is the rule the camera runs -- not a copy kept in step by hand.

Deliberately light: importing this module pulls in only `fallcore.config`, so a
test or a CLI can read and compare decisions without torch. `build` imports the
stream, and with it the model code, when a stream is actually wanted.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
import dataclasses
from dataclasses import dataclass
from pathlib import Path

from . import config as cfg

#: The calibrated pair the camera ships with. `scripts/fetch_weights.py` pins
#: these two files by hash; a test keeps the two lists in agreement.
DEPLOYED_CHECKPOINT = "augnone_ms_coords_hn_s99.pt"
DEPLOYED_HEAD = "probe_augnone_ms_coords_hn_s99.npz"

#: The environment variables demo-cam v2 has always read, and the field each
#: one sets. Their names are a contract with every `.env` and compose file.
_RULE_VARS: dict[str, tuple[str, type]] = {
    "BUFFER_LEN":          ("buffer", int),
    "FALL_THRESHOLD":      ("threshold", float),
    "FALL_CLEAR_BELOW":    ("clear_below", float),
    "FALL_CLEAR_WINDOWS":  ("clear_windows", int),
    "FALL_MIN_DESCENT":    ("min_descent", float),
    "STREAM_FRAME_STRIDE": ("frame_stride", int),
}
_RULE_FIELDS = ("buffer", "threshold", "clear_below", "clear_windows",
                "min_descent", "frame_stride", "step")


def _out_of_range(field: str, value) -> str | None:
    """Why `value` cannot be `field`, or None when it can."""
    if field in ("buffer", "clear_windows", "frame_stride", "step") and value < 1:
        return "must be at least 1"
    if field in ("threshold", "clear_below") and not 0.0 <= value <= 1.0:
        return "must be a probability between 0 and 1"
    if field == "min_descent" and value < 0.0:
        return "must be 0 (gate off) or more"
    return None


@dataclass(frozen=True)
class FallDecision:
    checkpoints: tuple[Path, ...] = (cfg.CKPT_DIR / DEPLOYED_CHECKPOINT,)
    heads: tuple[Path | None, ...] = (cfg.CKPT_DIR / DEPLOYED_HEAD,)
    buffer: int = 4
    # The camera's own setting, not the paper's Algorithm 2 threshold
    # (`config.EVAL`), though both are 0.5 today: retuning the replication
    # must not retune the deployment.
    threshold: float = 0.5
    clear_below: float = 0.2
    clear_windows: int = 2
    min_descent: float = 0.10
    frame_stride: int = cfg.PREPROCESS.frame_stride
    step: int = cfg.EVAL.sliding_stride

    def __post_init__(self):
        for field in _RULE_FIELDS:
            problem = _out_of_range(field, getattr(self, field))
            if problem:
                raise ValueError(f"{field} {problem}, not {getattr(self, field)!r}")
        if len(self.heads) != len(self.checkpoints):
            raise ValueError(f"{len(self.heads)} heads for "
                             f"{len(self.checkpoints)} checkpoints -- one head "
                             f"(or None) per checkpoint")

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> FallDecision:
        """The deployed decision with whatever `env` overrides applied.

        A value that does not parse or is out of range is refused with the
        variable's name, so a typo in `.env` stops the detector at startup
        rather than booting it mis-tuned.
        """
        changes = {}
        for var, (field, kind) in _RULE_VARS.items():
            if var not in env:
                continue
            try:
                value = kind(env[var])
            except ValueError:
                raise ValueError(f"{var}={env[var]!r} is not "
                                 f"{'an integer' if kind is int else 'a number'}"
                                 ) from None
            problem = _out_of_range(field, value)
            if problem:
                raise ValueError(f"{var} {problem}, not {env[var]!r}")
            changes[field] = value
        checkpoints = tuple(Path(p) for p in
                            env.get("ENSEMBLE_CKPTS", "").split(os.pathsep) if p)
        if checkpoints:
            changes["checkpoints"] = checkpoints
        if "CALIBRATION_HEAD" in env:
            # Empty runs every checkpoint on its own classifier.
            head = env["CALIBRATION_HEAD"].strip()
            changes["heads"] = ((Path(head),) if head else
                                (None,) * len(checkpoints or cls.checkpoints))
        n_ckpts = len(changes.get("checkpoints", cls.checkpoints))
        n_heads = len(changes.get("heads", cls.heads))
        if n_heads != n_ckpts:
            # A head is fitted on one backbone, so it cannot follow a swap to
            # several checkpoints on its own.
            raise ValueError(f"ENSEMBLE_CKPTS names {n_ckpts} checkpoints but "
                             f"CALIBRATION_HEAD gives {n_heads} head(s); set "
                             f"CALIBRATION_HEAD= (empty) to run them uncalibrated")
        return cls(**changes)

    def replace(self, **changes) -> FallDecision:
        """This decision with the given fields changed."""
        return dataclasses.replace(self, **changes)

    def build(self, device: str = "cpu"):
        """An `EnsembleStream` that runs this decision.

        The window length is not a setting: it is read from the checkpoints,
        which must agree, so a checkpoint trained on a different window cannot
        be fed one cut to the training default. Each head is checked against
        its checkpoint's backbone, features and window length before use.
        """
        from .calibrate import ProbeHead, validate
        from .stream import EnsembleStream
        from .train import load_checkpoint

        lengths = set()
        for path, head in zip(self.checkpoints, self.heads):
            _model, model_cfg, train_cfg, _scale = load_checkpoint(path, device="cpu")
            lengths.add(train_cfg.seq_len)
            if head is not None:
                validate(ProbeHead.load(head), path, model_cfg.features,
                         train_cfg.seq_len)
        if len(lengths) > 1:
            raise ValueError(f"checkpoints disagree on window length: "
                             f"{sorted(lengths)}")
        return EnsembleStream(
            list(self.checkpoints), heads=list(self.heads), buffer=self.buffer,
            threshold=self.threshold, clear_below=self.clear_below,
            clear_windows=self.clear_windows, min_descent=self.min_descent,
            seq_len=lengths.pop(), frame_stride=self.frame_stride,
            step=self.step, device=device)

    def __str__(self) -> str:
        models = ", ".join(
            f"{c.name} + {h.name}" if h is not None else f"{c.name} (own classifier)"
            for c, h in zip(self.checkpoints, self.heads))
        rule = " ".join(f"{f}={getattr(self, f)}" for f in _RULE_FIELDS)
        return f"{models} | {rule}"
