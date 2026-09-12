"""Online fall decisions: a lived-in keypoint buffer plus a rolling ensemble.

`fallcore.infer` scores a finished file and `fallcore.decide` labels a table of
windows; a camera needs the same arithmetic with only the past available, at a
fixed cadence, for one subject. Three stateful pieces, wrapped by a fourth:

* `KeypointBuffer` -- the last `seq_len * frame_stride` frames of one subject's
  `(y, x, conf)` rows in the cache convention, handing out one `seq_len` window
  every `step` frames, sub-sampled exactly as training sampled it;
* `EnsembleClassifier` -- any number of checkpoints, each scored with its own
  feature variant, returning the mean P(fall);
* `RollingDecision` -- a deque of the last `buffer` window probabilities, their
  mean, and a latched alarm with hysteresis: the online form of
  `decide.rolling_alarm`, which is what notebook 17 measured;
* `EnsembleStream` -- wires them together: call `observe(keypoints)` once per
  frame and read `state`.

Cadence, in frames of video: with the trained defaults a window spans 4 s
(`seq_len=60` at `frame_stride=2`) and a new one is due every 1 s
(`step=15` strided frames). `buffer=4` therefore smooths over the last four
seconds of evidence, and the alarm can only engage once the buffer holds
windows -- a live stream fills it in `seq_len * frame_stride` frames, four
seconds of warm-up, not four windows.

This module is deliberately not demo-specific: demo-cam v2 is the first caller,
an edge deployment would be the second.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from . import config as cfg
from .data import build_features
from .train import load_checkpoint


class KeypointBuffer:
    """The last `seq_len * frame_stride` keypoint rows, sampled into windows.

    `add` returns True on the frames a new window is due, so the caller only
    needs a counter-free `if buffer.add(kp): score(buffer.window())`. A window
    is due once the buffer is full, every `step * frame_stride` rows after
    that: one second of video at the trained defaults.

    Rows are appended even when nobody was detected (all-zero keypoints, as the
    cache stores them), because detection dropout is part of what the classifier
    was trained on and inventing a person would hide it.
    """

    def __init__(self, seq_len: int = cfg.FINAL_TRAIN.seq_len,
                 frame_stride: int = cfg.PREPROCESS.frame_stride,
                 step: int = cfg.EVAL.sliding_stride):
        self.seq_len = seq_len
        self.frame_stride = frame_stride
        self.frames_per_window = seq_len * frame_stride
        self.frames_per_step = step * frame_stride
        self._rows: deque = deque(maxlen=self.frames_per_window)
        self._since = 0

    def add(self, keypoints: np.ndarray) -> bool:
        """Append one frame's `(17, 3)` rows; True when a window is due."""
        self._rows.append(np.asarray(keypoints, dtype=np.float32))
        self._since += 1
        if len(self._rows) < self.frames_per_window:
            return False
        return (self._since - self.frames_per_window) % self.frames_per_step == 0

    @property
    def ready(self) -> bool:
        return len(self._rows) >= self.frames_per_window

    @property
    def buffered(self) -> int:
        return len(self._rows)

    def window(self) -> np.ndarray:
        """The current `(seq_len, 17, 3)` window, sampled like training."""
        if not self.ready:
            raise RuntimeError(f"buffer holds {len(self._rows)} of "
                               f"{self.frames_per_window} frames")
        return np.stack(self._rows)[::self.frame_stride][-self.seq_len:]

    def reset(self) -> None:
        self._rows.clear()
        self._since = 0


class EnsembleClassifier:
    """A mean of P(fall) over checkpoints, each with its own feature variant.

    `coords + hard neg` consumes 34 dimensions where the others consume 51, so
    the feature builder is chosen per checkpoint from its saved config rather
    than assumed. Returns the per-model probabilities alongside the mean --
    the demo shows the mean, but the spread is what tells an operator whether
    the members agree.
    """

    def __init__(self, checkpoints, device: str = "cpu"):
        self.device = device
        self.names: list[str] = []
        self._members: list[tuple] = []
        for path in checkpoints:
            model, model_cfg = load_checkpoint(path, device=device)[:2]
            self._members.append((model, model_cfg))
            self.names.append(Path(path).stem)

    def proba(self, window: np.ndarray) -> tuple[float, list[float]]:
        """(mean P(fall), per-checkpoint probabilities) for one window."""
        probs = []
        for model, model_cfg in self._members:
            x = build_features(window, model_cfg.features)
            with torch.no_grad():
                logit = model(torch.from_numpy(np.ascontiguousarray(x))
                              .float().unsqueeze(0).to(self.device))
            probs.append(float(torch.sigmoid(logit).item()))
        return float(np.mean(probs)), probs


class RollingDecision:
    """The last `buffer` window probabilities and a latched alarm.

    Engages when the rolling mean reaches `threshold`; clears only after
    `clear_windows` consecutive updates below `clear_below` (hysteresis, so the
    alarm does not strobe as the buffer slides past the event). `buffer=1`
    makes it the per-window rule the notebooks call the peak rule.
    """

    def __init__(self, buffer: int = 4, threshold: float = cfg.EVAL.threshold,
                 clear_below: float = 0.2, clear_windows: int = 2):
        if buffer < 1:
            raise ValueError(f"buffer must be >= 1, not {buffer}")
        self.buffer = buffer
        self.threshold = threshold
        self.clear_below = clear_below
        self.clear_windows = clear_windows
        self._q: deque = deque(maxlen=buffer)
        self._below = 0
        self.alarm = False
        self.rolling = 0.0
        self.window_p = 0.0
        self.n_windows = 0
        self.history: list[float] = []

    def update(self, window_p: float) -> bool:
        """Feed one window probability; returns the (possibly latched) alarm."""
        self.window_p = float(window_p)
        self._q.append(self.window_p)
        self.rolling = float(np.mean(self._q))
        self.n_windows += 1
        self.history.append(round(self.window_p, 4))

        if self.rolling >= self.threshold:
            self.alarm = True
            self._below = 0
        elif self.alarm and self.rolling < self.clear_below:
            self._below += 1
            if self._below >= self.clear_windows:
                self.alarm = False
                self._below = 0
        else:
            self._below = 0
        return self.alarm

    def reset(self, clear_alarm: bool = True) -> None:
        self._q.clear()
        self._below = 0
        self.rolling = 0.0
        self.window_p = 0.0
        self.n_windows = 0
        self.history.clear()
        if clear_alarm:
            self.alarm = False

    def status(self) -> dict:
        return {
            "alarm": self.alarm,
            "p_fall": self.window_p,
            "rolling_mean": self.rolling,
            "buffer_len": self.buffer,
            "buffer_values": [round(v, 4) for v in self._q],
            "threshold": self.threshold,
            "clear_below": self.clear_below,
            "n_windows": self.n_windows,
            "history": list(self.history[-60:]),
        }


class EnsembleStream:
    """Per-frame facade: `observe(kp)` -> state, for one subject.

    `observe` takes the `(17, 3)` `(y, x, conf)` rows the cache stores, returns
    True when a window was scored, and keeps the decision in `decision`. On a
    stream reconnect, `reset_buffer` is the honest move: the buffered keypoints
    span a time gap, so the next window must be built from fresh frames while
    any latched alarm stays latched until its hysteresis clears it.
    """

    def __init__(self, checkpoints, buffer: int = 4,
                 threshold: float = cfg.EVAL.threshold,
                 clear_below: float = 0.2, clear_windows: int = 2,
                 seq_len: int = cfg.FINAL_TRAIN.seq_len,
                 frame_stride: int = cfg.PREPROCESS.frame_stride,
                 step: int = cfg.EVAL.sliding_stride, device: str = "cpu"):
        self.classifier = EnsembleClassifier(checkpoints, device=device)
        self.buffer = KeypointBuffer(seq_len, frame_stride, step)
        self.decision = RollingDecision(buffer, threshold, clear_below,
                                        clear_windows)
        self.model_probs: list[float] = []

    def observe(self, keypoints: np.ndarray) -> bool:
        if not self.buffer.add(keypoints):
            return False
        mean_p, per_model = self.classifier.proba(self.buffer.window())
        self.model_probs = per_model
        self.decision.update(mean_p)
        return True

    def reset_buffer(self) -> None:
        self.buffer.reset()
        self.model_probs = []

    def state(self) -> dict:
        st = self.decision.status()
        st["models"] = self.classifier.names
        st["model_probs"] = [round(p, 4) for p in self.model_probs]
        st["buffered_frames"] = self.buffer.buffered
        st["ready"] = self.buffer.ready
        return st
