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

`SubjectStreams` runs one `EnsembleStream` per tracked person instead of one
for the largest box, sharing the classifier, and raises the room's alarm when
any of them does. See its docstring for how a subject outlives its tracker id.

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
from .calibrate import KINEMATIC_NAMES, window_kinematics
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
        self.step = step
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

    `heads` optionally replaces the pretrained classifier of a checkpoint with a
    room-fitted logistic head (`fallcore.calibrate.ProbeHead`), one entry per
    checkpoint or `None` to keep the pretrained head there. A head consumes the
    pooled embedding rather than the feature matrix, so the forward pass stops
    one step earlier (`encoder(embed(x))` then `_pool`); that is the whole
    difference, and it is the path `scripts/fit_calibration_head.py` measured.
    """

    def __init__(self, checkpoints, device: str = "cpu", heads=None):
        from .calibrate import ProbeHead

        self.device = device
        if heads is None:
            heads = [None] * len(checkpoints)
        heads = list(heads)
        if len(heads) != len(checkpoints):
            raise ValueError(f"{len(heads)} heads for {len(checkpoints)} "
                             f"checkpoints -- they are aligned one to one")
        self.names: list[str] = []
        self._members: list[tuple] = []
        for path, head in zip(checkpoints, heads):
            if isinstance(head, (str, Path)):
                head = ProbeHead.load(head)
            model, model_cfg = load_checkpoint(path, device=device)[:2]
            if head is not None and head.features != model_cfg.features:
                raise ValueError(
                    f"head calibrated for {head.features!r} features cannot "
                    f"score {Path(path).name}, which reads {model_cfg.features!r}")
            if (head is not None and head.backbone
                    and Path(head.backbone).name != Path(path).name):
                # A head is a function of one backbone's embedding space; on
                # another checkpoint it is a random linear map that still
                # returns confident-looking numbers.
                raise ValueError(
                    f"head was fitted for {Path(head.backbone).name}, not "
                    f"{Path(path).name} -- refit it "
                    f"(scripts/fit_calibration_head.py)")
            self._members.append((model, model_cfg, head))
            self.names.append(Path(path).stem + ("+head" if head is not None else ""))
        self.heads = [m[2] for m in self._members]

    def proba(self, window: np.ndarray,
              aspect: float = 1.0) -> tuple[float, list[float]]:
        """(mean P(fall), per-checkpoint probabilities) for one window."""
        probs = []
        for model, model_cfg, head in self._members:
            x = build_features(window, model_cfg.features)
            with torch.no_grad():
                xt = (torch.from_numpy(np.ascontiguousarray(x))
                      .float().unsqueeze(0).to(self.device))
                if head is None:
                    logit = model(xt)
                    probs.append(float(torch.sigmoid(logit).item()))
                else:
                    pooled = model._pool(model.encoder(model.embed(xt)))
                    extras = (window_kinematics(window, aspect)[None, :]
                              if head.n_extra else None)
                    probs.append(
                        float(head.proba(pooled.cpu().numpy(), extras)[0]))
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
        self.gated = False
        self.rolling = 0.0
        self.window_p = 0.0
        self.n_windows = 0
        self.history: list[float] = []

    def update(self, window_p: float, can_raise: bool = True) -> bool:
        """Feed one window probability; returns the (possibly latched) alarm.

        `can_raise=False` lets the mean keep tracking but forbids *engaging* the
        alarm on this window. It is how the descent gate works: a window the
        subject merely lies still through carries no evidence about how they got
        there, so it may sustain or clear an alarm but never start one. Falls
        are unaffected -- they engage on the descent, which is never gated.
        """
        self.window_p = float(window_p)
        self._q.append(self.window_p)
        self.rolling = float(np.mean(self._q))
        self.n_windows += 1
        self.history.append(round(self.window_p, 4))
        self.gated = not can_raise

        if self.rolling >= self.threshold and (can_raise or self.alarm):
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
            "gated": self.gated,
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
                 step: int = cfg.EVAL.sliding_stride, device: str = "cpu",
                 heads=None, min_descent: float = 0.0,
                 aspect: float = 1.0, classifier=None):
        # `classifier` shares an already-loaded one (see `fresh`) instead of
        # loading `checkpoints` again.
        self.classifier = (classifier if classifier is not None else
                           EnsembleClassifier(checkpoints, device=device,
                                              heads=heads))
        self.buffer = KeypointBuffer(seq_len, frame_stride, step)
        self.decision = RollingDecision(buffer, threshold, clear_below,
                                        clear_windows)
        self.model_probs: list[float] = []
        # Body lengths per second the hips must fall somewhere inside a window
        # before that window is allowed to raise an alarm. 0 disables the gate,
        # which is the behaviour every result before 2026-09-24 was measured at.
        self.min_descent = float(min_descent)
        # The source frame's width/height. The kinematics divide by the body's
        # longest dimension, which needs x and y in the same units; set this
        # from the camera (demo-cam) or per clip (picam.score_clip).
        self.aspect = float(aspect)
        self.kinematics = np.zeros(len(KINEMATIC_NAMES))

    def observe(self, keypoints: np.ndarray, can_raise: bool = True) -> bool:
        """Append one frame; True when a window was scored.

        `can_raise=False` forbids this frame's window from engaging the alarm
        on top of the descent gate (see `RollingDecision.update`).
        """
        if not self.buffer.add(keypoints):
            return False
        window = self.buffer.window()
        mean_p, per_model = self.classifier.proba(window, self.aspect)
        self.model_probs = per_model
        self.kinematics = window_kinematics(window, self.aspect)
        can_raise = can_raise and (self.min_descent <= 0.0
                                   or float(self.kinematics[0]) >= self.min_descent)
        self.decision.update(mean_p, can_raise=can_raise)
        return True

    def reset_buffer(self) -> None:
        self.buffer.reset()
        self.model_probs = []
        self.kinematics = np.zeros(len(KINEMATIC_NAMES))

    def fresh(self) -> EnsembleStream:
        """An empty stream running this one's rule on this one's classifier."""
        d, b = self.decision, self.buffer
        return EnsembleStream(
            [], buffer=d.buffer, threshold=d.threshold,
            clear_below=d.clear_below, clear_windows=d.clear_windows,
            seq_len=b.seq_len, frame_stride=b.frame_stride, step=b.step,
            min_descent=self.min_descent, aspect=self.aspect,
            classifier=self.classifier)

    def state(self) -> dict:
        st = self.decision.status()
        st["models"] = self.classifier.names
        st["model_probs"] = [round(p, 4) for p in self.model_probs]
        st["buffered_frames"] = self.buffer.buffered
        st["ready"] = self.buffer.ready
        st["min_descent"] = self.min_descent
        st.update({k: round(float(v), 4)
                   for k, v in zip(KINEMATIC_NAMES, self.kinematics)})
        return st


class _Subject:
    """One person the room is following, under however many tracker ids."""

    def __init__(self, sid: int, stream: EnsembleStream, frame: int,
                 centre: tuple[float, float], heir: bool = False):
        self.sid = sid
        self.stream = stream
        self.first_seen = frame
        self.last_seen = frame
        self.centre = centre
        self.track_ids: list[int] = []
        # Whether someone else was missing when this subject first appeared,
        # so it may be them, re-identified (see `SubjectStreams`).
        self.heir = heir
        self._lo = self._hi = np.asarray(centre, dtype=np.float64)

    def see(self, frame: int, centre: tuple[float, float]) -> None:
        self.last_seen, self.centre = frame, centre
        c = np.asarray(centre, dtype=np.float64)
        self._lo, self._hi = np.minimum(self._lo, c), np.maximum(self._hi, c)

    @property
    def travel(self) -> float:
        """The diagonal of every box centre seen so far, in frame fractions."""
        return float(np.hypot(*(self._hi - self._lo)))


class SubjectStreams:
    """One `EnsembleStream` per tracked subject, and the room's alarm.

    Call `observe(sightings)` once per frame with every tracked person
    (`fallcore.extract.sightings`). Each subject's keypoints fill its own
    buffer and are decided by its own rolling rule and descent gate, so a fall
    behind a closer bystander is scored on the faller's keypoints rather than
    the bystander's. The room's alarm is up while any subject's is.

    A subject is not a tracker id. The tracker re-identifies the faller in about
    40% of picam falls (34 of 82 with yolo26n), mostly half a second to a
    second into the descent, when the box turns from tall to wide. If a new id
    started a new stream, neither stream would hold the whole fall, and 20 of
    75 falls were lost that way. So:

    * **A lost subject keeps running** on zero rows (the same "nobody seen" rows
      a single-subject stream gets) for one window after its last sighting,
      so the windows that hold its descent are still scored. It is dropped
      after that, unless its alarm is still up; then it is kept until that
      alarm clears, but never past two windows, so a classifier that scores an
      empty window above `clear_below` cannot hold the room's alarm forever.
    * **A new tracker id reconnects** to the nearest lost subject last seen
      within `reconnect_sec` and within `reconnect_dist` of the new box's
      centre (fractions of the frame), and continues that subject's stream.
      Otherwise it starts a new subject, whose stream begins at that first
      sighting.

    With both, 75 of 82 single-person picam falls are caught on real ByteTrack
    ids (the largest-box rule also catches 75), against 72 with the first
    alone. Both settings come from `FallDecision`.

    Deciding on everyone also decides on furniture the detector takes for a
    person: picam-1's stacked chairs, picam-3's armchair cushions. The largest
    box never picks them while someone is in view; here each is a subject of
    its own. A head that reads low-and-wide as a fall scores the cushions 0.99,
    an occluded chair's empty rows score 0.92 (the classifier's "vanished"
    shortcut), and their keypoint jitter alone reads 0.11 body lengths/s, just
    over the descent gate. So:

    * **A subject that has never moved cannot raise the alarm.** Until its box
      centre has spanned `min_travel` of the frame, seen or lost, its windows
      may sustain an alarm but not start one. A person cannot fall in view
      without moving; furniture never does.
    * **Unless it appeared while someone else was missing.** Then it may be
      that person, re-identified out of reconnect range after a fall the
      detector lost them during, and already lying still.

    That takes the false alarms on picam's single-person everyday clips from
    9 to 7 of 82 (the largest box's 7) and catches the same 75 falls and all 4
    multi_fall recordings. `min_travel` is picked, not tuned: the furniture
    spans at most 0.035 and the stillest faller to raise an alarm 0.136, and
    any value in between gives those counts. Both rescued false alarms and
    both falls the exception keeps are single clips, so the counts are fragile.
    """

    def __init__(self, template: EnsembleStream, reconnect_sec: float = 2.0,
                 reconnect_dist: float = 0.35, min_travel: float = 0.07):
        self.template = template
        self.classifier = template.classifier
        # Frames, at whatever rate the caller observes them: one window, and
        # `reconnect_sec` counted in steps, each a second of video.
        self.keep = template.buffer.frames_per_window
        self.reconnect = round(reconnect_sec * template.buffer.frames_per_step)
        self.reconnect_dist = float(reconnect_dist)
        self.min_travel = float(min_travel)
        self.aspect = template.aspect
        self.n_windows = 0
        self._frame = 0
        self._next_sid = 1
        self._subjects: dict[int, _Subject] = {}
        self._by_track: dict[int, int] = {}
        self._empty = np.zeros((cfg.N_KEYPOINTS, 3), dtype=np.float32)

    @property
    def alarm(self) -> bool:
        return any(s.stream.decision.alarm for s in self._subjects.values())

    @property
    def subjects(self) -> list[_Subject]:
        return list(self._subjects.values())

    def alarm_of(self, track_id) -> bool:
        """Whether the subject behind this tracker id has its alarm up."""
        sid = self._by_track.get(track_id)
        return sid in self._subjects and self._subjects[sid].stream.decision.alarm

    def observe(self, sightings) -> bool:
        """Feed one frame's sightings; True when any subject scored a window."""
        frame = self._frame
        self._frame += 1
        rows: dict[int, np.ndarray] = {}
        # Largest first, so when two boxes land on one subject (a tracker
        # briefly holding an old and a new id on a person) the bigger one is used.
        for s in sorted(sightings, key=lambda s: -s.area):
            sid = self._by_track.get(s.track_id)
            if sid not in self._subjects:
                sid = self._reconnect(s, frame, rows) or self._start(s, frame)
                self._by_track[s.track_id] = sid
                self._subjects[sid].track_ids.append(s.track_id)
            if sid in rows:
                continue
            rows[sid] = s.row
            self._subjects[sid].see(frame, s.centre)

        scored = False
        for sid, subject in list(self._subjects.items()):
            subject.stream.aspect = self.aspect
            can_raise = subject.heir or subject.travel >= self.min_travel
            if subject.stream.observe(rows.get(sid, self._empty), can_raise):
                scored = True
                self.n_windows += 1
            unseen = frame - subject.last_seen
            if unseen >= self.keep and (not subject.stream.decision.alarm
                                        or unseen >= 2 * self.keep):
                self._drop(sid)
        return scored

    def _reconnect(self, s, frame: int, taken) -> int | None:
        """The lost subject this new tracker id is, or None."""
        best, best_dist = None, self.reconnect_dist
        for sid, subject in self._subjects.items():
            if sid in taken or subject.last_seen >= frame \
                    or frame - subject.last_seen > self.reconnect:
                continue
            dist = float(np.hypot(s.centre[0] - subject.centre[0],
                                  s.centre[1] - subject.centre[1]))
            if dist <= best_dist:
                best, best_dist = sid, dist
        return best

    def _start(self, s, frame: int) -> int:
        sid = self._next_sid
        self._next_sid += 1
        # Missing: not seen last frame and not yet dropped. Subjects seen last
        # frame but not yet assigned in this one still have last_seen frame - 1.
        heir = any(o.last_seen < frame - 1 for o in self._subjects.values())
        self._subjects[sid] = _Subject(sid, self.template.fresh(), frame, s.centre,
                                       heir=heir)
        return sid

    def _drop(self, sid: int) -> None:
        subject = self._subjects.pop(sid)
        for tid in subject.track_ids:
            if self._by_track.get(tid) == sid:
                del self._by_track[tid]

    def reset_buffer(self) -> None:
        """After a stream outage: forget the tracker's ids and every buffer.

        The tracker restarts its numbering on a new connection, so no old id
        means anything. A subject whose alarm is up keeps it (an outage is not
        evidence that the person got up) and can still be reconnected by
        position; every other subject is dropped.
        """
        self._by_track.clear()
        for sid, subject in list(self._subjects.items()):
            if subject.stream.decision.alarm:
                subject.stream.reset_buffer()
                subject.track_ids.clear()
            else:
                del self._subjects[sid]

    def _lead(self) -> _Subject | None:
        """The subject the room's numbers report: the alarmed one, else the
        highest rolling mean."""
        scored = [s for s in self._subjects.values() if s.stream.decision.n_windows]
        if not scored:
            return None
        return max(scored, key=lambda s: (s.stream.decision.alarm,
                                          s.stream.decision.rolling))

    def state(self) -> dict:
        """The lead subject's state with the room's alarm and window count.

        The same keys `EnsembleStream.state` gives, so a caller reading one
        reads the other, plus `subject` (the lead's id) and `subjects`.
        """
        lead = self._lead()
        st = (lead.stream if lead else self.template).state()
        if lead is None:
            st.update(alarm=False, gated=False, p_fall=0.0, rolling_mean=0.0,
                      buffer_values=[], history=[], model_probs=[],
                      buffered_frames=max((s.stream.buffer.buffered
                                           for s in self._subjects.values()),
                                          default=0),
                      ready=False)
        st["alarm"] = self.alarm
        st["n_windows"] = self.n_windows
        st["subject"] = lead.sid if lead else None
        st["subjects"] = [
            {"subject": s.sid, "track_ids": list(s.track_ids),
             "alarm": s.stream.decision.alarm,
             "p_fall": round(s.stream.decision.window_p, 4),
             "rolling_mean": round(s.stream.decision.rolling, 4),
             "seen": s.last_seen == self._frame - 1}
            for s in self._subjects.values()]
        return st
