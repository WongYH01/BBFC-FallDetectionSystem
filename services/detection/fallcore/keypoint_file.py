"""The keypoint file sent with a fall alert: the clip's span as pose numbers.

The Skeleton Clip is what a responder triages; this file is the same frames of
the same incident as the rows the Fall Decision read, so the Alert's Outcome
can be trained on. The notification service stores it beside the clip and, once the
Alert is triaged, files it under its Outcome (`keypoints/labelled/genuine/` or
`.../false_alarm/`): every alert becomes a labelled example from this camera,
and a False Alarm becomes a hard negative without anyone staging it again. It
holds keypoints only, never camera pixels (ADR 0001).

One `.npz` -- numpy's zip of arrays, so no extra dependency, and the
notification service checks it by its `PK\\x03\\x04` zip header. `np.load(path)`
gives:

per frame, F of them, oldest first (one per frame of the skeleton clip):

    t          (F,)         float64  capture time, unix seconds
    aspect     (F,)         float32  frame width / height; x and y are fractions
                                     of different lengths, so window kinematics
                                     need it to put them in the same units
    alarm      (F,)         bool     the room's alarm
    score      (F,)         float32  the rolling mean the alarm is decided on,
                                     the lead subject's
    lead       (F,)         int32    the subject the alarm and score are about:
                                     the alarmed one, else the highest rolling
                                     mean. -1 under FALL_SUBJECTS=largest

per sighting, N of them, in frame order:

    frame      (N,)         int32    index into the per-frame arrays
    track_id   (N,)         int32    the tracker's id for that person
    subject    (N,)         int32    the Subject the row was decided as. The
                                     tracker re-identifies a faller in about
                                     40% of falls, mid-descent; a Reconnect
                                     keeps both ids one subject, so this, not
                                     track_id, is one person's whole stream.
                                     -1 under FALL_SUBJECTS=largest
    keypoints  (N, 17, 3)   float32  (y_norm, x_norm, conf): the cache
                                     convention, `fallcore.extract.subject_row`
    centre     (N, 2)       float32  box centre (x, y), fractions of the frame
    area       (N,)         float32  box share of the frame
    fallen     (N,)         bool     that person's own alarm (under
                                     FALL_SUBJECTS=largest, the room's)

scalars:

    version    ()           int32    `FORMAT_VERSION`
    trigger    ()           int32    index of the frame the alarm latched on;
                                     the frames before it are the clip's
                                     pre-event lead-in
    raised_by  ()           int32    the subject whose alarm raised the Alert,
                                     `lead[trigger]`; -1 under
                                     FALL_SUBJECTS=largest

The Alert's Outcome is about `raised_by` alone. Another subject can be down in
the same clip -- a carer kneeling beside the resident who fell -- and a Genuine
Outcome says nothing about them.

A frame in which nobody was tracked has no rows. Rebuilding one person's stream
means taking their subject's rows and filling the frames without one with zero
rows, which is how the keypoint caches store absence and what the classifier
was trained on. The subject that raised the Alert may have no row on the
trigger frame at all: the detector can lose a faller as they land.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as cfg

FORMAT_VERSION = 1


@dataclass(frozen=True)
class Person:
    """One tracked person in one frame, as the file stores them."""

    track_id: int
    keypoints: np.ndarray            # (17, 3) as (y_norm, x_norm, conf)
    centre: tuple[float, float]
    area: float
    fallen: bool
    subject: int = -1                # -1: not decided per person


@dataclass(frozen=True)
class Frame:
    """One frame of the clip: when it was, the room's decision, and who was in it."""

    t: float
    aspect: float
    alarm: bool
    score: float
    people: tuple[Person, ...] = ()
    lead: int = -1                   # -1: not decided per person


def people(seen: Iterable, fallen_of: Callable[[int], bool],
           subject_of: Callable[[int], int | None] | None = None
           ) -> tuple[Person, ...]:
    """This frame's `fallcore.extract.Sighting`s as `Person`s.

    `fallen_of(track_id)` is whether that person's alarm is up: the subject's
    own when every tracked person is decided separately, the room's when only
    the largest box is. `subject_of(track_id)` is the subject it is decided
    as (`SubjectStreams.subject_of`); without it, or when it answers None,
    the row's subject is -1.
    """
    def subject(track_id) -> int:
        sid = subject_of(track_id) if subject_of is not None else None
        return -1 if sid is None else int(sid)

    return tuple(Person(track_id=int(s.track_id), keypoints=s.row,
                        centre=(float(s.centre[0]), float(s.centre[1])),
                        area=float(s.area), fallen=bool(fallen_of(s.track_id)),
                        subject=subject(s.track_id))
                 for s in seen)


def save(path: str | Path, frames: Sequence[Frame], trigger: int) -> None:
    """Write `frames` to `path`, which only ever appears complete.

    The file is written beside its final name and renamed into place, so a
    process that dies mid-write leaves a `.tmp` nobody uploads -- the same
    rule the skeleton clip's `.raw.mp4` follows.
    """
    if not 0 <= trigger < len(frames):
        raise ValueError(f"trigger {trigger} is not one of the {len(frames)} frames")
    rows = [(i, p) for i, f in enumerate(frames) for p in f.people]
    n = len(rows)
    keypoints = (np.stack([np.asarray(p.keypoints, dtype=np.float32) for _, p in rows])
                 if rows else np.zeros((0, cfg.N_KEYPOINTS, 3), dtype=np.float32))
    arrays = {
        "version": np.int32(FORMAT_VERSION),
        "trigger": np.int32(trigger),
        "raised_by": np.int32(frames[trigger].lead),
        "t": np.array([f.t for f in frames], dtype=np.float64),
        "aspect": np.array([f.aspect for f in frames], dtype=np.float32),
        "alarm": np.array([f.alarm for f in frames], dtype=bool),
        "score": np.array([f.score for f in frames], dtype=np.float32),
        "lead": np.array([f.lead for f in frames], dtype=np.int32),
        "frame": np.array([i for i, _ in rows], dtype=np.int32),
        "track_id": np.array([p.track_id for _, p in rows], dtype=np.int32),
        "subject": np.array([p.subject for _, p in rows], dtype=np.int32),
        "keypoints": keypoints,
        "centre": np.array([p.centre for _, p in rows], dtype=np.float32).reshape(n, 2),
        "area": np.array([p.area for _, p in rows], dtype=np.float32),
        "fallen": np.array([p.fallen for _, p in rows], dtype=bool),
    }
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        # A file object, not a name: given a name, numpy appends ".npz" to
        # anything that does not already end in it.
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
