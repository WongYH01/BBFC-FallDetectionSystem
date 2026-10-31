"""The Promotion Gate: whether a Candidate Head may replace the deployed one.

`judge(decision, deployed, candidate, heldout, evals, seq_len)` scores both
heads the way the room runs them: every clip is replayed from its first frame
through `decision`'s own camera (`FallDecision.build_camera`: rolling buffer,
threshold, latch, descent gate, one stream per tracked subject or the largest
box, as `decision.subjects` says), once on each head's classifier. A clip counts as raised when the room's alarm goes up
anywhere in it. The two eval sets:

* the held-out picam session, staged falls and everyday activities, which no
  Camera Head was fitted on;
* the Eval Alerts, each one Keypoint File from its first frame. A Genuine one
  raised is caught; a False Alarm one raised is a False Alarm.

Every Eval Alert exists because some earlier head fired on it, so the
deployed head catches every Genuine one and raises every False Alarm one: the
Eval Alerts measure False Alarm reduction, and recall rests on the held-out
session. Hence both, and the rule: the candidate passes when it loses no fall
the deployed head caught, on either set, and raises at least
`MIN_FEWER_FALSE_ALARMS` fewer False Alarms over the two together.

This module reads no files: the caller loads the clips and builds the two
classifiers (an `EnsembleClassifier` each, the deployed checkpoint with the
deployed head and with the candidate). The report is counts only, never an
Event ID or a time (ADR 0002).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import config as cfg
from .extract import Sighting
from .fall_decision import FallDecision
from .keypoint_file import Frame

#: A candidate must raise at least this many fewer False Alarms than the
#: deployed head: one fewer is inside the Seed Floor.
MIN_FEWER_FALSE_ALARMS = 2


@dataclass(frozen=True)
class Clip:
    """One clip the gate replays: its frames, and whether it is a fall (a
    staged fall, or a Genuine Eval Alert)."""

    frames: Sequence[Frame]
    fall: bool


@dataclass(frozen=True)
class Score:
    """One head on one eval set."""

    #: Fall clips the alarm went up in.
    falls: int
    #: Clips that are not falls the alarm went up in.
    false_alarms: int


@dataclass(frozen=True)
class SetReport:
    """Both heads on one eval set."""

    clips: int
    fall_clips: int
    deployed: Score
    candidate: Score
    #: Fall clips the deployed head caught and the candidate does not. Clip
    #: by clip, so a fall the candidate newly catches cannot pay for one it
    #: loses.
    lost: int


@dataclass(frozen=True)
class GateReport:
    heldout: SetReport
    evals: SetReport
    passed: bool


def _raised(decision: FallDecision, classifier, clip: Clip,
            seq_len: int) -> bool:
    """Whether the room's alarm goes up anywhere in `clip`, replayed from its
    first frame.

    Under FALL_SUBJECTS=largest the camera decides on the largest box alone,
    and a frame with nobody in it is a zero row, as the detector feeds it.
    """
    camera = decision.build_camera(classifier=classifier, seq_len=seq_len)
    tracked = decision.subjects == "tracked"
    empty = np.zeros((cfg.N_KEYPOINTS, 3), dtype=np.float32)
    for frame in clip.frames:
        camera.aspect = frame.aspect
        if tracked:
            camera.observe([Sighting(track_id=p.track_id, row=p.keypoints,
                                     centre=p.centre, area=p.area)
                            for p in frame.people])
            alarm = camera.alarm
        else:
            largest = max(frame.people, key=lambda p: p.area, default=None)
            camera.observe(empty if largest is None else largest.keypoints)
            alarm = camera.decision.alarm
        if alarm:
            return True
    return False


def _score(clips: Sequence[Clip], raised: Sequence[bool]) -> Score:
    return Score(falls=sum(r and c.fall for r, c in zip(raised, clips)),
                 false_alarms=sum(r and not c.fall
                                  for r, c in zip(raised, clips)))


def _set(decision, deployed, candidate, clips, seq_len) -> SetReport:
    before = [_raised(decision, deployed, c, seq_len) for c in clips]
    after = [_raised(decision, candidate, c, seq_len) for c in clips]
    return SetReport(clips=len(clips), fall_clips=sum(c.fall for c in clips),
                     deployed=_score(clips, before),
                     candidate=_score(clips, after),
                     lost=sum(c.fall and b and not a
                              for c, b, a in zip(clips, before, after)))


def judge(decision: FallDecision, deployed, candidate,
          heldout: Sequence[Clip], evals: Sequence[Clip],
          seq_len: int) -> GateReport:
    """The Promotion Gate's report on `candidate` against `deployed`.

    `deployed` and `candidate` are classifiers (`stream.EnsembleClassifier`,
    or anything with its `proba`), each run under `decision`'s rule;
    `seq_len` is the window their checkpoint was trained on.
    """
    sets = [_set(decision, deployed, candidate, clips, seq_len)
            for clips in (heldout, evals)]
    lost = sum(s.lost for s in sets)
    fewer = sum(s.deployed.false_alarms - s.candidate.false_alarms
                for s in sets)
    return GateReport(heldout=sets[0], evals=sets[1],
                      passed=not lost and fewer >= MIN_FEWER_FALSE_ALARMS)
