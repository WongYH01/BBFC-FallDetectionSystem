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
    #: A tracker id of the subject whose alarm raised the Alert. A Genuine
    #: Outcome is about that subject alone (ADR 0002), so in a fall clip only
    #: its windows are labelled a fall and everyone else's are unlabelled;
    #: in a clip that is not a fall every subject's are ADLs. None: every
    #: subject's windows carry the clip's label, for clips that do not say.
    #: A Keypoint File records `raised_by` as a subject, so the caller passes
    #: one of that subject's track ids. Window AUC only: the room's alarm
    #: going up for anyone in a fall clip still counts it caught.
    raised_by: int | None = None


@dataclass(frozen=True)
class Score:
    """One head on one eval set."""

    #: Fall clips the alarm went up in.
    falls: int
    #: Clips that are not falls the alarm went up in.
    false_alarms: int
    #: ROC AUC of every labelled window the head scored, each labelled as
    #: its clip is (see `Clip.raised_by`). Reported only: it ranks windows,
    #: and the gate decides on alarms. None when the set does not hold both
    #: kinds.
    auc: float | None


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
    """The Promotion Gate's judgement, as counts only. `str()` is what the run
    summary and a promotion's pull request carry, so it names no Alert and
    no time."""

    heldout: SetReport
    evals: SetReport
    passed: bool
    reason: str

    def __str__(self) -> str:
        return "\n".join([
            f"Promotion Gate {'passed' if self.passed else 'failed'}: "
            f"{self.reason}",
            _line(f"held-out session ({self.heldout.clips} clips, "
                  f"{self.heldout.fall_clips} falls)", self.heldout),
            _line(f"Eval Alerts ({self.evals.clips}, {self.evals.fall_clips} "
                  f"Genuine)", self.evals),
        ])


def _auc_text(auc: float | None) -> str:
    return "n/a" if auc is None else f"{auc:.3f}"


def _line(name: str, s: SetReport) -> str:
    d, c = s.deployed, s.candidate
    return (f"{name}: falls {d.falls} -> {c.falls}, False Alarms "
            f"{d.false_alarms} -> {c.false_alarms}, window AUC "
            f"{_auc_text(d.auc)} -> {_auc_text(c.auc)}")


def _replay(decision: FallDecision, classifier, clip: Clip, seq_len: int
            ) -> tuple[bool, list[tuple[float, tuple[int, ...] | None]]]:
    """Whether the room's alarm goes up anywhere in `clip`, replayed from its
    first frame to its last, and every window P(fall) scored on the way, each
    with the track ids of the subject that scored it (None under
    FALL_SUBJECTS=largest, where the camera has one subject).

    Under FALL_SUBJECTS=largest the camera decides on the largest box alone,
    and a frame with nobody in it is a zero row, as the detector feeds it.
    """
    camera = decision.build_camera(classifier=classifier, seq_len=seq_len)
    tracked = decision.subjects == "tracked"
    empty = np.zeros((cfg.N_KEYPOINTS, 3), dtype=np.float32)
    raised = False
    scored = []
    for frame in clip.frames:
        camera.aspect = frame.aspect
        if tracked:
            before = {s: s.stream.decision.n_windows for s in camera.subjects}
            camera.observe([Sighting(track_id=p.track_id, row=p.keypoints,
                                     centre=p.centre, area=p.area)
                            for p in frame.people])
            # Before and after: a subject dropped on this frame scored first.
            for s in dict.fromkeys([*before, *camera.subjects]):
                if s.stream.decision.n_windows > before.get(s, 0):
                    scored.append((s.stream.decision.window_p, s))
            raised |= camera.alarm
        else:
            largest = max(frame.people, key=lambda p: p.area, default=None)
            if camera.observe(empty if largest is None else largest.keypoints):
                scored.append((camera.decision.window_p, None))
            raised |= camera.decision.alarm
    # Track ids as of the clip's end, so a window scored before a Reconnect
    # joined the raising id to its subject is still that subject's.
    return raised, [(p, None if s is None else tuple(s.track_ids))
                    for p, s in scored]


def _labelled(clip: Clip, windows) -> list[float]:
    """The window P(fall)s in `clip` that carry its label: all of them, but
    in a fall clip that names `raised_by`, that subject's alone."""
    if not clip.fall or clip.raised_by is None:
        return [p for p, _ in windows]
    return [p for p, ids in windows if ids is None or clip.raised_by in ids]


def _auc(clips: Sequence[Clip], windows) -> float | None:
    from sklearn.metrics import roc_auc_score

    labelled = [(c.fall, p) for c, ws in zip(clips, windows)
                for p in _labelled(c, ws)]
    labels = [fall for fall, _ in labelled]
    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, [p for _, p in labelled]))


def _score(clips: Sequence[Clip], replays) -> Score:
    raised = [r for r, _ in replays]
    return Score(falls=sum(r and c.fall for r, c in zip(raised, clips)),
                 false_alarms=sum(r and not c.fall
                                  for r, c in zip(raised, clips)),
                 auc=_auc(clips, [ws for _, ws in replays]))


def _set(decision, deployed, candidate, clips, seq_len) -> SetReport:
    before = [_replay(decision, deployed, c, seq_len) for c in clips]
    after = [_replay(decision, candidate, c, seq_len) for c in clips]
    return SetReport(clips=len(clips), fall_clips=sum(c.fall for c in clips),
                     deployed=_score(clips, before),
                     candidate=_score(clips, after),
                     lost=sum(c.fall and b and not a for c, (b, _), (a, _)
                              in zip(clips, before, after)))


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
    heldout, evals = sets
    lost = heldout.lost + evals.lost
    fewer = sum(s.deployed.false_alarms - s.candidate.false_alarms
                for s in sets)
    alarms = (f"{fewer} fewer False Alarm{'' if abs(fewer) == 1 else 's'}"
              if fewer >= 0 else
              f"{-fewer} more False Alarm{'' if fewer == -1 else 's'}")
    if lost:
        reason = (f"{lost} fall{'' if lost == 1 else 's'} lost ({heldout.lost} "
                  f"on the held-out session, {evals.lost} on the Eval "
                  f"Alerts), {alarms}")
    elif fewer < MIN_FEWER_FALSE_ALARMS:
        reason = (f"no fall lost, but {alarms}, fewer than "
                  f"{MIN_FEWER_FALSE_ALARMS}")
    else:
        reason = f"no fall lost, {alarms}"
    return GateReport(heldout=heldout, evals=evals,
                      passed=not lost and fewer >= MIN_FEWER_FALSE_ALARMS,
                      reason=reason)
