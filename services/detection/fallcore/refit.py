"""The weekly Refit: one entry point from a mirrored bucket to a verdict.

`run(folder, decision, fitted_at)` is the whole Refit. The workflow around it
mirrors the bucket into `folder`, calls it, and acts on what comes back; this
module does no network I/O, so everything it decides is tested on a temporary
folder.

`folder` is laid out like the bucket:

    keypoints/incoming/<room>/<event id>.npz          untriaged; never used
    keypoints/labelled/genuine/<room>/<event id>.npz  triaged, by Outcome
    keypoints/labelled/false_alarm/<room>/<event id>.npz
    keypoints/commissioning/...                       the Commissioning Recording
    keypoints/heldout/...                             the held-out picam session
    pinned.sha256                                     the manifest, from git

The two pinned prefixes are uploaded once and never change; the manifest
(`sha256sum` format, paths relative to `folder`) pins every object under them,
and anything that does not match it is an error, not a verdict, so a changed
base set cannot silently move the Promotion Gate. The manifest is what
`sha256sum` prints, run from `folder` over the two prefixes.

An Alert's Event ID is its Keypoint File's object name without `.npz`, so
filing it under its Outcome does not move it across the eval split. Untriaged
files, and files whose `pose` is not the deployed backbone's, are left out and
counted.

"New" means raised after the deployed head's fit date, by the trigger frame's
capture time: the file carries it, so the loop keeps no state of its own. An
Alert raised before the fit date but triaged after it is trained on, but does
not count towards this week's new data.

Everything published -- the verdict's `str()`, which the workflow writes to
the run summary -- is counts and the training set's hash, never an Event ID
or a time (ADR 0002).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import keypoint_file
from .fall_decision import FallDecision

#: The objects every Refit fits and judges on, uploaded once.
PINNED = ("keypoints/commissioning", "keypoints/heldout")
#: Where the manifest for `PINNED` sits in the folder.
MANIFEST = "pinned.sha256"
LABELLED = "keypoints/labelled"
INCOMING = "keypoints/incoming"
#: The Outcomes, as the notification service names their folders.
GENUINE, FALSE_ALARM = "genuine", "false_alarm"
OUTCOMES = (GENUINE, FALSE_ALARM)

#: A week with fewer new triaged training Alerts than this is skipped: the
#: candidate would be fitted on nearly the deployed head's data, and the gate
#: would be measuring noise.
MIN_NEW = 5

SKIPPED = "skipped"
#: What a week that is not skipped returns until fitting and the Promotion
#: Gate are built.
NOT_YET_IMPLEMENTED = "not yet implemented"


class PinnedDataError(RuntimeError):
    """The pinned base data does not match its manifest."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check_pinned(folder: Path) -> None:
    """Refuse unless every object under `PINNED` is exactly the manifest's."""
    manifest = folder / MANIFEST
    if not manifest.is_file():
        raise PinnedDataError(f"no manifest at {MANIFEST}")
    pinned: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.strip().split(maxsplit=1)
        # `sha256sum -b` marks its names with a "*".
        pinned[name.removeprefix("*")] = digest.lower()
    found = {p.relative_to(folder).as_posix(): p
             for prefix in PINNED for p in sorted((folder / prefix).rglob("*"))
             if p.is_file()}
    for prefix in PINNED:
        if not any(name.startswith(prefix + "/") for name in pinned):
            raise PinnedDataError(f"the manifest pins nothing under {prefix}/")
    for name in sorted(pinned.keys() | found.keys()):
        if name not in found:
            raise PinnedDataError(f"{name} is in the manifest but missing")
        if name not in pinned:
            raise PinnedDataError(f"{name} is not in the manifest")
        if _sha256(found[name]) != pinned[name]:
            raise PinnedDataError(f"{name} does not match its SHA-256 in the "
                                  f"manifest")


@dataclass(frozen=True)
class Alert:
    """One triaged Alert's Keypoint File, as the Refit sorts it."""

    event_id: str
    outcome: str
    #: When the alarm latched, unix seconds: the trigger frame's capture time.
    raised_at: float
    path: Path


@dataclass(frozen=True)
class Report:
    """What a Refit used, as counts only."""

    #: Keypoint Files left out: never triaged, or another pose backbone's.
    untriaged: int = 0
    other_pose: int = 0
    training: int = 0
    new_training: int = 0
    new_genuine: int = 0
    eval: int = 0
    eval_genuine: int = 0
    #: `training_hash` of the training Alerts: which Alerts a head was fitted
    #: on, published without naming one.
    training_hash: str = ""

    def __str__(self) -> str:
        return "\n".join([
            f"training Alerts: {self.training} ({self.new_training} new, "
            f"{self.new_genuine} of them Genuine)",
            f"Eval Alerts: {self.eval} ({self.eval_genuine} Genuine)",
            f"left out: {self.untriaged} untriaged, {self.other_pose} from "
            f"another pose backbone",
            f"training set SHA-256: {self.training_hash}",
        ])


@dataclass(frozen=True)
class Verdict:
    """One week's result. `str()` is the run summary: it is published, so it
    names no Alert and no time."""

    kind: str
    reason: str
    report: Report

    def __str__(self) -> str:
        return f"Refit {self.kind}: {self.reason}\n{self.report}"


def is_eval(event_id: str) -> bool:
    """Whether the Alert is an Eval Alert: SHA-256 of its Event ID is 0 mod 4.

    A quarter of Alerts, and the same quarter every week, so no Refit ever
    trains on an Alert a Promotion Gate judges on.
    """
    return int(hashlib.sha256(event_id.encode()).hexdigest(), 16) % 4 == 0


def training_hash(event_ids) -> str:
    """SHA-256 of the sorted Event IDs, one per line."""
    return hashlib.sha256("\n".join(sorted(event_ids)).encode()).hexdigest()


def _alerts(folder: Path, pose: str) -> tuple[list[Alert], int]:
    """Every triaged Keypoint File in `folder` whose keypoints are `pose`'s,
    in a fixed order, and how many were another pose backbone's."""
    alerts, other_pose = [], 0
    for outcome in OUTCOMES:
        for path in sorted((folder / LABELLED / outcome).rglob("*.npz")):
            f = keypoint_file.load(path)
            if f.pose != pose:
                other_pose += 1
                continue
            alerts.append(Alert(event_id=path.stem, outcome=outcome,
                                raised_at=f.frames[f.trigger].t, path=path))
    return alerts, other_pose


def run(folder: str | Path, decision: FallDecision,
        fitted_at: datetime | None) -> Verdict:
    """This week's Refit of `decision`'s Camera Head, as a verdict.

    `fitted_at` is when the deployed head was fitted, the last Promotion: the
    Alerts raised after it are this week's new data. None for a head that
    records no fit date, which was fitted on the Commissioning Recording
    alone, so every triaged Alert is new to it.
    """
    if fitted_at is not None and fitted_at.tzinfo is None:
        raise ValueError("fitted_at must carry its time zone; a naive time "
                         "would be read as the runner's local time")
    since = float("-inf") if fitted_at is None else fitted_at.timestamp()
    folder = Path(folder)
    _check_pinned(folder)
    alerts, other_pose = _alerts(folder, decision.pose)
    evals = [a for a in alerts if is_eval(a.event_id)]
    training = [a for a in alerts if not is_eval(a.event_id)]
    new = [a for a in training if a.raised_at > since]
    report = Report(untriaged=len(list((folder / INCOMING).rglob("*.npz"))),
                    other_pose=other_pose,
                    training=len(training), new_training=len(new),
                    new_genuine=sum(a.outcome == GENUINE for a in new),
                    eval=len(evals),
                    eval_genuine=sum(a.outcome == GENUINE for a in evals),
                    training_hash=training_hash(a.event_id for a in training))
    if report.new_training < MIN_NEW:
        return Verdict(SKIPPED, f"{report.new_training} new training Alerts "
                                f"since the deployed head was fitted, fewer "
                                f"than {MIN_NEW}", report)
    if not report.new_genuine:
        return Verdict(SKIPPED, f"no new Genuine training Alert among the "
                                f"{report.new_training} since the deployed "
                                f"head was fitted", report)
    return Verdict(NOT_YET_IMPLEMENTED, "enough new data; fitting and the "
                                        "Promotion Gate are not built yet", report)
