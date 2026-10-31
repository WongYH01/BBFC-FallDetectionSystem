"""The weekly Refit: one entry point from a mirrored bucket to a verdict.

`run(folder, decision)` is the whole Refit. The workflow around it
mirrors the bucket into `folder`, calls it, and acts on what comes back; this
module does no network I/O, so everything it decides is tested on a temporary
folder.

`folder` is laid out like the bucket:

    keypoints/incoming/<room>/<event id>.npz          untriaged; never used
    keypoints/labelled/genuine/<room>/<event id>.npz  triaged, by Outcome
    keypoints/labelled/false_alarm/<room>/<event id>.npz
    keypoints/commissioning/{fall,adl}/<clip>.npy     the Commissioning Recording
    keypoints/heldout/{fall,adl}/<clip>.npz           the held-out picam session
    pinned.sha256                                     the manifest, from git

The two pinned prefixes are uploaded once and never change; the manifest
(`sha256sum` format, paths relative to `folder`) pins every object under them,
and anything that does not match it is an error, not a verdict, so a changed
base set cannot silently move the Promotion Gate. The manifest is what
`sha256sum` prints, run from `folder` over the two prefixes. Each pinned clip
is filed under whether it is a fall. The Commissioning Recording is the
largest-box keypoint caches the shipped Camera Head was fitted on, one
`(frames, 17, 3)` array per clip as `data/keypoints/<pose>/` holds them; the
held-out session is Keypoint Files, because the Promotion Gate replays every
tracked person.

An Alert's Event ID is its Keypoint File's object name without `.npz`, so
filing it under its Outcome does not move it across the eval split. Untriaged
files, files whose `pose` is not the deployed backbone's, and files that are
not a Keypoint File this reader knows (another extension, a newer format
version) are left out and counted.

"New" means raised after the deployed Camera Head's fit date, by the trigger
frame's capture time. The head records its fit date and the file its trigger
time, so the loop keeps no state of its own. An Alert raised before the fit
date but triaged after it is trained on, but does not count towards this
week's new data.

A week that is not skipped fits a Candidate Head and judges it:

1. Each training Alert's windows are cut as the live camera cut them, one
   stream per subject, short of the windows live keeps scoring on a lost
   subject whose alarm is up (`_windows` says why), and labelled as ADR 0002
   says. In a Genuine Alert only
   `raised_by` is labelled: its windows ending from `FALL_FROM` to `FALL_TO`
   seconds of the trigger frame are Fall, and those ending more than
   `ADL_BEFORE` seconds before it are ADL; the rest, the time spent lying
   after the fall included, are left out. In a False Alarm every subject's
   windows are ADL. No Eval Alert's windows are cut.
2. The Candidate Head is fitted with the camera-head fitting recipe
   (`fallcore.head_fitting`) the deployed head was: on the deployed
   Checkpoint's pooled embeddings, with segment labels and the synthetic
   nobody-visible windows on the Commissioning Recording, for the deployed
   pose backbone. The Alerts' windows are added to that.
3. The Promotion Gate judges it against the deployed Camera Head on the
   held-out session and the Eval Alerts (`fallcore.promotion_gate`).

The verdict is *promoted*, with the Candidate Head, which records its pose
backbone, its fit date, the training set's hash and the gate's report; or
*gate failed*, with the gate's numbers.

Everything published -- the verdict's `str()`, which the workflow writes to
the run summary, and the gate report the head carries -- is counts and the
training set's hash, never an Event ID or a time (ADR 0002).
"""
from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

from . import calibrate, head_fitting, keypoint_file, promotion_gate
from . import config as cfg
from .calibrate import ProbeHead
from .fall_decision import FallDecision
from .keypoint_file import KeypointFile
from .promotion_gate import Clip, GateReport

#: The objects every Refit fits and judges on, uploaded once.
COMMISSIONING = "keypoints/commissioning"
HELDOUT = "keypoints/heldout"
PINNED = (COMMISSIONING, HELDOUT)
#: Where the manifest for `PINNED` sits in the folder.
MANIFEST = "pinned.sha256"
LABELLED = "keypoints/labelled"
INCOMING = "keypoints/incoming"
#: The folders a pinned clip is filed under, by whether it is a fall.
FALL, ADL = "fall", "adl"


class Outcome(Enum):
    """A triaged Alert's Outcome, valued as the notification service names
    its folder under `LABELLED`."""

    GENUINE = "genuine"
    FALSE_ALARM = "false_alarm"


#: A week with fewer new triaged training Alerts than this is skipped: the
#: Candidate Head would be fitted on nearly the deployed Camera Head's data,
#: and the gate would be measuring noise.
MIN_NEW = 5

#: Seconds from a Genuine Alert's trigger frame to the end of one of
#: `raised_by`'s windows: from `FALL_FROM` to `FALL_TO`, both included, the
#: window is Fall, the descent the alarm latched on; ending more than
#: `ADL_BEFORE` before it, it is ADL, the upright activity before the fall.
#: The windows between are neither, and so are the later ones, lying still
#: on the floor, which a head must not learn is a fall (ADR 0002).
FALL_FROM, FALL_TO, ADL_BEFORE = -2.0, 3.0, -6.0
#: Copies of each nobody-visible window flavour, as the shipped head was
#: fitted (`fit_calibration_head.py --empty-negatives 10`).
EMPTY_NEGATIVES = 10


class Kind(Enum):
    """What a week's Refit came to, valued as the run summary names it."""

    SKIPPED = "skipped"
    PROMOTED = "promoted"
    GATE_FAILED = "gate failed"


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


def _pinned_clips(folder: Path, prefix: str, suffix: str
                  ) -> list[tuple[Path, bool]]:
    """Every pinned clip under `prefix`, in a fixed order, with whether it is
    a fall. Anything not filed as `{fall,adl}/<clip><suffix>` is an error:
    the base set is not what this Refit knows how to read."""
    clips = []
    for path in sorted((folder / prefix).rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(folder / prefix)
        if len(rel.parts) != 2 or rel.parts[0] not in (FALL, ADL) \
                or path.suffix != suffix:
            raise PinnedDataError(f"{prefix}/{rel.as_posix()} is not a "
                                  f"{suffix} clip under {FALL}/ or {ADL}/")
        clips.append((path, rel.parts[0] == FALL))
    return clips


@dataclass(frozen=True)
class Alert:
    """One triaged Alert's Keypoint File, as the Refit sorts it."""

    event_id: str
    outcome: Outcome
    file: KeypointFile

    @property
    def raised_at(self) -> float:
        """When the alarm latched, unix seconds."""
        return self.file.raised_at


@dataclass(frozen=True)
class Report:
    """What a Refit used, as counts only."""

    #: Files left out: never triaged; another pose backbone's; or not a
    #: Keypoint File this reader knows.
    untriaged: int
    other_pose: int
    unreadable: int
    training: int
    new_training: int
    new_genuine: int
    eval: int
    eval_genuine: int
    #: `training_hash` of the training Alerts: which Alerts a Candidate Head
    #: was fitted on, published without naming one.
    training_hash: str

    def __str__(self) -> str:
        return "\n".join([
            f"training Alerts: {self.training} ({self.new_training} new, "
            f"{self.new_genuine} of them Genuine)",
            f"Eval Alerts: {self.eval} ({self.eval_genuine} Genuine)",
            f"left out: {self.untriaged} untriaged, {self.other_pose} from "
            f"another pose backbone, {self.unreadable} unreadable",
            f"training set SHA-256: {self.training_hash}",
        ])


@dataclass(frozen=True)
class Fit:
    """The windows a Candidate Head was fitted on, as counts."""

    #: The training Alerts' labelled windows.
    fall: int
    adl: int
    #: The Commissioning Recording's, its nobody-visible windows included.
    commissioning: int

    def __str__(self) -> str:
        return (f"Candidate Head fitted on {self.fall} Fall and {self.adl} "
                f"ADL windows of the training Alerts, and "
                f"{self.commissioning} of the Commissioning Recording")


@dataclass(frozen=True)
class Verdict:
    """One week's result. `str()` is the run summary: it is published, so it
    names no Alert and no time.

    A week that is not skipped carries what was fitted and the gate's report
    on it; a promoted one carries the Candidate Head too."""

    kind: Kind
    reason: str
    report: Report
    fit: Fit | None = None
    gate: GateReport | None = None
    head: ProbeHead | None = None

    def __str__(self) -> str:
        parts = [f"Refit {self.kind.value}: {self.reason}", str(self.report)]
        if self.fit is not None:
            parts.append(str(self.fit))
        if self.gate is not None:
            parts.append(str(self.gate))
        return "\n".join(parts)


def is_eval(event_id: str) -> bool:
    """Whether the Alert is an Eval Alert: SHA-256 of its Event ID is 0 mod 4.

    A quarter of Alerts, and the same quarter every week, so no Refit ever
    trains on an Alert a Promotion Gate judges on.
    """
    return int(hashlib.sha256(event_id.encode()).hexdigest(), 16) % 4 == 0


def training_hash(event_ids: Iterable[str]) -> str:
    """SHA-256 of the sorted Event IDs, one per line."""
    return hashlib.sha256("\n".join(sorted(event_ids)).encode()).hexdigest()


def _alerts(folder: Path, pose: str) -> tuple[list[Alert], int, int]:
    """Every triaged Keypoint File in `folder` whose keypoints are `pose`'s,
    in a fixed order, and how many files were another pose backbone's or
    could not be read."""
    alerts, other_pose, unreadable = [], 0, 0
    for outcome in Outcome:
        for path in sorted((folder / LABELLED / outcome.value).rglob("*")):
            if not path.is_file():
                continue
            if path.suffix != ".npz":
                unreadable += 1
                continue
            try:
                kf = keypoint_file.load(path)
            except ValueError:
                unreadable += 1
                continue
            if kf.pose != pose:
                other_pose += 1
                continue
            alerts.append(Alert(event_id=path.stem, outcome=outcome,
                                file=kf))
    return alerts, other_pose, unreadable


def _aware(when: datetime, what: str) -> datetime:
    if when.tzinfo is None:
        raise ValueError(f"{what} must carry its time zone; a naive time "
                         f"would be read as the runner's local time")
    return when


def _fitted_at(head: ProbeHead) -> datetime | None:
    """When `head` was fitted, the last Promotion; None for a head that
    records no fit date."""
    if not head.fitted_at:
        return None
    return _aware(datetime.fromisoformat(head.fitted_at),
                  "the deployed Camera Head's fit date")


class _Checkpoint:
    """The deployed Checkpoint, loaded: the embedding a Candidate Head is
    fitted on, and the classifier the camera runs with a head on it.

    `run` takes anything with these four members, which is how the tests run
    a Refit without weights.
    """

    def __init__(self, path: Path):
        from .train import load_checkpoint

        self.path = Path(path)
        self._model, self._model_cfg, self._train_cfg = (
            load_checkpoint(self.path, device="cpu")[:3])
        self.features = self._model_cfg.features
        self.seq_len = self._train_cfg.seq_len

    def embed(self, windows: np.ndarray) -> np.ndarray:
        """Pooled embeddings of raw `(n, T, 17, 3)` windows."""
        return head_fitting.embed_windows(self._model, self._model_cfg,
                                          windows)

    def classifier(self, head: ProbeHead):
        """The classifier the camera runs: this Checkpoint with `head`."""
        from .stream import EnsembleClassifier

        return EnsembleClassifier([self.path], heads=[head])


def _commissioning(folder: Path, checkpoint, pose: str
                   ) -> tuple[np.ndarray, np.ndarray]:
    """The Commissioning Recording's windows as the shipped Camera Head was
    fitted on them, `(embeddings, labels)`: every window at the trained
    sampling, segment labels on the fall clips, and the nobody-visible
    windows, cut from its own ADL clips."""
    import pandas as pd

    from .data import prepare_sequence, sliding_starts, take_window

    clips = _pinned_clips(folder, COMMISSIONING, ".npy")
    keypoints = {path.stem: np.load(path) for path, _ in clips}

    def load(_pose, video_id):
        return keypoints[video_id]

    fs, seq_len = cfg.PREPROCESS.frame_stride, checkpoint.seq_len
    rows, windows = [], []
    for path, fall in clips:
        seq = prepare_sequence(keypoints[path.stem], None, fs)
        for start in sliding_starts(len(seq), seq_len, cfg.EVAL.sliding_stride):
            rows.append((path.stem, start, int(fall)))
            windows.append(take_window(seq, seq_len, start=start))
    table = pd.DataFrame(rows, columns=["video_id", "start", "label"])
    keep, labels = head_fitting.segment_labels(table, pose, seq_len, load=load)
    negatives = head_fitting.NobodyNegatives(
        seq_len, checkpoint.embed, copies=EMPTY_NEGATIVES, pose=pose,
        load=load)
    embeddings, labels, _ = head_fitting.fold_fitting_set(
        np.ones(len(table), dtype=bool), table,
        checkpoint.embed(np.stack(windows)), labels, keep,
        negatives=negatives)
    return embeddings, labels


def _row(frame, subject: int) -> np.ndarray | None:
    """`subject`'s row in `frame`, the largest box's when it has two (a
    tracker briefly holding an old and a new id, as `SubjectStreams` takes
    it), or None when it is not in view. Subject -1, a file decided on the
    largest box, is the largest box."""
    rows = [p for p in frame.people if p.subject == subject]
    if not rows:
        return None
    return max(rows, key=lambda p: p.area).keypoints


def _windows(kf: KeypointFile, subject: int, decision: FallDecision,
             seq_len: int) -> list[tuple[np.ndarray, float]]:
    """Every window the camera scored on `subject`, rebuilt from the file,
    each with when it ends, in seconds from the trigger frame.

    The stream is the live one (`fallcore.stream.SubjectStreams`): from the
    subject's first row, the empty row on frames they were not seen, cut by
    the decision's buffer, and ended once they have been unseen for a
    window. Windows that ended before the file began cannot be rebuilt.

    Unlike the live cut, it ends there even while the subject's alarm is up,
    which live keeps them for up to a second window. Deliberately: those
    windows are mostly empty rows, and labelling them Fall after a Genuine
    trigger would teach the head that losing the person is a fall -- the
    shortcut the Checkpoint already takes.
    """
    from .extract import empty_row
    from .stream import KeypointBuffer

    buffer = KeypointBuffer(seq_len, decision.frame_stride, decision.step)
    out, started, unseen = [], False, 0
    for frame in kf.frames:
        row = _row(frame, subject)
        if row is not None:
            started, unseen = True, 0
        elif not started:
            continue
        else:
            unseen += 1
            if unseen > buffer.frames_per_window:
                break
            row = empty_row()
        if buffer.add(row):
            end = round(frame.t - kf.raised_at, 3)
            out.append((buffer.window(), end))
    return out


def _labelled(alert: Alert, decision: FallDecision, seq_len: int
              ) -> list[tuple[np.ndarray, bool]]:
    """A training Alert's labelled windows, each with whether it is a fall
    (ADR 0002): in a Genuine Alert `raised_by`'s alone, Fall around the
    trigger and ADL well before it; in a False Alarm everyone's, all ADL."""
    kf = alert.file
    if alert.outcome is Outcome.FALSE_ALARM:
        subjects = sorted({p.subject for f in kf.frames for p in f.people})
        return [(w, False) for s in subjects
                for w, _ in _windows(kf, s, decision, seq_len)]
    out = []
    for window, end in _windows(kf, kf.raised_by, decision, seq_len):
        if FALL_FROM <= end <= FALL_TO:
            out.append((window, True))
        elif end < ADL_BEFORE:
            out.append((window, False))
    return out


def _eval_clip(alert: Alert) -> Clip:
    """An Eval Alert as the Promotion Gate replays it, from its first frame.
    The gate names the raising subject by a track id of theirs."""
    kf = alert.file
    track = (None if kf.raised_by == -1 else
             next((p.track_id for f in kf.frames for p in f.people
                   if p.subject == kf.raised_by), None))
    return Clip(kf.frames, fall=alert.outcome is Outcome.GENUINE,
                raised_by=track)


def _heldout(folder: Path, pose: str) -> list[Clip]:
    clips = []
    for path, fall in _pinned_clips(folder, HELDOUT, ".npz"):
        kf = keypoint_file.load(path)
        if kf.pose != pose:
            raise PinnedDataError(f"{HELDOUT}/{path.parent.name}/{path.name} "
                                  f"holds {kf.pose or 'unnamed'} keypoints, "
                                  f"not {pose}")
        clips.append(Clip(kf.frames, fall=fall))
    return clips


def _fit(folder: Path, decision: FallDecision, checkpoint,
         training: list[Alert], now: datetime) -> tuple[ProbeHead, Fit]:
    """The Candidate Head, fitted on the Commissioning Recording plus the
    training Alerts, and what it was fitted on."""
    seq_len = checkpoint.seq_len
    base, base_labels = _commissioning(folder, checkpoint, decision.pose)
    labelled = [lw for a in training for lw in _labelled(a, decision, seq_len)]
    embeddings, labels = [base], [base_labels]
    if labelled:
        embeddings.append(checkpoint.embed(np.stack([w for w, _ in labelled])))
        labels.append(np.array([fall for _, fall in labelled], dtype=int))
    head = calibrate.fit(np.concatenate(embeddings), np.concatenate(labels),
                         backbone=decision.checkpoints[0].name,
                         features=checkpoint.features, seq_len=seq_len,
                         pose=decision.pose)
    head = dataclasses.replace(
        head, fitted_at=now.isoformat(),
        training_hash=training_hash(a.event_id for a in training))
    fall = sum(f for _, f in labelled)
    return head, Fit(fall=fall, adl=len(labelled) - fall,
                     commissioning=len(base))


def run(folder: str | Path, decision: FallDecision, checkpoint=None,
        now: datetime | None = None) -> Verdict:
    """This week's Refit of `decision`'s Camera Head, as a verdict.

    The Alerts raised after the deployed Camera Head's recorded fit date, the
    last Promotion, are this week's new data. A head that records no fit
    date was fitted by hand on the Commissioning Recording alone, so every
    triaged Alert is new to it.

    `checkpoint` is the deployed Checkpoint, loaded (default: loaded from
    `decision` when a week is not skipped, so a skipped week needs no
    weights). `now` is when this Refit runs, the time zone included
    (default: the clock); the Candidate Head records it as its fit date.
    """
    now = _aware(now or datetime.now(timezone.utc), "now")
    if len(decision.heads) != 1 or decision.heads[0] is None:
        raise ValueError("a Refit refits one Camera Head on one Checkpoint, "
                         f"not {decision}")
    deployed = ProbeHead.load(decision.heads[0])
    fitted_at = _fitted_at(deployed)
    since = float("-inf") if fitted_at is None else fitted_at.timestamp()
    folder = Path(folder)
    _check_pinned(folder)
    alerts, other_pose, unreadable = _alerts(folder, decision.pose)
    evals: list[Alert] = []
    training: list[Alert] = []
    for alert in alerts:
        (evals if is_eval(alert.event_id) else training).append(alert)
    new = [a for a in training if a.raised_at > since]
    untriaged = sum(p.is_file() for p in (folder / INCOMING).rglob("*"))
    report = Report(
        untriaged=untriaged, other_pose=other_pose, unreadable=unreadable,
        training=len(training), new_training=len(new),
        new_genuine=sum(a.outcome is Outcome.GENUINE for a in new),
        eval=len(evals),
        eval_genuine=sum(a.outcome is Outcome.GENUINE for a in evals),
        training_hash=training_hash(a.event_id for a in training))
    if report.new_training < MIN_NEW:
        return Verdict(Kind.SKIPPED,
                       f"{report.new_training} new training Alerts since the "
                       f"deployed Camera Head was fitted, fewer than {MIN_NEW}",
                       report)
    if not report.new_genuine:
        return Verdict(Kind.SKIPPED,
                       f"no new Genuine training Alert among the "
                       f"{report.new_training} since the deployed Camera Head "
                       f"was fitted", report)

    if checkpoint is None:
        checkpoint = _Checkpoint(decision.checkpoints[0])
    if deployed.n_extra:
        raise ValueError(f"the deployed Camera Head takes {deployed.extra} "
                         f"alongside the embedding; a Refit fits on the "
                         f"embedding alone")
    calibrate.validate(deployed, decision.checkpoints[0], checkpoint.features,
                       checkpoint.seq_len, pose=decision.pose)
    candidate, fit = _fit(folder, decision, checkpoint, training, now)
    gate = promotion_gate.judge(
        decision, checkpoint.classifier(deployed),
        checkpoint.classifier(candidate), _heldout(folder, decision.pose),
        [_eval_clip(a) for a in evals], seq_len=checkpoint.seq_len)
    if not gate.passed:
        return Verdict(Kind.GATE_FAILED, gate.reason, report, fit, gate)
    candidate = dataclasses.replace(candidate, gate=str(gate))
    return Verdict(Kind.PROMOTED, gate.reason, report, fit, gate, candidate)
