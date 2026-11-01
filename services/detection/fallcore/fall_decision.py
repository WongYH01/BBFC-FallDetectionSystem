"""The Fall Decision demo-cam v2 deploys, defined once.

A Fall Decision is the calibrated pair (one checkpoint plus the room head fitted
on its embedding), the rule that turns its window scores into an alarm (rolling
buffer, threshold, hysteresis, descent gate), the sampling the windows are
cut with, and whose keypoints are decided on (every tracked person, or the
largest box). The live detector, `replay.py` and picam scoring all build their
streams from one of these, so the rule a replay or a rendered clip scores is
the rule the camera runs -- not a copy kept in step by hand. picam scoring reads
keypoint caches that hold the largest box only, so it scores `build`'s single
stream whatever `subjects` says.

Deliberately light: importing this module pulls in only `fallcore.config`, so a
test or a CLI can read and compare decisions without torch. `build` imports the
stream, and with it the model code, when a stream is actually wanted.
"""
from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from dataclasses import dataclass
from pathlib import Path

from . import config as cfg

#: The one calibrated pair the camera runs, and the keypoints its head was
#: fitted on: the yolo26m INT8 export at 384x640 (`fallcore.pose_backbone`),
#: whose keypoint cache is `data/keypoints/yolo26m_r640_ortint8rr/`. A head
#: reads one pose backbone's keypoints only, so the three are one choice.
#: `scripts/fetch_weights.py` pins these files by hash; a test keeps the lists
#: in agreement.
DEPLOYED_CHECKPOINT = "augnone_ms_coords_hn_s99.pt"
DEPLOYED_HEAD = "probe_augnone_ms_coords_hn_s99_yolo26m-int8.npz"
DEPLOYED_POSE = "yolo26m_r640_ortint8rr"

#: The environment variables demo-cam v2 has always read, and the field each
#: one sets. Their names are a contract with every `.env` and compose file.
_RULE_VARS: dict[str, tuple[str, type]] = {
    "BUFFER_LEN":          ("buffer", int),
    "FALL_THRESHOLD":      ("threshold", float),
    "FALL_CLEAR_BELOW":    ("clear_below", float),
    "FALL_CLEAR_WINDOWS":  ("clear_windows", int),
    "FALL_MIN_DESCENT":    ("min_descent", float),
    "STREAM_FRAME_STRIDE": ("frame_stride", int),
    "FALL_SUBJECTS":       ("subjects", str),
    "FALL_RECONNECT_SEC":  ("reconnect_sec", float),
    "FALL_RECONNECT_DIST": ("reconnect_dist", float),
}
#: Variables that once picked the pose model, checkpoints or head one by one,
#: which let a head run on keypoints it was never fitted on. Refused.
RETIRED_VARS = ("POSE_SCALE", "POSE_MODEL", "POSE_IMGSZ", "ENSEMBLE_CKPTS",
                "CALIBRATION_HEAD")
_RULE_FIELDS = ("buffer", "threshold", "clear_below", "clear_windows",
                "min_descent", "frame_stride", "step", "subjects",
                "reconnect_sec", "reconnect_dist")

#: How a camera picks whose keypoints it decides on: every tracked person, each
#: with their own stream (`fallcore.stream.SubjectStreams`), or the largest box
#: alone, which is the paper's rule and every result before 2026-10-16.
SUBJECT_RULES = ("tracked", "largest")


def _out_of_range(field: str, value) -> str | None:
    """Why `value` cannot be `field`, or None when it can."""
    if field in ("buffer", "clear_windows", "frame_stride", "step") and value < 1:
        return "must be at least 1"
    if field in ("threshold", "clear_below") and not 0.0 <= value <= 1.0:
        return "must be a probability between 0 and 1"
    if field == "min_descent" and value < 0.0:
        return "must be 0 (gate off) or more"
    if field == "subjects" and value not in SUBJECT_RULES:
        return f"must be one of {', '.join(SUBJECT_RULES)}"
    if field in ("reconnect_sec", "reconnect_dist") and value < 0.0:
        return "must be 0 (never reconnect) or more"
    return None


@dataclass(frozen=True)
class FallDecision:
    checkpoints: tuple[Path, ...] = (cfg.CKPT_DIR / DEPLOYED_CHECKPOINT,)
    heads: tuple[Path | None, ...] = (cfg.CKPT_DIR / DEPLOYED_HEAD,)
    # The keypoints the heads read, named as their keypoint cache. A head that
    # records another pose backbone is refused in `build`.
    pose: str = DEPLOYED_POSE
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
    # Whose keypoints are decided on (`SUBJECT_RULES`), and, for "tracked",
    # how a new tracker id is joined to a subject that was just lost: last
    # seen at most `reconnect_sec` ago, with its box centre at most
    # `reconnect_dist` (fractions of the frame) from the new one. Picked, not
    # tuned: they reproduce the largest-box rule's catches on picam's
    # single-person falls, see `SubjectStreams`.
    subjects: str = "tracked"
    reconnect_sec: float = 2.0
    reconnect_dist: float = 0.35

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
        rather than booting it mis-tuned. So is a variable that used to swap
        the pose model, checkpoint or head (`RETIRED_VARS`): the models are not
        a setting, and an old `.env` must not be silently ignored.
        """
        changes = {}
        for var, (field, kind) in _RULE_VARS.items():
            if var not in env:
                continue
            try:
                value = kind(env[var].strip().lower() if kind is str else env[var])
            except ValueError:
                raise ValueError(f"{var}={env[var]!r} is not "
                                 f"{'an integer' if kind is int else 'a number'}"
                                 ) from None
            problem = _out_of_range(field, value)
            if problem:
                raise ValueError(f"{var} {problem}, not {env[var]!r}")
            changes[field] = value
        retired = [var for var in RETIRED_VARS if var in env]
        if retired:
            raise ValueError(
                f"{', '.join(retired)} no longer read: the camera runs one pose "
                f"backbone and the head fitted on its keypoints "
                f"({DEPLOYED_POSE}, {DEPLOYED_HEAD}). Remove "
                f"{'it' if len(retired) == 1 else 'them'} from the environment")
        return cls(**changes)

    def replace(self, **changes) -> FallDecision:
        """This decision with the given fields changed."""
        return dataclasses.replace(self, **changes)

    def build(self, device: str = "cpu", classifier=None,
              seq_len: int | None = None):
        """An `EnsembleStream` that runs this decision.

        The window length is not a setting: it is read from the checkpoints,
        which must agree, so a checkpoint trained on a different window cannot
        be fed one cut to the training default. Each head is checked against
        its checkpoint's backbone, features and window length, and against
        `pose`, before use.

        `classifier` runs this decision's rule on an already-built classifier
        instead of `checkpoints` and `heads` -- the Promotion Gate scores a
        Candidate Head this way -- and nothing is loaded or checked. `seq_len`
        must then say the window its checkpoints were trained on.
        """
        if classifier is not None:
            if seq_len is None:
                raise ValueError("a classifier given as built needs its "
                                 "window length, seq_len")
            return self._stream([], None, seq_len, device, classifier)

        from .calibrate import ProbeHead, validate
        from .train import load_checkpoint

        lengths = set()
        for path, head in zip(self.checkpoints, self.heads):
            _model, model_cfg, train_cfg, _scale = load_checkpoint(path, device="cpu")
            lengths.add(train_cfg.seq_len)
            if head is not None:
                validate(ProbeHead.load(head), path, model_cfg.features,
                         train_cfg.seq_len, pose=self.pose)
        if len(lengths) > 1:
            raise ValueError(f"checkpoints disagree on window length: "
                             f"{sorted(lengths)}")
        return self._stream(list(self.checkpoints), list(self.heads),
                            lengths.pop(), device)

    def _stream(self, checkpoints, heads, seq_len: int, device: str,
                classifier=None):
        from .stream import EnsembleStream

        return EnsembleStream(
            checkpoints, heads=heads, buffer=self.buffer,
            threshold=self.threshold, clear_below=self.clear_below,
            clear_windows=self.clear_windows, min_descent=self.min_descent,
            seq_len=seq_len, frame_stride=self.frame_stride,
            step=self.step, device=device, classifier=classifier)

    def build_subjects(self, device: str = "cpu", classifier=None,
                       seq_len: int | None = None):
        """A `SubjectStreams` that runs this decision once per tracked subject.

        Every subject's stream is `build`'s rule on one shared classifier, so
        the models load once however many people are in view.
        """
        from .stream import SubjectStreams

        return SubjectStreams(self.build(device, classifier, seq_len),
                              reconnect_sec=self.reconnect_sec,
                              reconnect_dist=self.reconnect_dist)

    def build_camera(self, device: str = "cpu", classifier=None,
                     seq_len: int | None = None):
        """What a camera runs: `build_subjects` for "tracked", `build` for "largest"."""
        if self.subjects == "tracked":
            return self.build_subjects(device, classifier, seq_len)
        return self.build(device, classifier, seq_len)

    def __str__(self) -> str:
        models = ", ".join(
            f"{c.name} + {h.name}" if h is not None else f"{c.name} (own classifier)"
            for c, h in zip(self.checkpoints, self.heads))
        rule = " ".join(f"{f}={getattr(self, f)}" for f in _RULE_FIELDS)
        return f"{models} on {self.pose} | {rule}"
