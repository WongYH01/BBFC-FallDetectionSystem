"""The Pi-camera zero-shot set: `data/picam`.

Recorded 2026-09-08 off the `demo-cam` RTSP feed with VLC, 1280x720 at ~30 fps,
60 trimmed clips of 12 actions by 5 subjects. This is the only data in the
project that matches the deployment conditions -- same camera, same room, same
`yolo26n-pose` backbone at the same `imgsz` the demo app runs. That makes it a
deployment acceptance test rather than a benchmark: one room and one viewpoint
cannot measure cross-environment generalisation, but nothing else here measures
the thing the system will actually be pointed at.

**Subject comes from the capture timestamp, not the folder.** The session ran
one person at a time through the same action list, five times over, so the
timestamp ranges are the subjects. Deriving it that way means re-filing a clip
into a different action folder changes only its label, never its subject -- and
the folders do need re-filing from time to time.

The action names are the folder names. Two of them are subtle and worth stating
because the whole point of this set is the hard negatives:

* `lie down` -- **starts standing** and lowers to the floor under control. A
  No-Fall that ends in exactly the posture a fall ends in.
* `lie down fall` -- **starts already lying down** and then falls. A Fall with
  no standing-to-horizontal transition at all, which is precisely the cue the
  models were measured leaning on, so it is the hardest positive in the set.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibrate
from . import config as cfg

CLIPS_DIR = cfg.DATA_DIR / "picam"

# vlc-record-2026-09-08-18h20m39s-rtsp___wyhrp4_8554_cam-.mp4
_STAMP = re.compile(r"vlc-record-(\d{4})-(\d{2})-(\d{2})-(\d{2})h(\d{2})m(\d{2})s")

#: folder name -> (is_fall, note). The binary label is the deployment question:
#: should the system raise an alarm during this clip?
ACTIONS: dict[str, tuple[int, str]] = {
    "fall forward":  (1, "fall from standing"),
    "fall backward": (1, "fall from standing"),
    "fall left":     (1, "fall from standing"),
    "fall right":    (1, "fall from standing"),
    "fall sitting":  (1, "fall from a seated position"),
    "lie down fall": (1, "already lying, then falls -- no upright transition"),
    "walking":       (0, "baseline locomotion"),
    "jumping":       (0, "large vertical motion, stays upright"),
    "sit chair":     (0, "controlled descent onto a chair"),
    "sit floor":     (0, "controlled descent to the floor -- ends low"),
    "tie shoelace":  (0, "bends to floor level and returns"),
    "lie down":      (0, "controlled descent to lying -- ends horizontal"),
}

#: Every model the project has produced that is worth pointing at a camera,
#: as name -> checkpoint filename under `runs/checkpoints`. Notebook 16 and
#: `scripts/render_picam.py` both read this, so a run and a render of that run
#: cannot end up describing different checkpoints.
#:
#: Two entries can be the same trained model: notebook 02's replication and
#: notebook 10's `full / baseline` arm share features, seed, split and learning
#: rate, and training is deterministic under a fixed seed, so they come out
#: bit-identical. Both are listed because both are things a reader looks for by
#: name; notebook 16 de-duplicates by weight hash before reporting.
MODELS: dict[str, str] = {
    "paper replication": "final_yolo26n.pt",
    "Kaggle only":       "hn10_full_base_s99.pt",
    "Kaggle + hard neg": "hn10_full_hn_s99.pt",
    "coords + hard neg": "hn10_coords_hn_s99.pt",
    "OmniFall":          "omnifall_cs_full.pt",
    # The multi-scale arm: one classifier trained with yolo26n/s/m keypoint
    # caches drawn per item, no keypoint-space augmentation. Better ranking than
    # every arm above on picam (window AUC 0.887-0.899 over seeds 99/7/2024
    # against 0.871) and on CAUCAFall (F1 0.82-0.92 against 0.79), at some cost
    # on LE2I; but at threshold 0.5 its clip F1 is *lower* than `coords + hard
    # neg` -- it scores higher overall, so ship it with the room head below or
    # pick its threshold off picam. Evidence: runs/metrics/augmented_single.csv.
    "ms coords + hard neg": "augnone_ms_coords_hn_s99.pt",
}

#: Backbone + room-fitted head pairs, as (checkpoint, head npz) filenames under
#: `runs/checkpoints`. The head is fitted on the room's own clips by
#: `scripts/fit_calibration_head.py` (`fallcore.calibrate.ProbeHead`); it
#: replaces the pretrained classifier on the checkpoint's pooled embedding, so
#: inference is still one forward pass. Measured by leave-one-subject-out on
#: picam: window AUC 0.992, causal rolling b4/b8 F1 0.967 (1 FP, 1 FN) against
#: 0.836/0.889 for the best single model without it -- see
#: `runs/metrics/calibration_head.csv`.
#:
#: The head shipped here is fitted on all 60 picam clips, so scoring picam with
#: it is in-sample and its numbers are a description of the fitted set, not a
#: prediction. Quote the cross-subject figure, not this file's score.
CALIBRATED: dict[str, tuple[str, str]] = {
    "ms coords + room head": ("augnone_ms_coords_hn_s99.pt",
                              "probe_augnone_ms_coords_hn_s99.npz"),
    # The same backbone and fitting recipe, plus the three scale-free descent
    # measures appended to every embedding (fit_calibration_head.py
    # --extra-kinematics).
    #
    # **Not an improvement at three seeds.** On seed 99 alone it looked like one
    # (picam-1 leave-one-subject-out F1 0.967 -> 0.983); over seeds 99/7/2024 it
    # is a wash in the fitted room (0.961 both ways) and slightly *worse* in an
    # unseen one -- picam-2 window AUC -0.008, the same sign on all three seeds
    # (`runs/metrics/coords_vs_full.csv`). Kept because it is measured and
    # reproducible, not because it is better; the descent gate below is the
    # change that carries.
    "ms coords + kinematic head": ("augnone_ms_coords_hn_s99.pt",
                                   "probe_augnone_ms_coords_hn_s99_kin.npz"),
}

#: Body lengths per second the hips must fall somewhere inside a window before
#: that window may *raise* an alarm. It exists because the decision was being
#: re-made every second on windows holding no descent at all: on picam-2 the
#: windows that false-alarmed on `lie_down` contained 27% of the descent, and
#: two identical motionless bodies -- one that fell, one that lay down -- cannot
#: be told apart by any classifier. Latching and clearing are untouched, so a
#: fall still alarms on its descent and stays alarmed.
#:
#: 0.10 silences 88% of still windows while blocking no fall window measured on
#: either room. 0.12 removes the last two picam-2 false alarms as well, but
#: costs a `fall sitting` clip whose fastest window is 0.11 -- a fall from a
#: seated position barely moves the hips. `runs/metrics/descent_gate_sweep.csv`.
V2_MIN_DESCENT = 0.10

#: Subject boundaries, as the first capture time of each round. Taken from the
#: session timeline rather than from a gap threshold: the round 4 -> 5 gap is
#: 0.6 min, shorter than several gaps *inside* rounds, so no gap rule recovers
#: these. Edit here if the session is re-cut.
ROUND_STARTS = ["18:15:00", "18:29:00", "18:38:21", "18:44:48", "18:51:25"]
#: The later lie-down session alternates (lie down, lie down fall) per subject,
#: in the same subject order, so consecutive pairs map to subjects 1..5.
LATE_SESSION_HOUR = 19


def _stamp(name: str) -> datetime | None:
    m = _STAMP.search(name)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(g) for g in m.groups())
    return datetime(y, mo, d, h, mi, s)


def scan(clips_dir: Path | None = None) -> pd.DataFrame:
    """One row per clip: id, action, binary label, subject, and video properties."""
    import cv2

    root = Path(clips_dir or CLIPS_DIR)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found")

    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        action = d.name
        if action not in ACTIONS:
            raise KeyError(
                f"folder {action!r} is not in picam.ACTIONS -- add it with a "
                f"binary label before scanning, rather than letting it be "
                f"silently dropped")
        label, note = ACTIONS[action]
        for f in sorted(d.glob("*.mp4")):
            cap = cv2.VideoCapture(str(f))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            rows.append({
                "video_id": f"picam__{f.stem}",
                "path": str(f), "action": action, "label": label, "note": note,
                "t": _stamp(f.name), "fps": fps, "n_frames": n,
                "width": w, "height": h,
                "duration": n / fps if fps else 0.0,
            })

    man = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    man["subject"] = _subjects(man)
    return man


def _subjects(man: pd.DataFrame) -> pd.Series:
    """Subject id per clip, from capture time. 0 means "could not place"."""
    out = pd.Series(0, index=man.index, dtype=int)
    if man.t.isna().all():
        return out

    day = man.t.dropna().iloc[0].date()
    bounds = [datetime.combine(day, datetime.strptime(s, "%H:%M:%S").time())
              for s in ROUND_STARTS]

    early = man.t.dt.hour < LATE_SESSION_HOUR
    # searchsorted gives the round each timestamp falls into; anything before
    # the first boundary stays 0 rather than being forced into round 1.
    idx = np.searchsorted(bounds, man.loc[early, "t"].to_numpy(), side="right")
    out.loc[early] = idx

    # The late session is strictly alternating pairs in subject order.
    late = man.index[~early]
    for k, i in enumerate(late):
        out.loc[i] = k // 2 + 1
    return out


def posture_profile(man: pd.DataFrame, scale: str = cfg.DEFAULT_SCALE
                    ) -> pd.DataFrame:
    """Per-clip body height in frame, for spotting clips filed under the wrong action.

    Uses vertical position rather than the `demo-cam` width/height rule. That
    rule is viewpoint-dependent -- someone falling toward or away from the lens
    never makes a box wider than it is tall, and on this camera it mislabelled
    18 of the 29 fall clips as "not a fall". Height in frame has no such
    problem: whatever direction they go, a person who ends up on the floor ends
    up low in shot. Measured here it separates the two classes cleanly, falls
    ending at 0.66-0.77 against 0.33-0.47 for the ADLs.

    `y` in the cache is normalised to frame height with 0 at the top, so a large
    `y_end` means low in frame. This is geometry over the cached keypoints, not
    a second opinion from the classifier being evaluated.

    `x_end` is the horizontal twin, and it is here to be *disproved*: if falls
    separate on x as well as on y, the rule is reading where in the room each
    action was staged rather than anything about the body, and the whole result
    is a confound. Notebook 16 runs that check.
    """
    from .extract import keypoints_path

    rows = []
    for _, v in man.iterrows():
        p = keypoints_path(scale, v.video_id)
        if not p.exists():
            continue
        kp = np.load(p)
        seen = kp[:, :, 2].sum(axis=1) > 0
        if seen.sum() < 5:
            rows.append({"video_id": v.video_id, "action": v.action,
                         "subject": int(v.subject), "detected": float(seen.mean()),
                         "y_start": np.nan, "y_end": np.nan, "drop": np.nan,
                         "x_end": np.nan})
            continue
        conf = kp[:, :, 2] > 0
        y = np.where(conf, kp[:, :, 0], np.nan)
        x = np.where(conf, kp[:, :, 1], np.nan)
        # A frame where no keypoint cleared the confidence bar is all-NaN, and
        # nanmean warns on it. NaN is the right answer there, so silence the
        # warning rather than special-casing the frame.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Mean of empty slice")
            centroid = np.nanmean(y, axis=1)[seen]
            centroid_x = np.nanmean(x, axis=1)[seen]
        k = max(1, len(centroid) // 5)          # first and last fifth
        y0 = float(np.nanmean(centroid[:k]))
        y1 = float(np.nanmean(centroid[-k:]))
        rows.append({"video_id": v.video_id, "action": v.action,
                     "subject": int(v.subject), "detected": float(seen.mean()),
                     "y_start": y0, "y_end": y1, "drop": y1 - y0,
                     "x_end": float(np.nanmean(centroid_x[-k:]))})
    return pd.DataFrame(rows)


def suspect_clips(profile: pd.DataFrame, man: pd.DataFrame | None = None
                  ) -> pd.DataFrame:
    """Clips whose end posture sits on the wrong side of the class boundary.

    The threshold is the midpoint between the two class means rather than a
    fixed constant, so it travels if the camera is re-sited. This flags
    candidates to re-watch and decides nothing: `sit floor` legitimately ends
    low, and a fall away from the camera legitimately ends less low.
    """
    d = profile.dropna(subset=["y_end"]).copy()
    is_fall = d.action.map(lambda a: ACTIONS.get(a, (0, ""))[0] == 1)
    thr = (d.loc[is_fall, "y_end"].mean() + d.loc[~is_fall, "y_end"].mean()) / 2

    wrong = np.where(is_fall, d.y_end < thr, d.y_end > thr)
    out = d[wrong].copy()
    out["threshold"] = round(float(thr), 3)
    out["issue"] = np.where(is_fall[wrong], "fall ends high", "no-fall ends low")
    if man is not None:
        out = out.merge(man[["video_id", "path"]], on="video_id", how="left")
    return out.sort_values(["issue", "y_end"])


def build_clips(man: pd.DataFrame, scale: str = cfg.DEFAULT_SCALE,
                seq_len: int = cfg.FINAL_TRAIN.seq_len,
                stride: int = cfg.EVAL.sliding_stride,
                frame_stride: int | None = None) -> pd.DataFrame:
    """Sliding windows over each clip, every window carrying the clip's label.

    These are trimmed single-action clips, so unlike the densely annotated
    corpora there is no within-clip boundary to respect: the action is the whole
    clip. That makes the per-window label simply the clip label, and leaves the
    per-clip decision -- did the system alarm at any point -- as the metric that
    actually matches deployment.
    """
    from .data import sliding_starts
    from .extract import keypoints_path

    fs = cfg.PREPROCESS.frame_stride if frame_stride is None else frame_stride
    rows = []
    for _, v in man.iterrows():
        if not keypoints_path(scale, v.video_id).exists():
            continue
        n_seq = -(-int(v.n_frames) // fs)
        for s in sliding_starts(n_seq, seq_len, stride):
            rows.append({"video_id": v.video_id, "start": int(s),
                         "label": int(v.label), "action": v.action,
                         "subject": int(v.subject)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rendering one clip with a detector's decision drawn over it
#
# Shared by notebook 19 and scripts/render_picam.py, so a video rendered from
# either one comes out of the same scoring and the same file layout.
# ---------------------------------------------------------------------------

#: demo-cam v2's decision, by registry name: the mean P(fall) of these four,
#: smoothed over the last `buffer` windows and latched. Keep in step with
#: `_CKPT_NAMES` in demo-cam/v2/detector.py -- the four files must be the same.
ENSEMBLES: dict[str, tuple[str, ...]] = {
    "v2 ensemble": ("paper replication", "Kaggle + hard neg",
                    "coords + hard neg", "OmniFall"),
}

#: demo-cam v2's defaults (BUFFER_LEN, FALL_CLEAR_BELOW, FALL_CLEAR_WINDOWS).
V2_BUFFER, V2_CLEAR_BELOW, V2_CLEAR_WINDOWS = 4, 0.2, 2


def resolve_checkpoint(model: str | Path) -> Path:
    """A checkpoint path from a `MODELS` name, a file in runs/checkpoints, or a path."""
    if isinstance(model, str) and model in MODELS:
        return cfg.CKPT_DIR / MODELS[model]
    p = Path(model)
    for cand in (cfg.CKPT_DIR / p.name if p.parent == Path(".") else None, p):
        if cand is not None and cand.exists():
            return cand
    raise FileNotFoundError(
        f"{model!r} is not a picam.MODELS name ({', '.join(MODELS)}), an "
        f"ensemble ({', '.join(ENSEMBLES)}), a file in {cfg.CKPT_DIR}, or an "
        f"existing path")


def resolve_head(head: str | Path) -> Path:
    """A fitted head's path from a `CALIBRATED` tuple, a file in runs/checkpoints, or a path."""
    p = Path(head)
    for cand in (cfg.CKPT_DIR / p.name if p.parent == Path(".") else None, p):
        if cand is not None and cand.exists():
            return cand
    raise FileNotFoundError(
        f"head {head!r} is not a file in {cfg.CKPT_DIR} or an existing path")


def model_slug(model: str | Path) -> str:
    """Folder name for a model's renders; registry names keep their old slug."""
    named = isinstance(model, str) and (model in MODELS or model in ENSEMBLES
                                        or model in CALIBRATED)
    name = model if named else Path(model).stem
    return name.replace(" ", "_").replace("+", "and")


@dataclass
class Detector:
    """What a clip is scored with: one checkpoint, an ensemble, or one with a head.

    A single checkpoint uses notebook 16's rule -- alarm if any window reaches
    the threshold. The ensemble and the calibrated pair use v2's, and run through
    the very class the live detector does (`fallcore.stream.EnsembleStream`), fed
    the cached keypoints one frame at a time, so their windows, rolling mean and
    latch are the live ones rather than an imitation of them.
    """
    name: str
    scale: str
    features: str
    checkpoints: list[Path]
    model: object = None
    model_cfg: object = None
    train_cfg: object = None
    stream: object = None
    heads: list[Path] = field(default_factory=list)

    @property
    def is_ensemble(self) -> bool:
        return self.stream is not None

    @property
    def is_calibrated(self) -> bool:
        return bool(self.heads)

    @property
    def rule(self) -> str:
        if not self.is_ensemble:
            return "any window >= threshold"
        d = self.stream.decision
        who = (f"1 model + room head [{self.heads[0].name}], "
               if self.is_calibrated else f"mean of {len(self.checkpoints)} models, ")
        return (f"{who}rolling mean of {d.buffer} windows >= threshold, "
                f"latched until below {d.clear_below} for {d.clear_windows} windows")


def load_detector(model: str | Path, device: str = "cpu",
                  threshold: float = cfg.EVAL.threshold,
                  buffer: int = V2_BUFFER,
                  clear_below: float = V2_CLEAR_BELOW,
                  clear_windows: int = V2_CLEAR_WINDOWS,
                  min_descent: float = 0.0) -> Detector:
    """A `Detector` from a registry name, a `CALIBRATED` name, or a checkpoint.

    The ensemble's members must agree on backbone and window length, since they
    score the same buffered keypoints; mixing scales is refused rather than
    silently fed the wrong pose. A calibrated pair is checked the same way: the
    head records the backbone it was fitted against, and a mismatch is an error
    rather than a silently misapplied head.
    """
    from . import train as T

    if isinstance(model, str) and model in CALIBRATED:
        from .calibrate import ProbeHead
        from .stream import EnsembleStream

        ckpt_name, head_name = CALIBRATED[model]
        path = resolve_checkpoint(ckpt_name)
        head_path = resolve_head(head_name)
        head = ProbeHead.load(head_path)
        m, mc, tc, scale = T.load_checkpoint(path, device="cpu")
        calibrate.validate(head, path, mc.features, tc.seq_len)
        stream = EnsembleStream(
            [path], heads=[head], buffer=buffer, threshold=threshold,
            clear_below=clear_below, clear_windows=clear_windows,
            min_descent=min_descent,
            seq_len=tc.seq_len, frame_stride=cfg.PREPROCESS.frame_stride,
            step=cfg.EVAL.sliding_stride, device=device)
        return Detector(name=model, scale=scale,
                        features=f"{mc.features} -> room head",
                        checkpoints=[path], train_cfg=tc, stream=stream,
                        heads=[head_path])

    if isinstance(model, str) and model in ENSEMBLES:
        from .stream import EnsembleStream

        paths = [resolve_checkpoint(m) for m in ENSEMBLES[model]]
        cfgs = [T.load_checkpoint(p, device="cpu") for p in paths]
        scales = {c[3] for c in cfgs}
        lengths = {c[2].seq_len for c in cfgs}
        if len(scales) > 1 or len(lengths) > 1:
            raise ValueError(f"{model!r} members disagree: scales {scales}, "
                             f"window lengths {lengths}")
        stream = EnsembleStream(
            paths, buffer=buffer, threshold=threshold, clear_below=clear_below,
            clear_windows=clear_windows, min_descent=min_descent,
            seq_len=lengths.pop(), frame_stride=cfg.PREPROCESS.frame_stride,
            step=cfg.EVAL.sliding_stride, device=device)
        return Detector(name=model, scale=scales.pop(),
                        features=", ".join(c[1].features for c in cfgs),
                        checkpoints=paths, train_cfg=cfgs[0][2], stream=stream)

    path = resolve_checkpoint(model)
    m, mc, tc, scale = T.load_checkpoint(path, device=device)
    return Detector(name=str(model), scale=scale, features=mc.features,
                    checkpoints=[path], model=m, model_cfg=mc, train_cfg=tc)


def cached_track(clip, scale: str):
    """A `PoseTrack` rebuilt from the keypoint and box caches -- no inference.

    Boxes only draw the rectangle, so a clip without cached boxes still renders,
    just without one.
    """
    from .extract import boxes_path, load_boxes, load_keypoints
    from .infer import PoseTrack

    kp = load_keypoints(scale, clip.video_id)
    if boxes_path(scale, clip.video_id).exists():
        boxes = load_boxes(scale, clip.video_id)
    else:
        boxes = np.zeros((len(kp), 5), dtype=np.float32)
    n = min(len(kp), len(boxes))
    return PoseTrack(path=Path(clip.path), keypoints=kp[:n], boxes=boxes[:n],
                     fps=float(clip.fps), width=int(clip.width),
                     height=int(clip.height))


def replay_ensemble(det: Detector, track) -> pd.DataFrame:
    """Feed a track through the ensemble stream; one row per scored window.

    `end_frame` is the frame whose arrival completed the window -- the first
    moment the live detector could have acted on it.
    """
    s = det.stream
    s.reset_buffer()
    s.decision.reset()
    rows = []
    for f, kp in enumerate(track.keypoints):
        if not s.observe(kp):
            continue
        win = s.buffer.window()
        d = s.decision
        rows.append({
            "window": len(rows),
            "start_frame": f - s.buffer.frames_per_window + 1,
            "end_frame": f,
            "end_sec": f / track.fps,
            "dropout": float((win[:, :, 2].sum(axis=1) == 0).mean()),
            "window_p": d.window_p,
            "rolling_mean": d.rolling,
            "alarm": d.alarm,
            **{f"p[{n}]": p for n, p in zip(s.classifier.names, s.model_probs)},
        })
    return pd.DataFrame(rows)


def _spread(windows: pd.DataFrame, column: str, n_frames: int) -> np.ndarray:
    """A per-window column held on every frame from its window's end onwards."""
    from .infer import frame_probabilities

    return frame_probabilities(
        windows.assign(prob=windows[column].astype(float)), n_frames)


def score_clip(clip, det: Detector, threshold: float = cfg.EVAL.threshold,
               max_dropout: float | None = None, device: str | None = None):
    """Score one clip; returns (index row, window table, PoseTrack).

    The decision is the detector's own: any window over threshold for a single
    checkpoint, the rolling mean and latch for an ensemble or a room-fitted
    head. This is everything `render_clip` reports except where the video went,
    so comparing detectors costs the scoring and none of the drawing.
    """
    from . import infer

    track = cached_track(clip, det.scale)
    if det.is_ensemble:
        if max_dropout is not None:
            raise ValueError("the rolling-rule detectors have no detection "
                             "gate; leave max_dropout as None")
        det.stream.decision.threshold = threshold
        # The descent measures divide by the body's longest dimension, which
        # needs x and y in the same units; this clip's frame supplies the ratio.
        det.stream.aspect = track.width / max(track.height, 1)
        windows = replay_ensemble(det, track)
        fired = bool(windows.alarm.any()) if len(windows) else False
        first = windows.index[windows.alarm.to_numpy()][0] if fired else None
        decidable = bool(len(windows))
        peak_score = float(windows.rolling_mean.max()) if decidable else None
        peak_window = float(windows.window_p.max()) if decidable else None
        trigger = float(windows.end_sec[first]) if fired else None
    else:
        windows = infer.score_track(track, det.model, det.model_cfg,
                                    det.train_cfg, device=device)
        v = infer.verdict(windows, threshold=threshold, max_dropout=max_dropout)
        fired, decidable = bool(v["fall"]), bool(v["decidable"])
        peak_score = peak_window = v["peak_prob"]
        trigger = v["trigger_sec"]

    correct = fired == bool(clip.label)
    row = {"action": clip.action, "subject": int(clip.subject),
           "label": int(clip.label), "fired": fired,
           "outcome": "OK" if correct else ("FALSE_ALARM" if fired else "MISSED"),
           "decidable": decidable,
           "peak_score": None if peak_score is None else round(peak_score, 4),
           "peak_window_p": None if peak_window is None else round(peak_window, 4),
           "trigger_sec": trigger,
           "n_windows": len(windows),
           "detection_rate": round(track.detection_rate, 3),
           "duration": round(track.duration, 1)}
    return row, windows, track


def render_clip(clip, det: Detector, root: Path,
                threshold: float = cfg.EVAL.threshold,
                max_dropout: float | None = None,
                skip_correct: bool = False, h264: bool = True,
                overwrite: bool = False, device: str | None = None) -> dict:
    """Score one picam clip and write the annotated video; returns its index row.

    Single checkpoint: notebook 16's decision -- 60-row windows at stride 15, no
    warm-up windows, fire if any window reaches `threshold`. The banner is held
    for 3 s after a crossing.

    `v2 ensemble`: demo-cam v2's decision, replayed through its own stream. The
    clip fires if the latch ever engages; the banner *is* the latch, and the
    readout shows the rolling mean it decides on beside the latest window's
    ensemble P(fall). `max_dropout` is refused here, because v2 has no gate.

    Either way the drawing is causal: a frame shows only what had been decided
    by the time it arrived. `peak_score` is the peak of the quantity the rule
    thresholds (window P for one model, rolling mean for the ensemble);
    `peak_window_p` is the highest single-window P(fall) in the clip.

    Files land at `root/<action>/S<subject>_<OK|FALSE_ALARM|MISSED>.mp4`, plus an
    `_h264` twin a browser will play. A clip rendered earlier under a different
    outcome (a new threshold, a new checkpoint in the same folder) has its old
    files removed, so a folder never holds two verdicts for one clip.

    `skip_correct` scores the clip but writes nothing when the detector gets it
    right; the row comes back with `path=None`.
    """
    from . import infer

    row, windows, track = score_clip(clip, det, threshold=threshold,
                                     max_dropout=max_dropout, device=device)
    row |= {"cached": False, "path": None}
    tag = row["outcome"]
    correct = tag == "OK"
    if skip_correct and correct:
        return row

    out_dir = Path(root) / clip.action.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"S{int(clip.subject)}"
    dest = out_dir / f"{stem}_{tag}.mp4"
    playable = dest.with_name(dest.stem + "_h264.mp4")

    for old in out_dir.glob(f"{stem}_*.mp4"):
        if old not in (dest, playable):
            old.unlink()

    if dest.exists() and not overwrite:
        row["cached"] = True
    else:
        n = track.n_frames
        if det.is_ensemble:
            alarm = np.nan_to_num(_spread(windows, "alarm", n)) >= 0.5
            infer.render(track, _spread(windows, "rolling_mean", n), dest,
                         threshold=threshold, frame_alarm=alarm,
                         frame_detail=_spread(windows, "window_p", n),
                         progress=False)
        else:
            probs = infer.frame_probabilities(windows, n)
            gated = (infer.frame_gates(windows, n, max_dropout)
                     if max_dropout is not None else None)
            infer.render(track, probs, dest, threshold=threshold,
                         frame_gated=gated, progress=False)
        if h264:
            playable.unlink(missing_ok=True)
            infer.to_h264(dest, playable)

    row["path"] = str(playable if playable.exists() else dest)
    return row
