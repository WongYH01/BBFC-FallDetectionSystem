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
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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
}

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
