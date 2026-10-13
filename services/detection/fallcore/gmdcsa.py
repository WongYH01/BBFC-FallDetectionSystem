"""GMDCSA24 adapter: second-based annotations over a mixed-frame-rate corpus.

GMDCSA24 (Alam et al., Data in Brief 2024) is 160 videos of 4 subjects in a home
setting, split into `Fall/` and `ADL/` folders with one CSV of annotations per
subject per folder. It brings hard negatives the other sets lack -- `Sleeping`
(a person lying on a bed), `Exercising`, `Reading` on the floor -- which is
exactly the class of activity that breaks a fall detector.

Three things differ from `fallcore.caucafall` and each one changes the code.

**Annotations are in seconds, and the frame rate is not constant.** 131 videos
run at ~30 fps and 29 at 15 fps, so a fixed seconds-to-frames factor would
misplace every onset in the 15 fps subset by a factor of two. Every conversion
here uses the video's own measured fps. The same applies to the safety margin,
which is therefore specified in seconds (`cfg.GMDCSA_MARGIN_SEC`) rather than
frames.

**Only the onset is real.** `fall_end` equals the CSV's integer "Length
(seconds)" column in 76 of 77 fall videos, and that column is the floor of the
true duration (0.42 s short on average). It records "the fall lasts to the end
of the clip", not a measured event end -- the same state-style annotation
CAUCAFall uses. So the fall zone is defined against the onset, and
`caucafall.zone_labels` is reused unchanged rather than reimplemented: the
protocol question was settled there, and using one function keeps the two
cross-dataset results genuinely comparable.

**The CSV is not reliably comma-delimited.** The Description column contains
commas, a few rows separate class intervals with commas, one row writes
`Falling (BW)[2.3 6]` without the "to", and one writes a bare `Falling (FW)`
with no interval at all. `SPAN` pulls every `Name[a to b]` out of the raw line
and ignores field boundaries; the video with no interval has no recoverable
onset and is dropped by `scan_gmdcsa`, which reports it rather than guessing.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .caucafall import zone_labels          # identical event semantics; see above

# "Name[a to b]", tolerating a missing "to" and a missing closing bracket.
SPAN = re.compile(r"([A-Za-z][A-Za-z ]*?(?:\([A-Z]{2}\)?)?)\s*\[\s*"
                  r"([\d.]+)\s*(?:to\s*|\s)([\d.]+)\s*\]?", re.I)
FALL_CLASS = re.compile(r"^fall", re.I)
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov"}


def parse_spans(line: str) -> list[tuple[str, float, float]]:
    """Every `(class, start_sec, end_sec)` interval mentioned on a CSV line."""
    return [(m.group(1).strip(), float(m.group(2)), float(m.group(3)))
            for m in SPAN.finditer(line)]


def _probe(path: Path) -> tuple[int, float, int, int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    return n, fps, w, h


def scan_gmdcsa(root: Path | None = None, probe: bool = True) -> pd.DataFrame:
    """Manifest of GMDCSA24 videos with fall onsets in seconds.

    A row in a `Fall/` folder whose annotation carries no parseable interval is
    marked `annotated = 0`: its onset is genuinely unknown, and inventing one
    would put a mislabelled positive into the benchmark.
    """
    root = Path(root or cfg.GMDCSA_DIR)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Extract GMDCSA24 there so that it contains "
            f"'Subject 1' .. 'Subject 4'."
        )

    rows = []
    for subject in sorted(root.glob("Subject *")):
        for kind in ("ADL", "Fall"):
            csv = subject / f"{kind}.csv"
            if not csv.exists():
                continue
            for line in csv.read_text(errors="ignore").splitlines()[1:]:
                if not line.strip():
                    continue
                name = line.split(",")[0].strip()
                video = subject / kind / name
                if video.suffix.lower() not in VIDEO_SUFFIXES or not video.exists():
                    continue

                spans = parse_spans(line)
                falls = [s for s in spans if FALL_CLASS.match(s[0])]
                is_fall_folder = kind == "Fall"
                n_frames, fps, w, h = _probe(video) if probe else (0, 0.0, 0, 0)

                rows.append({
                    "video_id": f"gm_{subject.name}_{kind}_{name}".replace(" ", "_"),
                    "path": str(video),
                    "subject": subject.name,
                    "kind": kind,
                    "activities": ";".join(sorted({s[0] for s in spans})),
                    "fall_start": min((f[1] for f in falls), default=-1.0),
                    "fall_end": max((f[2] for f in falls), default=-1.0),
                    "has_fall": int(bool(falls)),
                    # A Fall-folder video needs a parseable onset to be usable;
                    # an ADL video needs nothing.
                    "annotated": int(bool(falls) or not is_fall_folder),
                    "folder_says_fall": int(is_fall_folder),
                    "n_frames": n_frames, "fps": fps, "width": w, "height": h,
                })

    if not rows:
        raise FileNotFoundError(f"No annotated videos found under {root}")
    return pd.DataFrame(rows)


def build_clips(
    manifest: pd.DataFrame,
    scale: str = cfg.DEFAULT_SCALE,
    seq_len: int = cfg.FINAL_TRAIN.seq_len,
    stride: int = cfg.EVAL.sliding_stride,
    margin_sec: float = cfg.GMDCSA_MARGIN_SEC,
    frame_stride: int | None = None,
) -> pd.DataFrame:
    """Expand videos into labelled, unambiguous evaluation windows.

    Seconds become cache indices per video, using that video's own fps -- the
    step that a constant factor would get wrong for the 15 fps subset. The cache
    holds one row per original frame, so both the onset and the margin are then
    divided by `frame_stride` to land in the strided index space the scorer
    windows over.
    """
    from .extract import keypoints_path

    fs = cfg.PREPROCESS.frame_stride if frame_stride is None else frame_stride
    rows = []
    for _, v in manifest.iterrows():
        if not v.annotated:
            continue
        p = keypoints_path(scale, v.video_id)
        if not p.exists():
            continue
        n_seq = -(-len(np.load(p, mmap_mode="r")) // fs)
        fps = v.fps if v.fps and v.fps > 0 else 30.0

        onset = int(np.floor(v.fall_start * fps / fs)) if v.has_fall else -1
        end = int(np.ceil(v.fall_end * fps / fs)) if v.has_fall else -1

        for start, label in zone_labels(
            n_seq, onset=onset, end=end, seq_len=seq_len, stride=stride,
            margin=max(1, int(round(margin_sec * fps / fs))),
        ):
            rows.append({"video_id": v.video_id, "subject": v.subject,
                         "kind": v.kind, "activities": v.activities,
                         "fps_band": "15fps" if v.fps < 22 else "30fps",
                         "start": start, "label": label})

    return pd.DataFrame(rows)


def window_seconds(seq_len: int, frame_stride: int, fps: float) -> float:
    """Elapsed time one window covers at a given frame rate."""
    return seq_len * frame_stride / fps
