"""LE2I adapter: annotation parsing and the paper's zone-based label transfer.

LE2I (Charfi et al., ref [52]) ships videos alongside per-video annotation text
files. The convention across its subsets is that the first two integer lines are
the fall start and end frame, followed by one line per frame of bounding-box
coordinates; a video with no fall uses 0 for both. Layouts vary between the
Coffee_room / Home / Office / Lecture-room subsets, so `parse_annotation` is
written to tolerate variation rather than assume one exact format, and
`scan_le2i` reports what it matched so you can check it.

Section IV-E's zone protocol exists because a sliding window that straddles the
start of a fall has no honest label. Rather than guess, those windows are thrown
away and only unambiguous ones are scored.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}


def parse_annotation(path: Path) -> tuple[int, int]:
    """Return (fall_start, fall_end) frame indices; (0, 0) means no fall.

    Reads the first two standalone integers in the file. Lines carrying four or
    more numbers are per-frame bounding boxes, not the header, so they are
    skipped -- that is the one assumption worth knowing about here.
    """
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return 0, 0

    values: list[int] = []
    for line in text.splitlines():
        nums = re.findall(r"-?\d+", line)
        if len(nums) == 1:
            values.append(int(nums[0]))
        elif len(nums) >= 4:
            break          # reached the bounding-box block
        if len(values) >= 2:
            break

    if len(values) < 2:
        return 0, 0
    start, end = values[0], values[1]
    if start <= 0 or end <= start:
        return 0, 0
    return start, end


ANNOTATION_FOLDERS = (
    "Annotation_files",     # Coffee_room_01, Home_01, Home_02
    "Annotations_files",    # Coffee_room_02 -- yes, the plural differs
    "Annotation files",
    "Annotations",
)


def _subset_root(video: Path, root: Path) -> Path:
    """The subset directory this video belongs to (one level under `root`).

    Everything about annotation lookup must stay inside this directory. LE2I
    numbers its videos per subset, so `video (1).avi` exists in Coffee_room_01,
    Home_01, Lecture room and Office alike. A search that escapes the subset
    will happily return another subset's `video (1).txt` and label the clip
    with a fall that happened in a different room.
    """
    try:
        rel = video.relative_to(root)
    except ValueError:
        return video.parent
    return root / rel.parts[0] if len(rel.parts) > 1 else root


def _find_annotation(video: Path, root: Path) -> Path | None:
    """Locate the annotation file belonging to a video, or None.

    Tries a sibling .txt, then the conventional annotation folders, then a name
    match -- all confined to the video's own subset. Returning None matters:
    the Lecture room and Office subsets ship with no ground truth at all, and
    an unannotated video must be excluded rather than assumed fall-free.
    """
    stem = video.stem
    sibling = video.with_suffix(".txt")
    if sibling.exists():
        return sibling

    subset = _subset_root(video, root)
    for folder in ANNOTATION_FOLDERS:
        candidate = subset / folder / f"{stem}.txt"
        if candidate.exists():
            return candidate

    if subset == root:
        return None          # flat subset with no annotation folder of its own
    matches = sorted(subset.rglob(f"{stem}.txt"))
    return matches[0] if matches else None


def scan_le2i(root: Path | None = None) -> pd.DataFrame:
    """Build a manifest of LE2I videos with their fall annotations."""
    root = Path(root or cfg.LE2I_DIR)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Download the LE2I fall detection dataset and "
            f"extract it there (subsets: Coffee_room, Home, Office, Lecture room)."
        )

    rows = []
    for video in sorted(root.rglob("*")):
        if not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        ann = _find_annotation(video, root)
        start, end = parse_annotation(ann) if ann else (0, 0)
        rel = video.relative_to(root)
        rows.append({
            "video_id": "le2i_" + re.sub(r"[^A-Za-z0-9_.-]", "_", rel.as_posix()),
            "path": str(video),
            "rel_path": rel.as_posix(),
            "subset": rel.parts[0] if len(rel.parts) > 1 else "root",
            "annotation": str(ann) if ann else None,
            "annotated": int(ann is not None),
            "fall_start": start,
            "fall_end": end,
            "has_fall": int(start > 0),
        })

    if not rows:
        raise FileNotFoundError(f"No videos found under {root}")
    return pd.DataFrame(rows)


def zone_labels(
    n_frames: int,
    fall_start: int,
    fall_end: int,
    seq_len: int,
    stride: int = cfg.EVAL.sliding_stride,
    margin: int = cfg.LE2I_SAFETY_MARGIN,
) -> list[tuple[int, int]]:
    """Label each sliding window as ADL (0) or Fall (1), dropping ambiguous ones.

    Section IV-E:
      ADL zone   - the window ends before (fall_start - margin)   -> 0
      Fall zone  - the window fully contains the annotated event  -> 1
      otherwise  - boundary or post-fall, excluded entirely

    Returns [(start_frame, label)] for the windows that survive. A video with no
    annotated fall contributes every window as ADL.
    """
    out: list[tuple[int, int]] = []
    if n_frames < seq_len:
        return out

    for start in range(0, n_frames - seq_len + 1, stride):
        end = start + seq_len

        if fall_start <= 0:                    # no fall in this video at all
            out.append((start, 0))
            continue
        if end < fall_start - margin:          # safely before the event
            out.append((start, 0))
        elif start <= fall_start and end >= fall_end:   # fully contains it
            out.append((start, 1))
        # everything else is boundary or post-fall: no honest label, so skipped

    return out


def build_clips(
    le2i_manifest: pd.DataFrame,
    scale: str = cfg.DEFAULT_SCALE,
    seq_len: int = cfg.FINAL_TRAIN.seq_len,
    stride: int = cfg.EVAL.sliding_stride,
    margin: int = cfg.LE2I_SAFETY_MARGIN,
) -> pd.DataFrame:
    """Expand annotated videos into labelled, unambiguous evaluation clips.

    Annotations are in *original* frame numbers, but the cached keypoints were
    subsampled by PREPROCESS.frame_stride. Both the fall boundaries and the
    window length are converted into cache index space here -- forgetting that
    conversion silently shifts every label by a factor of the stride.
    """
    from .extract import keypoints_path

    fs = cfg.PREPROCESS.frame_stride
    rows = []
    for _, v in le2i_manifest.iterrows():
        # No annotation means no label, not "no fall". The Lecture room and
        # Office subsets have no ground-truth files; scoring their windows as
        # ADL would count every real fall in them as a false positive.
        if not getattr(v, "annotated", 1):
            continue
        p = keypoints_path(scale, v.video_id)
        if not p.exists():
            continue
        # The cache holds one row per ORIGINAL frame; prepare_sequence strides it
        # down at scoring time. Every quantity handed to zone_labels must live in
        # that same strided index space, sequence length included -- passing the
        # raw cache length here while converting only the fall boundaries
        # manufactures roughly 1/frame_stride too many windows, mislabels them,
        # and leaves take_window clamping the overflow onto the final window.
        n_cached = len(np.load(p, mmap_mode="r"))
        n_seq = -(-n_cached // fs)          # == len(arr[::fs])

        for start, label in zone_labels(
            n_seq,
            fall_start=int(np.ceil(v.fall_start / fs)) if v.fall_start > 0 else 0,
            fall_end=int(np.ceil(v.fall_end / fs)) if v.fall_end > 0 else 0,
            seq_len=seq_len,
            stride=stride,
            margin=max(1, margin // fs),
        ):
            rows.append({"video_id": v.video_id, "subset": v.subset,
                         "start": start, "label": label})

    return pd.DataFrame(rows)
