"""Turn a directory of videos into a labelled manifest, then a frozen split.

The Kaggle compilation (Charoensri, "Fall Video Dataset", ref [48] in the
paper) aggregates three sources into Fall / No-Fall folders. Real downloads are
messier than that: extra nesting, differing folder spellings, a stray README.
`scan_raw` walks whatever is there and infers the label from the path, and
`describe_tree` prints what it found so you can check the inference before
trusting it.

The split is computed once, written to splits.json, and thereafter loaded --
never recomputed. Every YOLO scale must be trained and scored on identical
videos or the size comparison measures the split as much as the backbone.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import config as cfg

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".m4v"}

# Path components that mark a clip as a fall / not a fall. Matched
# case-insensitively against each directory name on the way down. "nofall" is
# checked first because "fall" is a substring of it.
NOFALL_TOKENS = ("nofall", "no_fall", "no-fall", "not_fall", "notfall",
                 "adl", "normal", "nonfall", "non_fall", "non-fall")
FALL_TOKENS = ("fall", "falling", "fallen")


def label_from_path(path: Path, root: Path) -> int | None:
    """Infer 1 (Fall) / 0 (No-Fall) from a video's path, or None if unclear.

    Checks the deepest directory first: in a layout like `Fall/subject3/clip.mp4`
    the label sits above the subject folder, but in `train/Fall/clip.mp4` the
    nearest labelled ancestor is still the right answer either way. Returning
    None rather than guessing keeps unlabelled strays out of the manifest
    instead of silently poisoning a class.
    """
    parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
    for part in reversed(parts):
        squashed = re.sub(r"[^a-z]", "", part)
        if any(re.sub(r"[^a-z]", "", t) in squashed for t in NOFALL_TOKENS):
            return 0
        if any(t in part for t in FALL_TOKENS):
            return 1
    return None


def describe_tree(root: Path, max_depth: int = 3, max_entries: int = 12) -> str:
    """Render the directory tree under `root`, with video counts per folder.

    Run this before anything else. The whole pipeline rests on the labels being
    read correctly off the paths, and this is how you check that by eye.
    """
    root = Path(root)
    if not root.exists():
        return f"{root} does not exist"

    lines: list[str] = [str(root)]

    def walk(d: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(p for p in d.iterdir() if p.is_dir())
        except PermissionError:
            return
        n_videos = sum(1 for p in d.iterdir()
                       if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
        if n_videos:
            lines.append(f"{prefix}    [{n_videos} videos here]")
        for i, sub in enumerate(entries[:max_entries]):
            last = (i == len(entries[:max_entries]) - 1)
            lines.append(f"{prefix}{'`-- ' if last else '|-- '}{sub.name}/")
            walk(sub, prefix + ("    " if last else "|   "), depth + 1)
        if len(entries) > max_entries:
            lines.append(f"{prefix}... and {len(entries) - max_entries} more folders")

    walk(root, "", 1)
    return "\n".join(lines)


def probe_video(path: Path) -> dict:
    """Read frame count, fps and resolution without decoding the whole file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"n_frames": 0, "fps": 0.0, "width": 0, "height": 0, "readable": False}
    info = {
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "readable": True,
    }
    cap.release()
    # Some containers report a bogus frame count; treat non-positive as unknown
    # rather than as an empty video, so the clip still gets extracted.
    if info["n_frames"] <= 0:
        info["n_frames"] = -1
    return info


def scan_raw(root: Path | None = None, probe: bool = True) -> pd.DataFrame:
    """Walk `root` and build the manifest.

    Returns one row per labelled video with a stable `video_id` derived from the
    relative path, so keypoint cache filenames survive the directory being moved.
    Unlabelled videos are returned too, with label -1, so they show up in the
    audit rather than vanishing.
    """
    root = Path(root or cfg.RAW_DIR)
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        rel = path.relative_to(root)
        label = label_from_path(path, root)
        row = {
            "video_id": re.sub(r"[^A-Za-z0-9_.-]", "_", rel.as_posix()),
            "path": str(path),
            "rel_path": rel.as_posix(),
            # The top-level folder is the closest thing to a source label the
            # compilation gives us; useful for per-source breakdowns later.
            "source": rel.parts[0] if len(rel.parts) > 1 else "root",
            "label": -1 if label is None else label,
        }
        if probe:
            row.update(probe_video(path))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            f"No video files found under {root}. Extract the Kaggle "
            f"'payutch/fall-video-dataset' download into that directory first."
        )
    return df


def audit(df: pd.DataFrame) -> str:
    """Compare the manifest against the counts the paper reports."""
    labelled = df[df.label >= 0]
    n_total = len(labelled)
    n_fall = int((labelled.label == 1).sum())
    n_nofall = int((labelled.label == 0).sum())
    n_unlabelled = int((df.label < 0).sum())

    def cmp(name: str, got: int, want: int) -> str:
        mark = "OK " if got == want else "!! "
        return f"  {mark}{name:<12} {got:>6,}   paper: {want:>6,}   diff: {got - want:+,}"

    lines = [
        "Manifest vs. paper (Tables 2-3)",
        cmp("total", n_total, cfg.PAPER_TOTAL_CLIPS),
        cmp("Fall", n_fall, cfg.PAPER_FALL_CLIPS),
        cmp("No-Fall", n_nofall, cfg.PAPER_NOFALL_CLIPS),
    ]
    if n_total:
        lines.append(f"  class balance: {n_fall / n_total:.1%} Fall / "
                     f"{n_nofall / n_total:.1%} No-Fall "
                     f"(paper: 44.9% / 55.1%)")
    if n_unlabelled:
        lines.append(f"  !! {n_unlabelled:,} videos had no inferable label "
                     f"(label = -1); inspect them before continuing")
    unreadable = int((~df.get("readable", pd.Series(True, index=df.index))).sum())
    if unreadable:
        lines.append(f"  !! {unreadable:,} videos could not be opened by OpenCV")
    return "\n".join(lines)


def stratified_split(
    video_ids: list[str],
    labels: list[int],
    fractions: tuple[float, float, float] = cfg.SPLIT_FRACTIONS,
    seed: int = cfg.SPLIT_SEED,
) -> dict[str, list[str]]:
    """Partition into train/val/test, preserving the class ratio in each.

    Splitting each class independently and concatenating is what makes the
    result stratified; shuffling the pooled list would only be stratified in
    expectation.
    """
    rng = np.random.default_rng(seed)
    ids = np.asarray(video_ids)
    y = np.asarray(labels)
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for cls in np.unique(y):
        cls_ids = ids[y == cls]
        cls_ids = cls_ids[rng.permutation(len(cls_ids))]
        n = len(cls_ids)
        n_train = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        out["train"].extend(cls_ids[:n_train].tolist())
        out["val"].extend(cls_ids[n_train:n_train + n_val].tolist())
        out["test"].extend(cls_ids[n_train + n_val:].tolist())

    # Shuffle within each split so a batch is not class-ordered.
    for k in out:
        arr = np.asarray(out[k])
        out[k] = arr[rng.permutation(len(arr))].tolist()
    return out


def save_splits(splits: dict[str, list[str]], meta: dict | None = None,
                path: Path | None = None) -> Path:
    path = Path(path or cfg.SPLITS_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta or {}, "splits": splits}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_splits(path: Path | None = None) -> dict[str, list[str]]:
    path = Path(path or cfg.SPLITS_JSON)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run notebook 01 to build the keypoint cache "
            f"and freeze the split before training."
        )
    return json.loads(path.read_text(encoding="utf-8"))["splits"]


def load_manifest(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or cfg.MANIFEST_CSV)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run notebook 00 first.")
    return pd.read_csv(path)
