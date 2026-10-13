"""CAUCAFall adapter: frame-folder ingest and an event-semantics zone protocol.

CAUCAFall (Eraso et al., Universidad del Cauca) ships 100 sequences -- 10
subjects x 10 activities, five fall types and five ADLs -- as per-frame PNGs
with one YOLO-format .txt label beside each frame, plus a folder-local
classes.txt naming the class indices.

Two things make it different from LE2I, and both change the code rather than
just the paths.

**The label is a state, not an event.** LE2I annotates the falling *motion* and
the paper's zone protocol asks for a window that fully contains it. CAUCAFall
marks every frame from onset to the end of the clip as fall -- `fall_end` is the
final frame in 49 of the 50 fall sequences, and 65% of frames in a fall sequence
carry label 1. It is annotating "fallen", the person on the ground included, so
`le2i.zone_labels` run over it unchanged yields **four** fall clips out of 176.

The tempting repair -- accept any window lying *inside* `[onset, end]` -- is
wrong, and measurably so: it scores AUC **0.34**, below chance. The reason is
that the classifier is an event detector. Swept across a fall sequence its mean
P(fall) runs 0.99 spanning the onset, 0.79 just after, and **0.47** on the
post-fall plateau, while every one of the 50 sequences peaks above 0.5 within a
second of onset. A zone built from "inside the fall interval" is dominated by
that plateau, so it labels as positive the one stretch where a correct event
detector is silent.

`zone_labels` therefore keeps the paper's event semantics and reads CAUCAFall's
annotation for what it can support: the onset is a true transition, so a window
counts as Fall when it *contains* that transition. Windows lying wholly after it
are excluded, exactly as Section IV-E excludes LE2I's post-fall windows.

**The videos are not the ground truth.** Each folder also holds an .avi, but its
frame count disagrees with the label count in 99 of 100 folders -- by one frame
usually, by 23 in Subject.6/Fall backwards. The PNGs are keyed to the .txt files
by filename, so the frame folder is authoritative and the .avi is ignored.

Frame numbering has a trap: the prefix embeds the subject digit, so
`cams900140.txt` is subject 9, frame 140, and a naive \\d+ grab returns 900140.
`FRAME_RE` anchors on the last five digits instead.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

# Last run of five digits, with no further digits after it. Handles the
# subject-digit prefix (cams9|00140), the dashed variant (cfs1-|00001) and the
# three files carrying a trailing letter (sals800096a.txt).
FRAME_RE = re.compile(r"(\d{5})(?=\D*$)")

FALL_ACTIVITIES = frozenset({
    "Fall backwards", "Fall forward", "Fall left", "Fall right", "Fall sitting",
})


def _frame_number(name: str) -> int | None:
    m = FRAME_RE.search(name)
    return int(m.group(1)) if m else None


def read_classes(folder: Path) -> dict[int, str]:
    """Class index -> name from a folder's classes.txt.

    Read per folder rather than assumed globally: ADL folders declare only
    `nofall` while fall folders declare `nofall, fall`. All 100 folders in the
    release agree that index 1 means fall, but reading the file is what makes
    that a checked fact instead of a hope.
    """
    path = folder / "classes.txt"
    if not path.exists():
        return {}
    names = [ln.strip() for ln in path.read_text(errors="ignore").splitlines()
             if ln.strip()]
    return dict(enumerate(names))


def frame_labels(folder: Path) -> tuple[list[Path], np.ndarray]:
    """Ordered frame images and their per-frame labels for one activity folder.

    Returns only frames that have both a PNG and a .txt, in ascending frame
    order. Three folders are missing a single annotation and two have a
    one-frame key mismatch; dropping the odd frame out keeps images and labels
    index-aligned, which is the property everything downstream relies on.
    """
    classes = read_classes(folder)
    fall_idx = {i for i, n in classes.items() if n.lower().startswith("fall")}

    images: dict[int, Path] = {}
    # Sorted for determinism. The release ships one stray "cas200091 - copia.png"
    # with no original beside it; it is not a collision -- the folder has no two
    # files mapping to one frame -- because FRAME_RE reads the trailing five
    # digits, so "cas2|00091" slots it into the sequence as frame 91.
    for p in sorted(folder.iterdir(), key=lambda q: q.name):
        if p.suffix.lower() != ".png":
            continue
        n = _frame_number(p.name)
        if n is not None and n not in images:
            images[n] = p

    labels: dict[int, int] = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() != ".txt" or p.name == "classes.txt":
            continue
        n = _frame_number(p.name)
        if n is None:
            continue
        body = p.read_text(errors="ignore").split()
        labels[n] = int(int(body[0]) in fall_idx) if body else 0

    keys = sorted(set(images) & set(labels))
    return [images[k] for k in keys], np.array([labels[k] for k in keys], dtype=np.int64)


def scan_caucafall(root: Path | None = None) -> pd.DataFrame:
    """Manifest of CAUCAFall sequences with onset/end in 0-based frame index.

    `path` holds the ordered list of frame images, which is what
    `extract.extract_video` consumes for a frame-folder dataset. `onset` and
    `end` are indices into that list, so they need no separate alignment step.
    """
    root = Path(root or cfg.CAUCA_DIR)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Extract the CAUCAFall release so that "
            f"{root} contains Subject.1 .. Subject.10."
        )

    rows = []
    for subject in sorted(root.iterdir()):
        if not subject.is_dir():
            continue
        for folder in sorted(subject.iterdir()):
            if not folder.is_dir():
                continue
            frames, labels = frame_labels(folder)
            if not frames:
                continue
            fall = np.flatnonzero(labels == 1)
            contiguous = bool(fall.size == 0 or fall[-1] - fall[0] + 1 == fall.size)
            rows.append({
                "video_id": f"cauca_{subject.name}_{folder.name}".replace(" ", "_"),
                "path": frames,
                "subject": subject.name,
                "activity": folder.name,
                "n_frames": len(frames),
                "onset": int(fall[0]) if fall.size else -1,
                "end": int(fall[-1]) if fall.size else -1,
                "n_fall_frames": int(fall.size),
                "has_fall": int(fall.size > 0),
                "contiguous": int(contiguous),
                "declared_fall": int(folder.name in FALL_ACTIVITIES),
            })

    if not rows:
        raise FileNotFoundError(f"No annotated frame folders found under {root}")
    return pd.DataFrame(rows)


def zone_labels(
    n_frames: int,
    onset: int,
    end: int,
    seq_len: int,
    stride: int = cfg.EVAL.sliding_stride,
    margin: int = cfg.CAUCA_SAFETY_MARGIN,
) -> list[tuple[int, int]]:
    """Windows labelled under event semantics; ambiguous ones dropped.

      ADL zone   - window ends before (onset - margin)   -> 0
      Fall zone  - window contains the onset transition  -> 1
      otherwise  - the margin band, or the post-fall plateau: excluded

    Sequences with no annotated fall contribute every window as ADL, which is
    correct here because the five ADL activities contain no fall by
    construction. Those windows are the interesting half of the benchmark:
    kneeling, sitting and picking an object off the floor are downward motions
    ending near the ground, and are precisely what a fall detector must not
    fire on.

    `end` is accepted for interface symmetry with `fallcore.le2i` and for
    diagnostics, but carries almost no information: it is pinned to the final
    frame in 49 of 50 sequences. The onset is the only real event boundary
    CAUCAFall provides, which is why the fall zone is defined against it.
    """
    out: list[tuple[int, int]] = []
    if n_frames < seq_len:
        return out

    for start in range(0, n_frames - seq_len + 1, stride):
        stop = start + seq_len
        if onset < 0:
            out.append((start, 0))
        elif stop <= onset - margin:
            out.append((start, 0))
        elif start <= onset < stop:
            out.append((start, 1))
    return out


def state_zone_labels(
    n_frames: int,
    onset: int,
    end: int,
    seq_len: int,
    stride: int = cfg.EVAL.sliding_stride,
    margin: int = cfg.CAUCA_SAFETY_MARGIN,
) -> list[tuple[int, int]]:
    """The rejected state-semantics variant, kept so the choice stays checkable.

    Fall zone is any window lying entirely within `[onset, end]`. Notebook 08
    scores this alongside the event rule; it lands below chance, and seeing that
    is the argument for the event rule. Do not report numbers from it.
    """
    out: list[tuple[int, int]] = []
    if n_frames < seq_len:
        return out

    for start in range(0, n_frames - seq_len + 1, stride):
        stop = start + seq_len
        if onset < 0:
            out.append((start, 0))
        elif stop <= onset - margin:
            out.append((start, 0))
        elif start >= onset and stop <= end + 1:
            out.append((start, 1))
    return out


def build_clips(
    manifest: pd.DataFrame,
    scale: str = cfg.DEFAULT_SCALE,
    seq_len: int = cfg.FINAL_TRAIN.seq_len,
    stride: int = cfg.EVAL.sliding_stride,
    margin: int = cfg.CAUCA_SAFETY_MARGIN,
    frame_stride: int | None = None,
    rule=None,
) -> pd.DataFrame:
    """Expand sequences into labelled, unambiguous evaluation windows.

    The cache stores one row per original frame; `frame_stride` is applied at
    scoring time, so every quantity handed to `zone_labels` -- clip length,
    onset, end and margin alike -- is converted into that same strided index
    space here. Mixing the two spaces is the bug that silently multiplies the
    window count and clamps the overflow onto the final window.
    """
    from .extract import keypoints_path

    rule = zone_labels if rule is None else rule
    fs = cfg.PREPROCESS.frame_stride if frame_stride is None else frame_stride
    rows = []
    for _, v in manifest.iterrows():
        p = keypoints_path(scale, v.video_id)
        if not p.exists():
            continue
        n_seq = -(-len(np.load(p, mmap_mode="r")) // fs)

        for start, label in rule(
            n_seq,
            onset=(v.onset // fs) if v.onset >= 0 else -1,
            end=(v.end // fs) if v.end >= 0 else -1,
            seq_len=seq_len,
            stride=stride,
            margin=max(1, margin // fs),
        ):
            rows.append({"video_id": v.video_id, "subject": v.subject,
                         "activity": v.activity, "start": start, "label": label})

    return pd.DataFrame(rows)


def window_seconds(seq_len: int, frame_stride: int, fps: float = cfg.CAUCA_FPS) -> float:
    """Elapsed time one window covers -- the quantity that must be compared."""
    return seq_len * frame_stride / fps


# ---------------------------------------------------------------------------
# Hard-negative split (notebook 10)
# ---------------------------------------------------------------------------
# Subjects 1-6 may enter training; 7-10 are never trained on, so CAUCAFall
# survives as an evaluation set rather than being consumed by it. Fixed here
# rather than chosen per notebook, because a split that moves between
# experiments is not a held-out set.
TRAIN_SUBJECTS = tuple(f"Subject.{i}" for i in range(1, 7))


def training_negatives(manifest: pd.DataFrame,
                       subjects: tuple[str, ...] = TRAIN_SUBJECTS) -> pd.DataFrame:
    """The ADL sequences eligible to be added to training as hard negatives.

    ADL-only, deliberately. CAUCAFall labels the fallen *state* while the
    training corpus labels the fall *event*, so importing its fall sequences
    would teach the model both definitions at once -- a change to what the
    system detects rather than more data for the task it already does. Its ADL
    sequences carry no such conflict: they are No-Fall from first frame to last,
    which also means every random window `KeypointClipDataset` draws from one is
    correctly labelled, whereas a window drawn from a fall sequence could land
    before the onset.
    """
    return manifest[(manifest.subject.isin(subjects)) & (manifest.has_fall == 0)]


def heldout(manifest: pd.DataFrame,
            subjects: tuple[str, ...] = TRAIN_SUBJECTS) -> pd.DataFrame:
    """Sequences from the subjects that never enter training."""
    return manifest[~manifest.subject.isin(subjects)]
