"""OmniFall (Schneider et al., HuggingFace `simplexsigil2/omnifall`) as a corpus.

OmniFall unifies eight staged fall datasets, 818 in-the-wild OOPS clips and
12,000 synthetic videos under one 16-class taxonomy with dense temporal
segments. It ships *annotations only*: the staged videos must be obtained from
their original sources, which is why `coverage()` exists and is the first thing
any notebook using this module should call.

Two things make it a better training source than the Kaggle compilation this
project replicated, and both are about labels rather than pixels:

*The event/state distinction is stated, not inferred.* `fall` (id 1) is the act,
`fallen` (id 2) is being on the ground afterwards, and `lie_down` (5) / `lying`
(6) are the intentional versions. This project had to reverse-engineer that
distinction for CAUCAFall and got it backwards on the first attempt -- a
state-semantics rule scored AUC 0.34, below chance -- before switching to event
semantics. Here it is given.

*Subjects and cameras are identified*, so cross-subject and cross-view splits
are the dataset's own rather than something we impose on top of filenames.

The catch, and it is not small: LE2I, CAUCAFall and GMDCSA24 are all *inside*
OmniFall. Training on the staged corpus makes notebooks 06, 08 and 09 in-domain
rather than zero-shot. `contaminates()` names which of this project's
evaluations a given training selection would compromise, so that is a decision
taken deliberately and not by accident.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

HF_RAW = "https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/main/"

# Directory names as OmniFall spells them; note GMDCSA24 is the one in caps.
STAGED = ("caucafall", "cmdfall", "edf", "GMDCSA24", "le2i", "mcfd", "occu",
          "up_fall")

# The 16-class taxonomy, from LABELS.md.
CLASSES = {
    0: "walk", 1: "fall", 2: "fallen", 3: "sit_down", 4: "sitting",
    5: "lie_down", 6: "lying", 7: "stand_up", 8: "standing", 9: "other",
    10: "kneel_down", 11: "kneeling", 12: "squat_down", 13: "squatting",
    14: "crawl", 15: "jump",
}

# Binary collapse. `fall` is the event and `fallen` its aftermath; taking both
# as positive matches how the Kaggle corpus is labelled, where a clip containing
# a fall is Fall for its whole length, and matches the window assignment in
# notebooks 08 and 09. Everything else -- including `lie_down` and `lying`, the
# intentional versions -- is negative, which is exactly the distinction the
# model currently cannot make.
FALL_IDS = (1, 2)
FALL_EVENT_ONLY = (1,)

# Which of this project's evaluations each staged dataset would contaminate.
EVAL_OVERLAP = {
    "le2i": "LE2I (notebook 06)",
    "caucafall": "CAUCAFall (notebook 08)",
    "GMDCSA24": "GMDCSA24 (notebook 09)",
}

OMNIFALL_DIR = cfg.OMNIFALL_DIR

# In-the-wild component. Not staged, not one of this project's benchmarks, and
# the only real uncontaminated footage available here -- so it carries more
# weight in a training selection than its 818 videos suggest.
ITW = ("OOPS",)

# The 17 keypoint columns as the parquet cache spells them: x0,y0,c0,...,x16.
# Note the ordering is (x, y, conf) in pixels, where this project's cache is
# (y, x, conf) normalised. `to_keypoints` is the one place that is converted.
_KP_COLS = [f"{a}{i}" for i in range(cfg.N_KEYPOINTS) for a in ("x", "y", "c")]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def fetch(rel: str, root: Path = OMNIFALL_DIR, force: bool = False) -> Path | None:
    """Download one annotation file, cached on disk. Returns None if absent.

    Only labels and splits are fetched -- a few megabytes. Video is never
    downloaded from here, because OmniFall does not host the staged video.
    """
    dest = root / rel
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(HF_RAW + rel, timeout=60) as r:
            dest.write_bytes(r.read())
    except Exception:
        return None
    return dest


def fetch_all(datasets=STAGED, protocols=("cs", "cv"),
              root: Path = OMNIFALL_DIR) -> dict:
    """Pull every label and split file for the given datasets."""
    got = {"labels": [], "splits": [], "missing": []}
    fetch("labels/label2id.csv", root)
    for ds in datasets:
        if fetch(f"labels/{ds}.csv", root):
            got["labels"].append(ds)
        else:
            got["missing"].append(f"labels/{ds}.csv")
        for proto in protocols:
            for part in ("train", "val", "test"):
                for name in (ds, ds.lower(), ds.upper()):
                    rel = f"splits/{proto}/{name}/{part}.csv"
                    if fetch(rel, root):
                        got["splits"].append(rel)
                        break
                else:
                    got["missing"].append(f"splits/{proto}/{ds}/{part}.csv")
    return got


def load_labels(datasets=STAGED, root: Path = OMNIFALL_DIR) -> pd.DataFrame:
    """All segment annotations as one frame: path, label, start, end, subject,
    cam, dataset. `start`/`end` are seconds."""
    frames = []
    for ds in datasets:
        p = root / "labels" / f"{ds}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(
            f"no OmniFall labels under {root} -- call fetch_all() first")
    df = pd.concat(frames, ignore_index=True)
    df["class"] = df.label.map(CLASSES)
    return df


def load_split(dataset: str, protocol: str = "cs",
               root: Path = OMNIFALL_DIR) -> dict[str, list[str]]:
    """{'train': [...], 'val': [...], 'test': [...]} of OmniFall paths.

    Missing partitions come back empty rather than raising: not every dataset
    publishes both protocols, and a caller iterating over all eight should not
    have to special-case that.
    """
    # OmniFall is not internally consistent about case: labels ship as
    # `GMDCSA24.csv` while its splits live under `gmdcsa24/`. Try the spellings
    # rather than hard-coding a rename that would break if they fix it.
    base = root / "splits" / protocol
    names = [dataset, dataset.lower(), dataset.upper()]
    out = {}
    for part in ("train", "val", "test"):
        out[part] = []
        for name in names:
            f = base / name / f"{part}.csv"
            if f.exists():
                out[part] = pd.read_csv(f).path.astype(str).tolist()
                break
    return out


# ---------------------------------------------------------------------------
# Mapping OmniFall paths onto local files
# ---------------------------------------------------------------------------
def _key(s: str) -> str:
    """Fold the separator conventions that differ between OmniFall and disk.

    Every mismatch found so far is punctuation, not naming: OmniFall writes
    `Lecture_room` where the download ships `Lecture room`, `Pickupobject` for
    `Pick up object`, `Subject_1` for `Subject 1`. Folding spaces, underscores
    and hyphens away and lowercasing resolves all of them without a per-dataset
    table of special cases.
    """
    return "".join(c for c in s.lower() if c.isalnum())


def _index(root: Path, suffixes: tuple[str, ...] | None,
           dirs: bool = False) -> dict[str, Path]:
    """Folded-name -> path for everything under `root`, files or directories."""
    if not root.exists():
        return {}
    out: dict[str, Path] = {}
    for p in root.rglob("*"):
        if dirs and not p.is_dir():
            continue
        if not dirs and (not p.is_file()
                         or (suffixes and p.suffix.lower() not in suffixes)):
            continue
        out.setdefault(_key(p.stem if not dirs else p.name), p)
        # Also key on the last two components, which is what disambiguates
        # `Subject 1/ADL/01` from `Subject 2/ADL/01`.
        if p.parent != root:
            out.setdefault(_key(p.parent.name + (p.stem if not dirs else p.name)),
                           p)
        if p.parent.parent != root and p.parent.parent.name:
            out.setdefault(
                _key(p.parent.parent.name + p.parent.name
                     + (p.stem if not dirs else p.name)), p)
    return out


_INDEX_CACHE: dict[str, dict[str, Path]] = {}


def _local_index(dataset: str) -> dict[str, Path]:
    if dataset in _INDEX_CACHE:
        return _INDEX_CACHE[dataset]
    vids = (".mp4", ".avi", ".mov", ".mkv")
    if dataset == "le2i":
        idx = _index(cfg.LE2I_DIR, vids)
    elif dataset == "caucafall":
        idx = _index(cfg.CAUCA_DIR, None, dirs=True)   # frame folders, not files
    elif dataset == "GMDCSA24":
        idx = _index(cfg.GMDCSA_DIR, vids)
    else:
        # Datasets with no dataset-specific home: search the two OmniFall
        # trees and then data/ itself. EDF and OCCU sit under video/ nested as
        # EDF/EDF/EDF/<subject>/..., which rglob handles.
        idx = {}
        # The companion package's own tree first -- `omnifall prepare` writes
        # the canonical layout there and verifies it against the Hub labels, so
        # it is the most trustworthy copy when several exist.
        roots = [cfg.OMNIFALL_CACHE / "videos" / dataset / "video",
                 cfg.OMNIFALL_VIDEO, cfg.DATA_DIR]
        for root in roots:
            if root.name == "video":
                if root.exists():
                    idx = _index(root, vids)
                    if idx:
                        break
                continue
            for name in (dataset, dataset.lower(), dataset.upper()):
                cand = root / name
                if cand.exists():
                    idx = _index(cand, vids)
                    break
            if idx:
                break
    _INDEX_CACHE[dataset] = idx
    return idx


def resolve(dataset: str, path: str) -> Path | None:
    """Local file (or frame folder) for one OmniFall `(dataset, path)` pair.

    Tries the most specific key first -- the last three path components, then
    two, then the leaf -- so `Subject_1/ADL/01` cannot silently match another
    subject's `01`.
    """
    idx = _local_index(dataset)
    if not idx:
        return None
    parts = [p for p in str(path).split("/") if p]

    # CAUCAFall is the one dataset where folding separators is not enough: the
    # subject lives in OmniFall's *leaf* (`adl/HopS1`) and in the download's
    # *parent* (`Subject.1/Hop`), so the key has to be rebuilt rather than
    # normalised. Split the trailing S<digits> off and reassemble.
    if dataset == "caucafall" and parts:
        leaf = parts[-1]
        head, sep, tail = leaf.rpartition("S")
        if sep and tail.isdigit() and head:
            hit = idx.get(_key(f"Subject{tail}{head}"))
            if hit is not None:
                return hit

    for n in (3, 2, 1):
        if len(parts) >= n:
            hit = idx.get(_key("".join(parts[-n:])))
            if hit is not None:
                return hit
    return None


def coverage(datasets=STAGED, root: Path = OMNIFALL_DIR) -> pd.DataFrame:
    """How many of each dataset's videos are actually on this machine.

    OmniFall is annotations; this is the reality check. Anything at 0% needs
    downloading from its original source before it can contribute a single
    training window.
    """
    labels = load_labels(datasets, root)
    rows = []
    for ds in datasets:
        paths = sorted(labels[labels.dataset == ds].path.astype(str).unique())
        if not paths:
            continue
        found = sum(1 for p in paths if resolve(ds, p) is not None)
        # Pose is the column that actually gates training: a video with cached
        # keypoints needs no extraction, and one without needs both the file
        # and an hour of GPU. They come apart -- OOPS has pose here and no
        # video at all.
        posed = sum(1 for p in paths if pose_path(ds, p) is not None)
        rows.append({"dataset": ds, "videos": len(paths), "video": found,
                     "pose": posed, "missing": len(paths) - max(found, posed),
                     "coverage": max(found, posed) / len(paths),
                     "contaminates": EVAL_OVERLAP.get(ds, "")})
    return pd.DataFrame(rows)


def contaminates(datasets) -> list[str]:
    """Which of this project's evaluations a training selection would spoil."""
    return sorted({EVAL_OVERLAP[d] for d in datasets if d in EVAL_OVERLAP})


# ---------------------------------------------------------------------------
# The pre-extracted pose cache
# ---------------------------------------------------------------------------
def video_id(dataset: str, path: str) -> str:
    """Stable, filesystem-safe id for one OmniFall video.

    Prefixed so ingested OmniFall keypoints can share `KEYPOINTS_DIR` with the
    Kaggle corpus without any chance of collision, and so provenance is legible
    from a filename alone.
    """
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(path))
    return f"{cfg.OMNIFALL_PREFIX}{dataset}__{slug}"


def pose_path(dataset: str, path: str) -> Path | None:
    """Locate the parquet holding one video's detections, or None.

    The cache flattens `a/b/c` to `a__b__c.parquet`; a couple of datasets were
    written with a different separator, so fall back to a folded-name scan
    rather than failing on punctuation.
    """
    d = cfg.OMNIFALL_POSE / dataset
    if not d.exists():
        return None
    direct = d / (str(path).replace("/", "__") + ".parquet")
    if direct.exists():
        return direct
    want = _key(str(path))
    for p in d.glob("*.parquet"):
        if _key(p.stem) == want:
            return p
    return None


def to_keypoints(df: pd.DataFrame) -> np.ndarray:
    """Parquet detections -> (F, 17, 3) as (y_norm, x_norm, conf).

    Two conversions happen here and nowhere else. The cache stores (x, y) in
    pixels; this project stores (y, x) divided by frame width and height. And
    the cache keeps every detection in a frame, where this project keeps only
    the largest bounding box -- `extract._largest_person`, reproduced here so a
    video ingested from parquet is scored on the same person a video extracted
    directly would have been.

    Frames with no detection stay all-zero, matching `extract.extract_video`.
    """
    n = int(df.n_frames.iloc[0]) if len(df) else 0
    out = np.zeros((n, cfg.N_KEYPOINTS, 3), dtype=np.float32)
    if n == 0 or df.empty:
        return out

    w = float(df.img_w.iloc[0]) or 1.0
    h = float(df.img_h.iloc[0]) or 1.0

    area = (df.bx2 - df.bx1) * (df.by2 - df.by1)
    pick = df.assign(_area=area).sort_values("_area").groupby("frame_idx").tail(1)

    idx = pick.frame_idx.to_numpy()
    keep = (idx >= 0) & (idx < n)
    idx = idx[keep]
    vals = (pick.loc[keep, _KP_COLS].to_numpy(dtype=np.float32)
            .reshape(-1, cfg.N_KEYPOINTS, 3))
    out[idx, :, 0] = np.clip(vals[:, :, 1] / h, 0.0, 1.0)   # y first
    out[idx, :, 1] = np.clip(vals[:, :, 0] / w, 0.0, 1.0)   # then x
    out[idx, :, 2] = vals[:, :, 2]
    return out


def ingest(datasets=None, scale: str = cfg.DEFAULT_SCALE, overwrite: bool = False,
           progress: bool = True) -> pd.DataFrame:
    """Convert the parquet pose cache into this project's .npy layout.

    After this runs, every OmniFall video is indistinguishable from a Kaggle one
    as far as `data.py`, `train.py` and `evaluate.py` are concerned -- same
    directory, same array convention, same loader. That is the point: one ingest
    step instead of a bespoke adapter per source dataset.

    Returns the manifest, also written to `cfg.OMNIFALL_MANIFEST`: one row per
    video with its id, source dataset, subject, camera, frame count and fps.
    The fps matters because OmniFall's segment boundaries are in *seconds*.
    """
    datasets = list(datasets or (STAGED + ITW))
    out_dir = cfg.KEYPOINTS_DIR / scale
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for ds in datasets:
        d = cfg.OMNIFALL_POSE / ds
        if d.exists():
            files += [(ds, p) for p in sorted(d.glob("*.parquet"))]

    it = files
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(files, desc="ingest omnifall", unit="vid")

    rows = []
    for ds, p in it:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if df.empty:
            continue
        rel = str(df.path.iloc[0])
        vid = video_id(ds, rel)
        dest = out_dir / f"{vid}.npy"
        if dest.exists() and not overwrite:
            arr = np.load(dest, mmap_mode="r")
        else:
            arr = to_keypoints(df)
            np.save(dest, arr)
        rows.append({
            "video_id": vid, "dataset": ds, "path": rel,
            "subject": int(df.subject.iloc[0]) if "subject" in df else -1,
            "cam": int(df.cam.iloc[0]) if "cam" in df else -1,
            "n_frames": int(len(arr)),
            "fps": float(df.src_fps.iloc[0]) if "src_fps" in df else 30.0,
            "mean_conf": float(np.asarray(arr)[:, :, 2].mean()) if len(arr) else 0.0,
        })

    man = pd.DataFrame(rows)

    # A partial ingest must not erase the datasets it did not touch. Writing
    # `man` straight out looks harmless -- the .npy cache is untouched -- but it
    # drops every other dataset from the index, so `build_clips` silently yields
    # windows for a fraction of the corpus and the training set quietly shrinks.
    # Rows for the datasets just processed win; everything else is carried over.
    if cfg.OMNIFALL_MANIFEST.exists():
        try:
            prev = pd.read_csv(cfg.OMNIFALL_MANIFEST)
            keep = prev[~prev.dataset.isin(datasets)]
            man = pd.concat([keep, man], ignore_index=True)
        except Exception:
            pass                      # an unreadable index is worth replacing
    man = man.drop_duplicates("video_id", keep="last").sort_values(
        ["dataset", "video_id"], ignore_index=True)

    cfg.OMNIFALL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(cfg.OMNIFALL_MANIFEST, index=False)
    return man


def load_manifest() -> pd.DataFrame:
    if not cfg.OMNIFALL_MANIFEST.exists():
        raise FileNotFoundError(
            f"{cfg.OMNIFALL_MANIFEST} missing -- call omnifall.ingest() first")
    return pd.read_csv(cfg.OMNIFALL_MANIFEST)


# ---------------------------------------------------------------------------
# Segments -> windows
# ---------------------------------------------------------------------------
def frame_labels(segments: pd.DataFrame, n_frames: int, fps: float,
                 fall_ids=FALL_IDS) -> np.ndarray:
    """Per-frame binary label for one video, from its segment rows.

    Frames no segment covers stay 0. That is the right default here because
    OmniFall's annotation is dense over the part of the video it describes;
    an uncovered frame is one nobody called a fall.
    """
    out = np.zeros(n_frames, dtype=np.int8)
    for _, s in segments.iterrows():
        if int(s.label) not in fall_ids:
            continue
        a = max(0, int(round(float(s.start) * fps)))
        b = min(n_frames, int(round(float(s.end) * fps)) + 1)
        if b > a:
            out[a:b] = 1
    return out


def window_label(frame_lab: np.ndarray, start: int, seq_len: int,
                 frame_stride: int, min_overlap: float = 0.5,
                 margin: int = 0) -> int | None:
    """Label for one window, or None when it is too ambiguous to use.

    A window is Fall when at least `min_overlap` of its frames are inside a
    fall segment, and No-Fall when none of them are. Anything between is
    discarded -- the same zone logic notebooks 06/08/09 use, and for the same
    reason: a window straddling the start of a fall is neither a clean positive
    nor a clean negative, and training on it teaches the boundary rather than
    the event.

    `margin` widens the exclusion zone in *original* frames around any fall, so
    a No-Fall window must clear the event by that much to count.
    """
    a = start * frame_stride
    b = a + seq_len * frame_stride
    seg = frame_lab[a:b]
    if not len(seg):
        return None
    frac = float(seg.mean())
    if frac >= min_overlap:
        return 1
    if frac == 0.0:
        if margin:
            lo, hi = max(0, a - margin), min(len(frame_lab), b + margin)
            if frame_lab[lo:hi].any():
                return None
        return 0
    return None


def build_clips(manifest: pd.DataFrame | None = None,
                labels: pd.DataFrame | None = None,
                datasets=None,
                seq_len: int = cfg.FINAL_TRAIN.seq_len,
                stride: int = cfg.EVAL.sliding_stride,
                frame_stride: int | None = None,
                fall_ids=FALL_IDS,
                min_overlap: float = 0.5,
                margin_sec: float = 1.0,
                one_per_video: bool = False) -> pd.DataFrame:
    """Sliding windows with labels transferred from OmniFall's segments.

    Produces the same table every cross-dataset adapter in this project
    produces -- `video_id`, `start` in strided index space, `label` -- so
    `evaluate.score_clips` and `data.KeypointClipDataset` consume it unchanged.

    The difference is where the labels come from. `le2i.py`, `caucafall.py` and
    `gmdcsa.py` each reverse-engineered a fall's extent from that dataset's own
    annotation quirks, and the CAUCAFall attempt was wrong on the first pass.
    Here the extent is given, in seconds, by an annotator working to one
    published 16-class definition across all eight datasets.

    `margin_sec` is in seconds rather than frames because OmniFall mixes frame
    rates -- the same reason `config.GMDCSA_MARGIN_SEC` is.

    `one_per_video` keeps a single centred window per video, matching the
    random-clip protocol, for when the comparison target is a per-video score.
    """
    from .data import sliding_starts

    man = load_manifest() if manifest is None else manifest
    if datasets is not None:
        man = man[man.dataset.isin(list(datasets))]
    # Load labels for whatever the manifest actually contains: defaulting to
    # STAGED silently dropped every OOPS window, which is the one component
    # here that is real, in-the-wild, and not one of our benchmarks.
    lab = load_labels(sorted(man.dataset.unique())) if labels is None else labels

    fs = cfg.PREPROCESS.frame_stride if frame_stride is None else frame_stride
    seg_by = {k: v for k, v in lab.groupby(["dataset", "path"])}

    rows = []
    for _, v in man.iterrows():
        segs = seg_by.get((v.dataset, v.path))
        if segs is None or not v.n_frames:
            continue
        fps = float(v.fps) or cfg.TRAIN_NOMINAL_FPS
        fl = frame_labels(segs, int(v.n_frames), fps, fall_ids)
        n_seq = -(-int(v.n_frames) // fs)
        margin = int(round(margin_sec * fps))

        starts = ([max(0, (n_seq - seq_len) // 2)] if one_per_video
                  else sliding_starts(n_seq, seq_len, stride))
        for s in starts:
            y = window_label(fl, int(s), seq_len, fs, min_overlap, margin)
            if y is None:
                continue
            rows.append({"video_id": v.video_id, "start": int(s), "label": y,
                         "dataset": v.dataset, "subject": v.subject,
                         "cam": v.cam, "path": v.path})

    return pd.DataFrame(rows)


MULTICLASS = ("walk", "fall", "fallen", "sit_down", "sitting", "lie_down",
              "lying", "stand_up", "standing", "other")
#: OmniFall ids 0-9 already *are* those ten classes in order, so a model's
#: output index equals the label id and no lookup table is needed. Ids 10-15
#: (kneel_down, kneeling, squat_down, squatting, crawl, jump) carry 0.45% of the
#: annotated time between them -- too little to learn -- and fold into `other`.
OTHER_ID = 9


def build_clips_multiclass(manifest: pd.DataFrame | None = None,
                           labels: pd.DataFrame | None = None,
                           datasets=None,
                           seq_len: int = cfg.FINAL_TRAIN.seq_len,
                           stride: int = cfg.EVAL.sliding_stride,
                           frame_stride: int | None = None,
                           min_labelled: float = 0.5,
                           min_purity: float = 0.0) -> pd.DataFrame:
    """Sliding windows labelled with the activity that dominates each one.

    The binary `build_clips` asks "does a fall overlap this window", with a
    margin that *drops* the ambiguous boundary windows. That is right for a fall
    detector and wrong here: a multi-class model needs to see `sit_down` and
    `stand_up` as their own classes, and those are exactly the windows the
    binary margin throws away.

    So the rule is simpler -- each window takes the class covering most of its
    span. `min_labelled` discards windows the annotation barely covers;
    `min_purity` optionally discards mixed windows too, at the cost of removing
    most transition examples, which is usually the wrong trade.
    """
    from .data import sliding_starts

    man = load_manifest() if manifest is None else manifest
    if datasets is not None:
        man = man[man.dataset.isin(list(datasets))]
    lab = load_labels(sorted(man.dataset.unique())) if labels is None else labels

    fs = cfg.PREPROCESS.frame_stride if frame_stride is None else frame_stride
    seg_by = {k: v for k, v in lab.groupby(["dataset", "path"])}

    rows = []
    for _, v in man.iterrows():
        segs = seg_by.get((v.dataset, v.path))
        if segs is None or not v.n_frames:
            continue
        fps = float(v.fps) or cfg.TRAIN_NOMINAL_FPS
        n = int(v.n_frames)

        frame_cls = np.full(n, -1, dtype=np.int16)
        for _, sg in segs.iterrows():
            a = int(round(float(sg.start) * fps))
            b = int(round(float(sg.end) * fps))
            frame_cls[max(0, a):min(n, b)] = int(sg.label)

        n_seq = -(-n // fs)
        strided = frame_cls[::fs][:n_seq]
        for st in sliding_starts(n_seq, seq_len, stride):
            w = strided[st:st + seq_len]
            w = w[w >= 0]
            if len(w) < seq_len * min_labelled:
                continue
            vals, counts = np.unique(w, return_counts=True)
            if counts.max() / len(w) < min_purity:
                continue
            top = int(vals[counts.argmax()])
            rows.append({
                "video_id": v.video_id, "start": int(st),
                "label": top if top < len(MULTICLASS) else OTHER_ID,
                "purity": float(counts.max() / len(w)),
                "dataset": v.dataset, "subject": v.subject,
                "cam": v.cam, "path": v.path,
            })

    return pd.DataFrame(rows)


def split_video_ids(datasets=None, protocol: str = "cs",
                    root: Path = OMNIFALL_DIR) -> dict[str, list[str]]:
    """OmniFall's own train/val/test partitions, as this project's video ids.

    Uses the published cross-subject or cross-view lists rather than inventing a
    split, which is the whole reason for adopting OmniFall's protocol: the
    partitions are the ones its published numbers were measured on.

    The lists are returned exactly as published, and they are not always
    disjoint. In this release up_fall's `train` already contains every `val`
    path and 132 of its 264 `test` paths (verified against the Hub), so a caller
    that builds partitions must subtract whatever it holds out for selection --
    notebook 14 does, and `leave_one_dataset_out` below does.
    """
    datasets = list(datasets or STAGED)
    out = {"train": [], "val": [], "test": []}
    for ds in datasets:
        sp = load_split(ds, protocol, root)
        for part, paths in sp.items():
            out[part] += [video_id(ds, p) for p in paths]
    return out


def leave_one_dataset_out(held_out: str, datasets=None) -> dict[str, list[str]]:
    """Train on every dataset except one, test on that one.

    OmniFall's cross-domain protocol in miniature, and the only honest option
    while most of the staged corpus is unavailable: it keeps the held-out
    dataset genuinely unseen, which a cross-subject split within one dataset
    does not. Validation is carved from the training datasets' own val lists so
    nothing from the held-out domain is used for model selection.
    """
    datasets = [d for d in (datasets or STAGED) if d != held_out]
    train, val = [], []
    for ds in datasets:
        sp = load_split(ds, "cs")
        # The published lists can overlap -- up_fall's train already holds its
        # whole val list and half its test list -- so the val slice is removed
        # from the training material explicitly. See split_video_ids. A
        # dataset's own test videos are still training material when that
        # dataset is not the one held out; only the val slice must stay clean.
        val_ids = {video_id(ds, p) for p in sp["val"]}
        train += [video_id(ds, p) for p in sp["train"] + sp["test"]
                  if video_id(ds, p) not in val_ids]
        val += [video_id(ds, p) for p in sp["val"]]

    held = load_labels([held_out])
    test = [video_id(held_out, p) for p in sorted(held.path.astype(str).unique())]
    return {"train": train, "val": val, "test": test}
