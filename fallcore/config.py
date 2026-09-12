"""Paths and the paper's hyperparameters, in one place.

Everything here is transcribed from Benabdennour et al., "Real-Time Human Fall
Detection From Video Using YOLOv11 With Pose Estimation", IEEE Access vol. 14
(2026). Section numbers in the comments point back at the source so a value can
be argued with rather than guessed at.

The one substantive departure: the paper's backbone is YOLOv11n-pose, ours is
YOLO26-pose at three scales. The classifier is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR      = ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"           # the Kaggle compilation lands here
KEYPOINTS_DIR = DATA_DIR / "keypoints"     # <scale>/<video_id>.npy
# Person bounding boxes, cached separately because only the rule-based baseline
# in notebook 11 needs them; every other notebook reads keypoints alone.
BOXES_DIR     = DATA_DIR / "boxes"         # <scale>/<video_id>.npy
MANIFEST_CSV  = DATA_DIR / "manifest.csv"
SPLITS_JSON   = DATA_DIR / "splits.json"

# Drop-box for one-off clips to run through notebook 12. Deliberately at
# the repo root rather than under data/: nothing here is part of the corpus,
# it never reaches the manifest or a split, and it is not meant to be.
VIDEOS_DIR    = ROOT / "videos"

# ---------------------------------------------------------------------------
# OmniFall (Schneider et al., HF `simplexsigil2/omnifall`)
#
# A unified 16-class taxonomy with dense temporal segments over eight staged
# datasets, 818 in-the-wild OOPS clips and 12k synthetic videos. The Hub ships
# annotations only; video and pose have to be assembled locally. Everything
# lives under data/omnifall/ so the corpus travels with the repo:
#
#   labels/            per-dataset segment CSVs        (small, tracked in git)
#   splits/            the cs / cv partition lists     (small, tracked in git)
#   pose/<ds>/*.parquet    extracted poses, one file per video
#   video/<DS>/...         raw video fetched outside the companion package
#   cache/videos/<ds>/video/   what `omnifall prepare <ds>` writes
#
# The parquet pose cache was extracted with the same yolo26n-pose weights at
# the same imgsz this project uses -- verified against our own LE2I extraction
# at 97.8% detection agreement and a mean coordinate difference under two
# pixels.
# ---------------------------------------------------------------------------
OMNIFALL_DIR    = DATA_DIR / "omnifall"
OMNIFALL_POSE   = OMNIFALL_DIR / "pose"          # <dataset>/<path>.parquet
OMNIFALL_VIDEO  = OMNIFALL_DIR / "video"         # <DATASET>/... raw video
OMNIFALL_MANIFEST = DATA_DIR / "omnifall_manifest.csv"

# The `omnifall` companion package writes
#   <root>/videos/<component>/video/<path>.mp4
# and reads its root from OMNIFALL_ROOT. Pointing that at data/omnifall/cache
# keeps a `omnifall prepare` inside the repo like everything else; set the
# environment variable to override when the repo sits on a small disk.
import os as _os
OMNIFALL_CACHE = Path(_os.environ.get("OMNIFALL_ROOT", OMNIFALL_DIR / "cache"))

# Ingested OmniFall keypoints share KEYPOINTS_DIR with the Kaggle corpus. Their
# ids all begin with this prefix, so the two never collide and provenance is
# readable from the filename alone.
OMNIFALL_PREFIX = "of_"

RUNS_DIR    = ROOT / "runs"
CKPT_DIR    = RUNS_DIR / "checkpoints"
METRICS_DIR = RUNS_DIR / "metrics"
FIGURES_DIR = RUNS_DIR / "figures"

# A yolo26n-pose checkpoint already sits in the demo app's models folder.
# Reuse it instead of re-downloading; ultralytics fetches the rest on demand.
LOCAL_WEIGHTS_DIR = ROOT / "demo-cam" / "models"


def ensure_dirs() -> None:
    """Create every output directory. Safe to call repeatedly."""
    for d in (DATA_DIR, RAW_DIR, KEYPOINTS_DIR, BOXES_DIR, VIDEOS_DIR,
              RUNS_DIR, CKPT_DIR, METRICS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pose backbones
#
# Params/GFLOPs are the summaries recorded in ultralytics' own
# cfg/models/26/yolo26-pose.yaml, not measured here. The l and x scales are
# listed so the sweep can be widened without hunting for numbers; the default
# sweep is n/s/m.
# ---------------------------------------------------------------------------
POSE_BACKBONES: dict[str, dict] = {
    "yolo26n": {"weights": "yolo26n-pose.pt", "params": 3_747_554,  "gflops": 10.7},
    "yolo26s": {"weights": "yolo26s-pose.pt", "params": 11_870_498, "gflops": 29.6},
    "yolo26m": {"weights": "yolo26m-pose.pt", "params": 24_344_482, "gflops": 85.9},
    "yolo26l": {"weights": "yolo26l-pose.pt", "params": 28_747_938, "gflops": 104.3},
    "yolo26x": {"weights": "yolo26x-pose.pt", "params": 62_914_350, "gflops": 226.3},
}

SWEEP_SCALES = ["yolo26n", "yolo26s", "yolo26m"]
DEFAULT_SCALE = "yolo26n"

# COCO-17 keypoint names, in the order ultralytics emits them. Used by the
# mechanistic analysis to label saliency and by the skeleton overlay check.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# COCO skeleton edges (indices into KEYPOINT_NAMES) for drawing.
SKELETON_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]

# Left/right hip indices, used by the hip-centred feature variant.
LEFT_HIP, RIGHT_HIP = 11, 12

N_KEYPOINTS = 17
N_FEATURES = N_KEYPOINTS * 3  # 51 = 17 keypoints x (y_norm, x_norm, conf)


# ---------------------------------------------------------------------------
# Preprocessing (Section III-A-3, Algorithm 1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessConfig:
    # Videos whose mean keypoint confidence falls below this are dropped
    # entirely. The paper filtered 118 of 6,988 clips this way.
    min_mean_confidence: float = 0.2
    # Ultralytics inference size for the frozen backbone.
    imgsz: int = 640
    # Read at most this many frames from the head of each video, then take
    # every frame_stride-th one. 120/2 -> 60 frames, per Algorithm 1 line 6.
    max_frames: int = 120
    frame_stride: int = 2


PREPROCESS = PreprocessConfig()


# ---------------------------------------------------------------------------
# Data partitioning (Section III-A-4)
# ---------------------------------------------------------------------------
SPLIT_SEED = 99            # the paper's fixed seed, kept so splits are stable
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)

# Counts reported in the paper, used as sanity gates in notebook 00. If the
# ingest does not reproduce these, the later notebooks are measuring something
# other than what the paper measured.
PAPER_TOTAL_CLIPS = 6988
PAPER_FALL_CLIPS = 3140
PAPER_NOFALL_CLIPS = 3848
PAPER_FILTERED_CLIPS = 118      # dropped by the confidence filter
PAPER_SPLIT_SIZES = (5496, 687, 687)


# ---------------------------------------------------------------------------
# Model architecture (Section III-B-2, equations 1-9)
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    # "sinusoidal" | "learned" | "none"   (ablated in Section IV-A-1)
    positional_encoding: str = "sinusoidal"
    # "mean" | "max" | "last"
    pooling: str = "mean"
    # "relu" | "gelu" | "elu" | "leaky_relu" | "silu"
    activation: str = "relu"
    # Input feature variant, see fallcore.data.build_features:
    #   "full"      51 dims - (y, x, conf) per keypoint       [paper default]
    #   "coords"    34 dims - (y, x) only, confidence dropped
    #   "velocity"  85 dims - full + frame-to-frame deltas of (y, x)
    #   "centered"  51 dims - hip-centred, scale-normalised coordinates
    #   "accel", "angles" and "kinematic" add explicit derivatives/articulation;
    #   they are implemented and correct but lost every arm of the picam test,
    #   so they are not for use without new evidence (see data.build_features).
    features: str = "full"
    # Output width. 1 keeps the paper's single fall logit and the 1,208,897
    # parameter count. Above 1 switches the head to multi-class over OmniFall's
    # activity labels, where output index i *is* OmniFall's label id i, so
    # index 1 is `fall` and index 2 is `fallen`; see omnifall.MULTICLASS.
    n_classes: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


# The configuration the paper ships (Section III-B-2): 1,208,897 parameters.
FINAL_MODEL = ModelConfig()

# The shared starting point for all 54 ablation runs (Section IV-A), which the
# paper reports at F1 = 0.9449.
ABLATION_BASELINE_MODEL = ModelConfig(d_model=128)


# ---------------------------------------------------------------------------
# Training (Table 7, Algorithm 1)
#
# One deliberate departure: the epoch budget is 300 with patience 50, where the
# paper writes 100 with patience 25. The cached runs under `runs/metrics/` were
# trained at 300/50, and notebook 10 pins 100/25 so its cached comparisons stay
# comparable. Selection is not affected -- the final run's best epoch is 40,
# inside either budget -- but this is not the paper's schedule down to the epoch.
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # T. At frame_stride 2 this is 60 strided frames spanning 120 original
    # frames, i.e. 4 seconds at the nominal 30 fps. The paper's prose calls it
    # "2 seconds", but its Algorithm 1 says first 120 frames / every 2nd and the
    # rest of this project (infer.TRAIN_WINDOW_SECONDS) reads the window as 4 s.
    seq_len: int = 60
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-5
    max_epochs: int = 300
    patience: int = 50          # early stopping on validation F1
    seed: int = 99
    loss: str = "bce"           # "bce" | "focal" | "ce"  (ce = multi-class)
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    # Multi-class only. "balanced" weights each class by the inverse of its
    # frequency, which is not optional here: `lie_down` is 0.55% of the training
    # windows against `walk`'s 24.5%, a 45:1 imbalance, and it is precisely the
    # class the binary models get most wrong.
    class_weight: str | None = None
    num_workers: int = 0        # keypoints are tiny and already in RAM
    device: str = "cuda"
    # Robustness options, off by default so the paper's recipe is reproduced
    # exactly. See fallcore.data.fill_detection_gaps for why they exist.
    fill_gaps: bool = False
    frame_dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


FINAL_TRAIN = TrainConfig()

# Ablation baseline (Section IV-A): lower lr, shorter window, 25-epoch cap.
# patience == max_epochs is the paper's own combination ("up to 25 epochs ...
# with early stopping" at "patience of 25 epochs"), and it means early stopping
# can never fire -- every configuration runs the full 25. Kept faithful rather
# than corrected, because the sweep's numbers depend on it.
ABLATION_BASELINE_TRAIN = TrainConfig(lr=1e-4, seq_len=45, max_epochs=25,
                                      patience=25)


# ---------------------------------------------------------------------------
# Evaluation (Section III-C-3, Algorithm 2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalConfig:
    threshold: float = 0.5
    sliding_stride: int = 15    # frames between window starts
    # Algorithm 2 returns Fall on the first window over threshold rather than
    # scoring every window. That early exit is what biases the protocol toward
    # recall, and dropping it would change the reported numbers.
    early_exit: bool = True


EVAL = EvalConfig()


# ---------------------------------------------------------------------------
# Efficiency benchmarking (Section IV-D)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchConfig:
    trials: int = 3
    runs_per_trial: int = 200
    warmup: int = 50
    iqr_multiplier: float = 1.5


BENCH = BenchConfig()


# ---------------------------------------------------------------------------
# Cross-dataset evaluation on LE2I (Section IV-E)
# ---------------------------------------------------------------------------
LE2I_DIR = DATA_DIR / "le2i"
# Clips must end this many frames before the annotated fall start to count as
# unambiguous No-Fall. Clips that merely overlap the event are discarded.
LE2I_SAFETY_MARGIN = 30


# ---------------------------------------------------------------------------
# Cross-dataset evaluation on CAUCAFall (Eraso et al., Univ. del Cauca)
# ---------------------------------------------------------------------------
# 10 subjects x 10 activities (5 fall types, 5 ADLs) = 100 sequences, shipped as
# per-frame PNGs with one YOLO-format .txt label per frame.
CAUCA_DIR = DATA_DIR / "caucafall"

# CAUCAFall records at 20 fps against 25-30 fps in the training corpus, so a
# window of the same frame count covers more elapsed time here. Kept explicit
# because it is the confound most likely to be forgotten when reading a
# cross-dataset score.
CAUCA_FPS = 20.0
TRAIN_NOMINAL_FPS = 30.0

# Smaller than LE2I_SAFETY_MARGIN because these clips are shorter (median 203
# frames) and slower-framed; 30 frames of a 20 fps clip is 1.5 seconds.
CAUCA_SAFETY_MARGIN = 20


# ---------------------------------------------------------------------------
# Cross-dataset evaluation on GMDCSA24 (Alam et al., Data in Brief 2024)
# ---------------------------------------------------------------------------
# 160 videos, 4 subjects, split into Fall/ and ADL/ folders with per-subject
# CSV annotations. Brings hard negatives the other sets lack -- Sleeping,
# Exercising, Reading -- which is the failure mode CAUCAFall exposed.
GMDCSA_DIR = DATA_DIR / "gmdcsa24"

# In SECONDS, not frames: this corpus mixes ~30 fps and 15 fps videos, so a
# margin expressed in frames would mean two different durations depending on
# which video it landed on.
GMDCSA_MARGIN_SEC = 1.0
