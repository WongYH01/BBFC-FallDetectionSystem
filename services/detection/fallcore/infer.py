"""Run the trained detector over one arbitrary video file and render the result.

Every other module in this package works from the cached `.npy` keypoints of a
corpus that has already been scanned, split and labelled. This one starts from a
file on disk that the pipeline has never seen: it runs the pose backbone, slides
the classifier over the sequence, returns a verdict, and draws the whole thing
back onto the frames so the decision can be looked at rather than trusted.

Three things here are easy to get wrong and are handled explicitly:

*Index spaces.* Training subsamples by `PREPROCESS.frame_stride`, so a window of
`seq_len` classifier frames spans `seq_len * frame_stride` *original* frames.
Everything returned by this module is reported in original-frame and second
units, because those are the only ones that mean anything when pointing at a
video. `window_index_frames` is the single place the conversion happens.

*Frame rate.* The classifier was trained on ~30 fps clips, where the paper's
60x2 window covers four seconds of motion. Feed it a 15 fps video at the same
stride and the same 60 frames now cover eight seconds -- a fall spread over
twice as long, which is not what the weights encode. `suggested_frame_stride`
picks the stride that holds the window's *duration* fixed, and `run_pose`
records the fps so the notebook can say when the two disagree.

*Causality.* A window ending at frame t says nothing about frames after t. The
per-frame track in `frame_probabilities` is therefore causal by default: at
frame t it shows the most recent window that had actually finished by then, and
NaN for the opening frames where no window has. The non-causal variant reads
better on a timeline and is available for that, but it is hindsight and is
labelled as such wherever it is drawn.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import config as cfg
from .data import build_features, sliding_starts, take_window
from .extract import _largest_person, load_pose_model, pose_device

# The window duration the classifier was trained at: 60 frames, stride 2, 30 fps.
TRAIN_WINDOW_SECONDS = (
    cfg.FINAL_TRAIN.seq_len * cfg.PREPROCESS.frame_stride / cfg.TRAIN_NOMINAL_FPS
)


# ---------------------------------------------------------------------------
# Stage 1: pose over the raw file
# ---------------------------------------------------------------------------
@dataclass
class PoseTrack:
    """Per-frame pose output for one video, plus the geometry needed to draw it.

    `keypoints` follows the cache convention -- (y_norm, x_norm, conf), y first,
    both normalised -- so anything in `fallcore.data` can consume it unchanged.
    `boxes` is pixel-space (x1, y1, x2, y2, conf) for the same subject, kept
    because drawing wants pixels and the rule-based baseline wants the box.
    An undetected frame is an all-zero row in both.
    """

    path: Path
    keypoints: np.ndarray      # (F, 17, 3)
    boxes: np.ndarray          # (F, 5)
    fps: float
    width: int
    height: int

    @property
    def n_frames(self) -> int:
        return len(self.keypoints)

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    @property
    def detection_rate(self) -> float:
        """Fraction of frames in which the backbone found anybody.

        Worth printing before believing a verdict: a clip the detector loses for
        half its length is being classified largely on zero rows.
        """
        if not self.n_frames:
            return 0.0
        return float((self.keypoints[:, :, 2].sum(axis=1) > 0).mean())


def video_properties(path: str | Path) -> dict:
    """fps / width / height / frame count, straight from the container."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    props = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 0.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        # Container metadata, not a guarantee -- some files lie. The authority
        # is how many frames the decoder actually yields, which is what
        # `run_pose` records.
        "n_frames_reported": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return props


def run_pose(
    video_path: str | Path,
    scale: str = cfg.DEFAULT_SCALE,
    device: str = "cuda",
    imgsz: int = cfg.PREPROCESS.imgsz,
    model=None,
    progress: bool = True,
    fps_override: float | None = None,
) -> PoseTrack:
    """One streaming pass of the frozen backbone over every frame of a file.

    Frames are not retained -- a long clip at 1080p would not fit -- so the
    renderer decodes a second time. Pose dominates the cost by an order of
    magnitude, so the extra decode is not worth avoiding.

    Subject selection is `extract._largest_person`, the same rule the whole
    corpus was cached with. On a crowded scene that is a blunt instrument, and
    the paper says so; it is reproduced here rather than improved so that a
    video scored by this function is scored the way the training data was.
    """
    video_path = Path(video_path)
    props = video_properties(video_path)
    fps = float(fps_override or props["fps"] or cfg.TRAIN_NOMINAL_FPS)

    model = model or load_pose_model(scale, device=device)
    device = pose_device(model, device)
    stream = model.predict(source=str(video_path), stream=True, imgsz=imgsz,
                           device=device, verbose=False)

    if progress:
        from tqdm.auto import tqdm
        stream = tqdm(stream, total=props["n_frames_reported"] or None,
                      desc=f"pose {scale}", unit="frame")

    kps: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    for result in stream:
        h, w = result.orig_shape
        kp = np.zeros((cfg.N_KEYPOINTS, 3), dtype=np.float32)
        bx = np.zeros(5, dtype=np.float32)

        idx = _largest_person(result)
        if idx is not None:
            b = result.boxes
            bx[:4] = b.xyxy[idx].cpu().numpy()
            conf = getattr(b, "conf", None)
            bx[4] = float(conf[idx].cpu()) if conf is not None else 1.0

            k = getattr(result, "keypoints", None)
            if k is not None and k.data is not None and len(k.data) > idx:
                arr = k.data[idx].cpu().numpy()
                kp[:, 0] = np.clip(arr[:, 1] / max(h, 1), 0.0, 1.0)   # y first
                kp[:, 1] = np.clip(arr[:, 0] / max(w, 1), 0.0, 1.0)   # then x
                kp[:, 2] = arr[:, 2]

        kps.append(kp)
        boxes.append(bx)

    if not kps:
        raise RuntimeError(f"decoded zero frames from {video_path}")

    return PoseTrack(
        path=video_path,
        keypoints=np.stack(kps).astype(np.float32),
        boxes=np.stack(boxes).astype(np.float32),
        fps=fps,
        width=props["width"] or int(result.orig_shape[1]),
        height=props["height"] or int(result.orig_shape[0]),
    )


# ---------------------------------------------------------------------------
# Stage 2: slide the classifier
# ---------------------------------------------------------------------------
def suggested_frame_stride(fps: float, seq_len: int = cfg.FINAL_TRAIN.seq_len,
                           target_seconds: float = TRAIN_WINDOW_SECONDS) -> int:
    """The stride that keeps a window covering `target_seconds` at this fps.

    The classifier's unit of evidence is a duration, not a frame count. Holding
    the frame count fixed across frame rates changes the duration, which is the
    same mistake as evaluating a 20 fps corpus with a 30 fps window -- the
    confound `config.CAUCA_FPS` exists to flag.
    """
    return max(1, int(round(fps * target_seconds / seq_len)))


def window_index_frames(start: int, seq_len: int, frame_stride: int) -> tuple[int, int]:
    """Original-frame span [first, last] of a window at strided index `start`.

    The only conversion between the two index spaces in this module. A window
    starting at strided index s reads strided rows s .. s+seq_len-1, which are
    original frames s*fs .. (s+seq_len-1)*fs.
    """
    return start * frame_stride, (start + seq_len - 1) * frame_stride


def _window_plan(n_seq: int, seq_len: int, stride: int,
                 warmup: bool) -> list[tuple[int, int]]:
    """(start, n_real) pairs to score, where n_real is rows of actual sequence.

    Full windows are (s, seq_len) for the usual sliding starts. Warm-up windows
    all start at 0 and grow -- (0, stride), (0, 2*stride), ... -- so the first
    seconds of a clip produce a score instead of a blank. They are padded by
    repeating the last frame, exactly as `take_window` pads a short training
    clip, so they are not out-of-distribution inputs. They *are* weaker
    evidence, and are flagged `partial` so nothing silently treats a window that
    saw half a second as equivalent to one that saw four.
    """
    plan = [(int(s), seq_len) for s in sliding_starts(n_seq, seq_len, stride)]
    if not warmup:
        return plan
    limit = min(seq_len, n_seq)
    partial = [(0, k) for k in range(stride, limit, stride)]
    return partial + plan


@torch.no_grad()
def score_track(
    track: PoseTrack,
    model,
    model_cfg,
    train_cfg,
    frame_stride: int | None = None,
    stride: int = cfg.EVAL.sliding_stride,
    warmup: bool = False,
    device: str | None = None,
    head=None,
):
    """Slide the classifier over a PoseTrack; one row per window.

    Returns a DataFrame with the window's strided start, its original-frame
    span, that span in seconds, and P(fall). Windows are scored in one batch --
    Algorithm 2's early exit is a decision rule applied afterwards in `verdict`,
    not a way of doing less work.

    `frame_stride` defaults to the paper's 2. Pass `suggested_frame_stride(fps)`
    to hold window duration fixed on a video that is not ~30 fps.

    `warmup` adds the growing partial windows described in `_window_plan`. Off
    by default, because every benchmark number in this project was produced
    without them and adding them here would quietly change what "the same
    protocol" means. Turn it on for playback, where a four-second blank at the
    start of a six-second clip is worse than a flagged partial score.

    `head` (a `fallcore.calibrate.ProbeHead`) replaces the checkpoint's own
    classifier with a room-fitted one, scored on the pooled embedding. Same
    windows, same features, one dot product of extra work; the checkpoint must
    be the one the head was fitted for, which `calibrate.validate` checks.
    """
    import pandas as pd

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fs = cfg.PREPROCESS.frame_stride if frame_stride is None else frame_stride
    seq_len = train_cfg.seq_len

    seq = track.keypoints[::fs]
    plan = _window_plan(len(seq), seq_len, stride, warmup)
    # The angle-bearing feature variants need the frame's true shape to measure
    # a joint angle; every other variant ignores it.
    aspect = track.width / track.height if track.height else 1.0
    windows = np.stack([
        build_features(
            take_window(seq[s:s + n_real], seq_len, start=0),
            model_cfg.features, aspect)
        for s, n_real in plan
    ])

    model.to(device).eval()
    x = torch.from_numpy(windows).float().to(device)
    if head is None:
        probs = torch.sigmoid(model(x)).cpu().numpy()
    else:
        pooled = model._pool(model.encoder(model.embed(x))).cpu().numpy()
        probs = np.asarray(head.proba(pooled), dtype=np.float64)

    # Measured over the strided rows the classifier actually saw, and over the
    # *real* rows only: padding repeats the last frame, which is not evidence
    # about whether anybody was there.
    present = (seq[:, :, 2].sum(axis=1) > 0)

    rows = []
    for i, ((s, n_real), p) in enumerate(zip(plan, probs)):
        f0, _ = window_index_frames(s, seq_len, fs)
        # The verdict is available once the last *real* frame has been seen; the
        # padding after it is invented, not waited for. A full window on a short
        # clip can also run past the end, so clamp either way.
        f1 = min((s + n_real - 1) * fs, track.n_frames - 1)
        seen = present[s:s + n_real]
        rows.append({
            "window": i,
            "start_strided": s,
            "start_frame": f0,
            "end_frame": f1,
            "start_sec": f0 / track.fps,
            "end_sec": f1 / track.fps,
            "seen_sec": (f1 - f0 + 1) / track.fps,
            "partial": n_real < seq_len,
            "dropout": float(1.0 - seen.mean()) if len(seen) else 1.0,
            "prob": float(p),
        })
    return pd.DataFrame(rows)


def _ungated(windows, max_dropout: float | None):
    """Windows the classifier is entitled to an opinion about.

    A window whose frames are mostly empty contains nothing that could fall, so
    the number the network returns for it is not a judgement -- it is whatever
    the function happens to map a block of zeros to. Both models have such a
    fixed point and they disagree completely on it: `coords` scores an entirely
    empty window 0.624, over the threshold, while `full` scores it 0.001. That
    is not the models disagreeing about a fall, it is them disagreeing about an
    input neither should be asked to classify.

    Returns (kept, n_gated). `max_dropout=None` disables the gate, which is the
    default everywhere in `fallcore` so the benchmark notebooks keep measuring
    what they measured before.
    """
    if max_dropout is None or "dropout" not in getattr(windows, "columns", []):
        return windows, 0
    kept = windows[windows.dropout <= max_dropout]
    return kept, int(len(windows) - len(kept))


def verdict(windows, threshold: float = cfg.EVAL.threshold,
            early_exit: bool = cfg.EVAL.early_exit,
            ignore_partial: bool = True,
            max_dropout: float | None = None) -> dict:
    """Algorithm 2's video-level decision, with the moment it was made.

    `early_exit` reproduces the paper's "Fall if any window is Fall". The
    reported probability is then the *first* crossing rather than the largest,
    because that is the one a live system would have alarmed on; `peak_prob` is
    carried alongside for when the question is how confident it ever got.

    `ignore_partial` drops the warm-up windows before deciding, so the verdict
    matches what every other notebook in this project computes regardless of
    whether the caller asked for warm-up scores to draw with.

    `max_dropout` gates on detection -- see `_ungated`. When every window is
    gated the result carries `decidable=False`, and **`fall` is False only
    because something has to be returned**: it means "no subject was visible",
    not "no fall occurred". Check `decidable` before reading `fall`.
    """
    if ignore_partial and "partial" in windows.columns:
        full = windows[~windows.partial]
        # A clip shorter than one window has no full window at all; falling back
        # keeps it decidable rather than returning an empty verdict.
        windows = full if len(full) else windows

    windows, n_gated = _ungated(windows, max_dropout)
    if not len(windows):
        return {"fall": False, "decidable": False, "threshold": float(threshold),
                "n_windows": 0, "n_windows_over": 0, "n_windows_gated": n_gated,
                "peak_prob": None, "peak_sec": None, "trigger_window": None,
                "trigger_sec": None, "trigger_frame": None, "prob": None,
                "reason": "no subject visible in any window"}
    windows = windows.reset_index(drop=True)

    probs = windows["prob"].to_numpy()
    over = np.flatnonzero(probs >= threshold)
    peak = int(np.argmax(probs))

    out = {
        "fall": bool(len(over) > 0),
        "decidable": True,
        "threshold": float(threshold),
        "n_windows": int(len(probs)),
        "n_windows_over": int(len(over)),
        "n_windows_gated": n_gated,
        "peak_prob": float(probs[peak]),
        "peak_sec": float(windows.end_sec.iloc[peak]),
        "trigger_window": None,
        "trigger_sec": None,
        "trigger_frame": None,
        "prob": float(probs[peak]),
    }
    if len(over):
        first = int(over[0]) if early_exit else peak
        out.update(
            trigger_window=first,
            # The alarm can only be raised once the window has *finished*, so
            # the honest latency is the end of the window, not its start.
            trigger_sec=float(windows.end_sec.iloc[first]),
            trigger_frame=int(windows.end_frame.iloc[first]),
            prob=float(probs[first]),
        )
    return out


def events(windows, threshold: float = cfg.EVAL.threshold,
           merge_gap_sec: float = 1.0, min_duration_sec: float = 0.0,
           ignore_partial: bool = True, max_dropout: float | None = None):
    """Group over-threshold windows into candidate events with timestamps.

    `verdict` answers "did a fall occur in this file", which is the paper's
    question because the paper's clips are single-event and about six seconds
    long. On a long recording that question degenerates: with early exit, one
    false positive among a hundred windows decides the whole file, and a
    correct "yes" tells you nothing about where to look.

    This returns one row per contiguous run of over-threshold windows instead --
    when it started, when it stopped, how confident it got. Runs separated by
    less than `merge_gap_sec` are joined, since a single window dipping under
    the threshold mid-fall is noise rather than the end of an event.

    `min_duration_sec` drops events shorter than one window's worth of
    sustained evidence; raise it to suppress isolated single-window blips.
    """
    import pandas as pd

    cols = ["event", "start_sec", "end_sec", "duration_sec",
            "peak_prob", "peak_sec", "n_windows"]
    if ignore_partial and "partial" in windows.columns:
        windows = windows[~windows.partial]
    windows, _ = _ungated(windows, max_dropout)
    w = windows[windows.prob >= threshold].sort_values("end_sec")
    if not len(w):
        return pd.DataFrame(columns=cols)

    runs: list[list] = []
    for _, row in w.iterrows():
        # `end_sec` is when the alarm could be raised; `start_sec` is when the
        # evidence for it began. An event spans from the latter to the former.
        if runs and row.end_sec - runs[-1][-1].end_sec <= merge_gap_sec:
            runs[-1].append(row)
        else:
            runs.append([row])

    rows = []
    for i, run in enumerate(runs):
        probs = [r.prob for r in run]
        best = int(np.argmax(probs))
        start, end = float(run[0].end_sec), float(run[-1].end_sec)
        if end - start + 1e-9 < min_duration_sec:
            continue
        rows.append({
            "event": len(rows),
            "start_sec": round(start, 2),
            "end_sec": round(end, 2),
            "duration_sec": round(end - start, 2),
            "peak_prob": round(float(probs[best]), 4),
            "peak_sec": round(float(run[best].end_sec), 2),
            "n_windows": len(run),
        })
    return pd.DataFrame(rows, columns=cols)



def explain_window(track: PoseTrack, start_frame: int, end_frame: int) -> dict:
    """Decompose one window into the geometry a fall detector could key on.

    The classifier reports a probability and nothing else, so a false alarm is
    unreadable without this. Four numbers, all computed from the same keypoints
    the model was given, comparing the window's first fifth against its last:

    ``detected``   fraction of frames holding a person at all. Low values mean
                   the alarm may be about absence rather than motion -- see the
                   detection gate in notebook 13.
    ``drift``      how far the body centroid travelled down the frame. Positive
                   is downward. Falls do this; so does walking toward a camera
                   mounted above head height, because perspective moves an
                   approaching subject down the image plane.
    ``d_height``   change in the body's vertical extent. This is what separates
                   those two cases. A fall goes upright to horizontal, so the
                   extent *shrinks*; an approaching walker keeps their posture
                   and grows. Measured on the picam set, fall windows average
                   -0.137 and only 19% grow at all.
    ``fastest``    the largest downward centroid move across any half-second,
                   i.e. the rate rather than the distance. A fall and a slow
                   approach can cover the same distance at very different speeds.

    A window with a fall-sized ``drift`` but a positive ``d_height`` is the
    signature of an approach, not a fall -- and that is a conclusion available
    from the model's own input, which is what makes it a model failure rather
    than a missing sensor.
    """
    f0, f1 = max(0, int(start_frame)), min(track.n_frames - 1, int(end_frame))
    kp = track.keypoints[f0:f1 + 1]
    conf = kp[:, :, 2] > 0
    seen = conf.sum(axis=1) > 0
    out = {"start_frame": f0, "end_frame": f1,
           "start_sec": f0 / track.fps, "end_sec": f1 / track.fps,
           "detected": float(seen.mean())}
    if seen.sum() < 5:
        return {**out, "drift": float("nan"), "d_height": float("nan"),
                "fastest": float("nan")}

    with np.errstate(invalid="ignore"):
        ys = np.where(conf, kp[:, :, 0], np.nan)
        cy = np.nanmean(ys, axis=1)[seen]
        h = (np.nanmax(ys, axis=1) - np.nanmin(ys, axis=1))[seen]

    k = max(1, len(cy) // 5)                      # first and last fifth
    half = max(1, int(round(0.5 * track.fps)))
    fastest = (float(np.max(cy[half:] - cy[:-half]))
               if len(cy) > half else float(cy[-1] - cy[0]))
    return {**out,
            "drift": float(cy[-k:].mean() - cy[:k].mean()),
            "d_height": float(h[-k:].mean() - h[:k].mean()),
            "fastest": fastest}

def frame_probabilities(windows, n_frames: int, causal: bool = True) -> np.ndarray:
    """Spread window probabilities onto a per-frame track of length `n_frames`.

    Causal (the default): frame t carries the probability of the most recently
    *completed* window, and NaN until the first window has finished. This is
    what a live system could have known at frame t, so it is the only version
    fair to draw on a playing video.

    Non-causal: frame t carries the maximum over every window containing it,
    which is hindsight -- it lights up the run-in to a fall before any window
    covering it had closed. Useful on a static timeline, misleading on playback.
    """
    out = np.full(n_frames, np.nan, dtype=float)
    if not len(windows):
        return out

    if causal:
        for _, w in windows.iterrows():
            end = int(w.end_frame)
            if end < n_frames:
                out[end:] = w.prob
        return out

    for _, w in windows.iterrows():
        a, b = int(w.start_frame), min(int(w.end_frame) + 1, n_frames)
        seg = out[a:b]
        out[a:b] = np.where(np.isnan(seg), w.prob, np.maximum(seg, w.prob))
    return out


def frame_gates(windows, n_frames: int, max_dropout: float = 0.5) -> np.ndarray:
    """Per-frame "the subject was not visible" flag, aligned to the causal track.

    Assigned exactly as `frame_probabilities` assigns scores -- each window's
    state takes effect at the frame it closes and holds until the next window
    closes -- so a frame's gate and its probability always come from the same
    window and cannot disagree.
    """
    out = np.zeros(n_frames, dtype=bool)
    if not len(windows) or "dropout" not in windows.columns:
        return out
    for _, w in windows.iterrows():
        end = int(w.end_frame)
        if end < n_frames:
            out[end:] = bool(w.dropout > max_dropout)
    return out


def alarm_state(frame_probs: np.ndarray, threshold: float = cfg.EVAL.threshold,
                hold_sec: float = 3.0, fps: float = cfg.TRAIN_NOMINAL_FPS,
                gated: np.ndarray | None = None) -> np.ndarray:
    """Boolean alarm per frame: latch on a crossing, hold for `hold_sec`.

    Without the hold, the banner flickers off the instant one window dips back
    under the threshold, which reads as the system changing its mind rather than
    as an alarm. Deployment holds an alarm for an operator to see; so does this.

    `gated` frames neither raise the alarm nor hold one that is already up: with
    nobody in shot there is no evidence either way, and letting a stale alarm
    ride through an absence would put a red banner over an empty room.
    """
    hold = max(1, int(round(hold_sec * fps)))
    fired = np.nan_to_num(frame_probs, nan=0.0) >= threshold
    if gated is not None:
        fired = fired & ~gated
    out = np.zeros(len(fired), dtype=bool)
    remaining = 0
    for i, f in enumerate(fired):
        if f:
            remaining = hold
        if gated is not None and gated[i]:
            remaining = 0
        if remaining > 0:
            out[i] = True
            remaining -= 1
    return out


# ---------------------------------------------------------------------------
# Stage 3: draw it
# ---------------------------------------------------------------------------
_GREEN = (90, 200, 90)
_RED = (60, 60, 235)
_GREY = (150, 150, 150)
_WHITE = (245, 245, 245)
_DARK = (32, 32, 32)


def _timeline_strip(frame_probs: np.ndarray, width: int, height: int,
                    threshold: float, gated: np.ndarray | None = None):
    """A pre-rendered probability-vs-time bar, blitted under every frame.

    Built once rather than per frame: the track does not change during playback,
    only the playhead moves.

    Gated stretches are drawn as a full-height grey band rather than as a low
    score, because a low score is a claim and an absence is not. Rendering them
    as "probably not a fall" would be the same mistake the gate exists to stop.
    """
    import cv2

    strip = np.full((height, width, 3), _DARK, dtype=np.uint8)
    n = len(frame_probs)
    if n == 0:
        return strip

    plot_h = height

    # One column per output pixel, each summarising the frames that map to it.
    edges = np.linspace(0, n, width + 1).astype(int)
    for x in range(width):
        a, b = edges[x], max(edges[x + 1], edges[x] + 1)
        b = min(b, n)
        if gated is not None and len(gated) >= b and gated[a:b].any():
            cv2.line(strip, (x, plot_h - 3), (x, 2), (70, 70, 70), 1)
            continue
        seg = frame_probs[a:b]
        seg = seg[~np.isnan(seg)]
        if not len(seg):
            continue
        p = float(seg.max())
        h = int(round(p * (plot_h - 6)))
        colour = _RED if p >= threshold else _GREEN
        cv2.line(strip, (x, plot_h - 3), (x, plot_h - 3 - h), colour, 1)

    y_thr = plot_h - 3 - int(round(threshold * (plot_h - 6)))
    cv2.line(strip, (0, y_thr), (width, y_thr), _GREY, 1, cv2.LINE_AA)
    return strip


def render(
    track: PoseTrack,
    frame_probs: np.ndarray,
    out_path: str | Path,
    threshold: float = cfg.EVAL.threshold,
    conf_threshold: float = 0.3,
    hold_sec: float = 3.0,
    strip_height: int = 54,
    fourcc: str = "mp4v",
    frame_gated: np.ndarray | None = None,
    frame_alarm: np.ndarray | None = None,
    frame_detail: np.ndarray | None = None,
    progress: bool = True,
) -> Path:
    """Write an annotated copy of the video: skeleton, box, score, alarm, timeline.

    Decodes the source a second time. `frame_probs` must be one entry per frame
    of `track`; NaN renders as "warming up", which is the honest label for the
    opening seconds where no window has closed yet.

    `frame_gated` (from `frame_gates`) marks frames whose window had no subject
    in it. Those draw grey as "NO SUBJECT" and cannot raise or sustain an alarm
    -- three states on screen, matching the three the verdict can return.

    `frame_alarm` supplies the alarm state directly, for a decision rule that is
    not "over threshold, then hold" -- demo-cam v2's latched rolling mean. With
    it, `frame_probs` is read as that rule's score (the rolling mean), and
    `frame_detail`, if given, is shown beside it as the latest window's own
    P(fall), so the panel says both what was decided on and what just arrived.
    """
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(track.path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {track.path}")

    w, h = track.width, track.height
    canvas_h = h + strip_height
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc),
                             track.fps, (w, canvas_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cv2 could not open a writer for {out_path}")

    gated = (np.zeros(track.n_frames, dtype=bool) if frame_gated is None
             else np.asarray(frame_gated, dtype=bool))
    if frame_alarm is None:
        alarm = alarm_state(frame_probs, threshold, hold_sec, track.fps,
                            gated=gated)
    else:
        alarm = np.asarray(frame_alarm, dtype=bool) & ~gated
    strip = _timeline_strip(frame_probs, w, strip_height, threshold, gated=gated)

    # Overlay scale, so the annotation reads the same on a 320x240 LE2I clip and
    # on 1080p phone footage. Clamped: a 4K frame does not want a 6x banner.
    s = float(np.clip(w / 640.0, 0.7, 1.8))
    m = int(round(9 * s))
    lw = max(1, int(round(2 * s)))

    it = range(track.n_frames)
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(it, desc="render", unit="frame")

    for i in it:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = np.full((canvas_h, w, 3), _DARK, dtype=np.uint8)
        canvas[:h] = frame

        firing = bool(alarm[i])
        absent = bool(gated[i])
        colour = _GREY if absent else (_RED if firing else _GREEN)

        box = track.boxes[i]
        if box[2] > box[0] and box[3] > box[1]:
            cv2.rectangle(canvas, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), colour, lw)

        kp = track.keypoints[i]
        ys, xs, cf = kp[:, 0] * h, kp[:, 1] * w, kp[:, 2]
        for a, b in cfg.SKELETON_EDGES:
            if cf[a] >= conf_threshold and cf[b] >= conf_threshold:
                cv2.line(canvas, (int(xs[a]), int(ys[a])), (int(xs[b]), int(ys[b])),
                         colour, lw, cv2.LINE_AA)
        for j in np.flatnonzero(cf >= conf_threshold):
            cv2.circle(canvas, (int(xs[j]), int(ys[j])), max(2, int(3 * s)),
                       (40, 40, 220), -1, cv2.LINE_AA)

        p = frame_probs[i]
        if absent:
            state, label = "NO SUBJECT", "detector sees nobody"
        else:
            state = "FALL DETECTED" if firing else "no fall"
            if np.isnan(p):
                label = "warming up"
            elif frame_detail is not None:
                label = f"mean {p:.2f}  P {frame_detail[i]:.2f}"
            else:
                label = f"P(fall) {p:.2f}"

        # LE2I is 320x240 and a phone clip is 1080p; a panel sized in absolute
        # pixels covers a third of the former and vanishes on the latter.
        panel_w = int((260 if frame_detail is not None else 220) * s)
        cv2.rectangle(canvas, (m, m), (m + panel_w, m + int(58 * s)), _DARK, -1)
        cv2.putText(canvas, state, (m + int(7 * s), m + int(24 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62 * s, colour,
                    max(1, int(round(2 * s))), cv2.LINE_AA)
        cv2.putText(canvas, f"{label}   t={i / track.fps:5.1f}s",
                    (m + int(7 * s), m + int(47 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s, _WHITE,
                    max(1, int(round(1 * s))), cv2.LINE_AA)

        canvas[h:] = strip
        x = int(i / max(track.n_frames - 1, 1) * (w - 1))
        cv2.line(canvas, (x, h), (x, canvas_h), _WHITE, 1)

        writer.write(canvas)

    cap.release()
    writer.release()
    return out_path


# ---------------------------------------------------------------------------
# Playback in a notebook
# ---------------------------------------------------------------------------
def _ffmpeg() -> str | None:
    """An ffmpeg binary, preferring the one imageio ships with the venv."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def to_h264(path: str | Path, out_path: str | Path | None = None,
            crf: int = 23) -> Path:
    """Transcode to H.264/yuv420p so a browser will actually play it.

    OpenCV's `mp4v` writer produces MPEG-4 Part 2, which Chrome and the Jupyter
    HTML5 player refuse -- the usual symptom is a video element that renders
    black with working controls. Returns the input unchanged if no ffmpeg is
    available, so the caller still gets a file, just not an embeddable one.
    """
    path = Path(path)
    exe = _ffmpeg()
    if exe is None:
        return path

    out_path = Path(out_path) if out_path else path.with_name(path.stem + "_h264.mp4")
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(path),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           # Odd dimensions are not representable in yuv420p; the strip makes an
           # odd height easy to hit, so round both down to even.
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError):
        return path
    return out_path


def show_video(path: str | Path, max_embed_mb: float = 60.0, width: int = 720):
    """An HTML5 player for a rendered clip, embedding the bytes when small enough.

    Embedding is what makes the video survive being sent elsewhere or reopened
    from a saved .ipynb; above `max_embed_mb` it would bloat the notebook past
    what most viewers will load, so a path-relative player is used instead --
    which works in JupyterLab but only while the file is on disk.
    """
    from IPython.display import HTML, Video

    path = Path(path)
    size_mb = path.stat().st_size / 1e6
    if size_mb <= max_embed_mb:
        return Video(str(path), embed=True, width=width,
                     html_attributes="controls loop")
    return HTML(
        f'<p><b>{path.name}</b> is {size_mb:.0f} MB - too large to embed; '
        f'playing from disk.</p>'
        f'<video src="{path.as_posix()}" width="{width}" controls loop></video>'
    )


# ---------------------------------------------------------------------------
# One call, end to end
# ---------------------------------------------------------------------------
def analyse(
    video_path: str | Path,
    model,
    model_cfg,
    train_cfg,
    scale: str = cfg.DEFAULT_SCALE,
    device: str = "cuda",
    frame_stride: int | None = None,
    match_fps: bool = False,
    stride: int = cfg.EVAL.sliding_stride,
    threshold: float = cfg.EVAL.threshold,
    warmup: bool = True,
    max_dropout: float | None = None,
    pose_model=None,
    progress: bool = True,
    head=None,
) -> dict:
    """Pose, slide, decide -- everything except drawing, in one call.

    `match_fps` swaps the paper's fixed stride for the one that holds window
    *duration* at four seconds on this file's frame rate. Off by default so the
    result matches every other number in this project; turn it on when the input
    is not ~30 fps and the question is what the detector would do in the field
    rather than how it scores on the benchmark.

    `warmup` scores the opening partial windows so playback has something to
    draw before the first full window closes. The verdict ignores them, so this
    changes what you see and not what is decided.

    `max_dropout` gates the verdict on detection -- see `_ungated`. None (the
    default) keeps the paper's behaviour; **0.95** is the deployment value.
    Not 0.5: falling genuinely degrades detection, so a mid-range threshold
    discards real falls. Notebook 13 measures that at 29 true positives lost
    across the four benchmarks to remove 5 false ones. No benchmark window
    exceeds 0.917, so 0.95 catches an empty frame and nothing else.

    `head` scores every window with a room-fitted `calibrate.ProbeHead` instead
    of the checkpoint's own classifier -- one forward pass either way. Note the
    decision rule is *not* the same one this notebook applies by default: the
    shipped calibrated model uses the rolling mean and latch
    (`fallcore.stream.RollingDecision`, `demo-cam`'s rule), which is what
    `runs/metrics/calibration_head.csv` measured. `verdict` here is still the
    paper's "any window over threshold"; section 3 computes both.
    """
    track = run_pose(video_path, scale=scale, device=device, model=pose_model,
                     progress=progress)
    fs = (suggested_frame_stride(track.fps, train_cfg.seq_len)
          if match_fps and frame_stride is None
          else frame_stride)
    fs = cfg.PREPROCESS.frame_stride if fs is None else fs

    windows = score_track(track, model, model_cfg, train_cfg, frame_stride=fs,
                          stride=stride, warmup=warmup, device=device, head=head)
    v = verdict(windows, threshold=threshold, max_dropout=max_dropout)
    v["frame_stride"] = fs
    v["max_dropout"] = max_dropout
    v["window_seconds"] = train_cfg.seq_len * fs / track.fps
    v["head_backbone"] = (None if head is None
                          else str(getattr(head, "backbone", "") or "head"))
    # The soonest any full window can close, and so the floor on how quickly
    # this configuration could ever alarm on a fall that starts at t=0.
    v["min_latency_sec"] = v["window_seconds"]

    return {"track": track, "windows": windows, "verdict": v}
