"""Export the frozen pose backbone to ONNX and measure it.

The classifier is not the problem on a CPU-only deployment: it is ~1 ms per
window against a pose backbone measured at ~22 ms per frame (notebook 03's
protocol on this machine). The backbone is the budget, so this module owns the
one lever that changes it without touching the weights -- ONNX Runtime -- and
the evidence needed to trust that swap:

* `export_pose` writes a static-shape ONNX file under `runs/onnx/`, idempotently;
* `frame_keypoints` runs one frame through either backend and returns the exact
  `(17, 3)` `(y, x, conf)` row `extract.extract_video` writes to the cache, so
  the two can be diffed frame by frame;
* `predict_timings` times the full Ultralytics pipeline (pre + model + post),
  `session_timings` times the raw ONNX Runtime forward alone;
* `window_probabilities` scores an in-memory keypoint array with a classifier
  checkpoint, which is how the notebook shows ONNX keypoints give the same
  decisions as the cached PyTorch ones without writing into the shared cache.

Nothing here trains or changes the model. Ultralytics runs the `.onnx` file
through ONNX Runtime because it is installed; the same file also runs under the
raw session, which is what a non-Ultralytics deployment would use.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from . import config as cfg
from .bench import _iqr_filter
from .extract import _largest_person, resolve_weights

#: Where exported graphs live. `runs/` is gitignored, so exports are a local
#: artifact like the checkpoints, not something to commit.
ONNX_DIR = cfg.RUNS_DIR / "onnx"


def onnx_path(scale: str = "yolo26n", imgsz: int = 640) -> Path:
    """Stable filename for one scale/size export."""
    return ONNX_DIR / f"{scale}-pose-imgsz{imgsz}.onnx"


def export_pose(scale: str = "yolo26n", imgsz: int = 640,
                force: bool = False) -> Path:
    """Export the frozen pose weights to a static-shape ONNX file.

    Idempotent: an existing file is returned unless `force`. Ultralytics writes
    the graph next to the source weights, so the result is moved into
    `runs/onnx/` where every export lives in one place and the demo-cam weights
    directory stays clean. Exports on CPU with a fixed batch of 1, which is the
    streaming case; a dynamic batch can be added later if cameras are batched.
    """
    target = onnx_path(scale, imgsz)
    if target.exists() and not force:
        return target

    from ultralytics import YOLO

    model = YOLO(resolve_weights(scale))
    produced = Path(model.export(format="onnx", imgsz=imgsz, simplify=False,
                                 device="cpu"))
    target.parent.mkdir(parents=True, exist_ok=True)
    if produced.resolve() != target.resolve():
        shutil.move(str(produced), str(target))
    return target


def load_onnx_model(path: str | Path):
    """Ultralytics wrapper around an exported file, backed by ONNX Runtime."""
    from ultralytics import YOLO

    return YOLO(str(path))


def load_pose(scale: str = "yolo26n", imgsz: int = 640):
    """The exported ONNX pose model, ready to hand to `extract_video`/`run_pose`.

    Same interface as `extract.load_pose_model`, so it drops into
    `infer.analyse(..., pose_model=...)` unchanged; the only difference is the
    backend. Note the input size is baked into the graph -- the `imgsz` here
    must be the one the file was exported at, which is why the size is in the
    filename.

    Measured on a Ryzen 7 9800X3D, CPU, single frame (`runs/metrics/
    pose_onnx_cheap.csv`): 640 fp32 ~10 ms raw / ~12 ms through the Ultralytics
    pipeline, 480 ~6/7 ms, 320 ~3/4 ms; dynamically quantized INT8 builds are
    *slower* than fp32 here (24 ms at 640) and move keypoints, so do not use
    them as a speedup. Threads scale the raw forward roughly linearly below the
    core count, and `session.intra_op.allow_spinning=0` removes the busy-wait
    for ~nothing in latency when the box has other work to do.
    """
    path = onnx_path(scale, imgsz)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist -- export it first: "
            f"onnxpose.export_pose('{scale}', {imgsz}) (notebook 18 exports "
            f"640/480/320)")
    return load_onnx_model(path)


def tuned_session_options(threads: int | None = None,
                          spinning: bool = False):
    """ONNX Runtime session options with the thread count made explicit.

    Measured at 640 fp32 on a 16-core Ryzen 7 9800X3D, raw graph, per frame:

        threads      wall ms   cpu ms   cores busy
        default(8)      8.4     69.7      8.3
        4              10.3      ~         ~
        2              16.4     32.6      2.0
        1              31.5      ~         ~

    CPU time per frame is roughly constant and even *falls* with fewer threads
    (the busy-wait between ops is what costs: 70 ms at default, 33 ms at two
    threads), while wall time is that work divided across the pool. So the knob
    is a budget, not a speedup: pick the smallest pool that still meets your
    per-frame deadline and leave the rest of the box alone. `spinning=False`
    stops the busy-wait; it costs ~4 ms of latency at 640 and cuts CPU burn
    (12.7 ms wall / 59 ms CPU against 8.4 / 70).
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1          # one frame at a time is the use case
    if not spinning:
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return so


def use_tuned_sessions(threads: int | None = None, spinning: bool = False) -> None:
    """Make Ultralytics build ONNX sessions with `tuned_session_options`.

    Ultralytics calls `onnxruntime.InferenceSession(weight, providers=...)` with
    no session options (verified in 8.4.104, `nn/backends/onnx.py`), so an ONNX
    pose model in the demo grabs every core for every frame. Wrapping the
    constructor is the only seam it offers; call this before loading the model.
    Idempotent -- wrapping twice does not stack.
    """
    import onnxruntime as ort

    if getattr(ort.InferenceSession, "_bbfc_tuned", False):
        return
    original = ort.InferenceSession

    def patched(path, sess_options=None, **kwargs):
        so = sess_options or tuned_session_options(threads, spinning)
        return original(path, sess_options=so, **kwargs)

    patched._bbfc_tuned = True
    ort.InferenceSession = patched


def session_summary(path: str | Path) -> dict:
    """Providers, I/O signature and file size of an ONNX graph, for the record."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=ort.get_available_providers())
    return {
        "path": str(path),
        "file_mb": round(Path(path).stat().st_size / 1e6, 1),
        "providers": sess.get_providers(),
        "inputs": [(i.name, tuple(i.shape)) for i in sess.get_inputs()],
        "outputs": [(o.name, tuple(o.shape)) for o in sess.get_outputs()],
    }


def frame_keypoints(model, frame: np.ndarray, imgsz: int = 640) -> np.ndarray:
    """One frame -> the cache's `(17, 3)` `(y, x, conf)` row for the largest person.

    Byte-for-byte the same convention as `extract.extract_video`: most pixels
    -> largest box -> keypoints in pixels -> swapped and divided by frame height
    and width. Frames with no detection come back as zeros, as in the cache.
    """
    r = model.predict(frame, imgsz=imgsz, device="cpu", verbose=False)[0]
    kp = np.zeros((cfg.N_KEYPOINTS, 3), dtype=np.float32)
    idx = _largest_person(r)
    keypoints = getattr(r, "keypoints", None)
    if idx is not None and keypoints is not None and keypoints.data is not None \
            and len(keypoints.data) > idx:
        k = keypoints.data[idx].cpu().numpy()
        h, w = r.orig_shape
        kp[:, 0] = np.clip(k[:, 1] / max(h, 1), 0.0, 1.0)
        kp[:, 1] = np.clip(k[:, 0] / max(w, 1), 0.0, 1.0)
        kp[:, 2] = k[:, 2]
    return kp


def predict_timings(model, frame: np.ndarray, imgsz: int = 640,
                    warmup: int = 5, runs: int = 30) -> dict:
    """Per-frame latency of the whole Ultralytics predict call on one frame.

    `bench.benchmark_pose` answers the same question for the PyTorch model; this
    is its backend-agnostic twin, kept here so an ONNX file can be timed through
    the identical preprocessing and postprocessing path.
    """
    for _ in range(warmup):
        model.predict(frame, imgsz=imgsz, device="cpu", verbose=False)

    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(frame, imgsz=imgsz, device="cpu", verbose=False)
        timings.append((time.perf_counter() - t0) * 1000.0)

    clean = _iqr_filter(np.asarray(timings))
    return {
        "device": "cpu",
        "imgsz": imgsz,
        "mean_ms": float(clean.mean()),
        "std_ms": float(clean.std()),
        "median_ms": float(np.median(clean)),
        "p95_ms": float(np.percentile(clean, 95)),
        "fps": float(1000.0 / clean.mean()) if clean.mean() > 0 else None,
        "n_after_iqr": int(clean.size),
    }


def session_timings(path: str | Path, imgsz: int = 640,
                    providers: list[str] | None = None,
                    warmup: int = 10, runs: int = 60) -> dict:
    """Raw ONNX Runtime forward latency on a fixed tensor.

    No preprocessing, no decoding: this is the streaming cost of the graph
    itself, and the number to scale when asking "how many cameras". A zero
    tensor is enough for latency -- the graph is dense and its cost does not
    depend on content.
    """
    import onnxruntime as ort

    providers = providers or ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(path), providers=providers)
    shape = sess.get_inputs()[0].shape
    shape = [int(d) if isinstance(d, int) and d > 0 else s
             for d, s in zip(shape, (1, 3, imgsz, imgsz))]
    x = np.zeros(shape, dtype=np.float32)

    for _ in range(warmup):
        sess.run(None, {sess.get_inputs()[0].name: x})
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {sess.get_inputs()[0].name: x})
        timings.append((time.perf_counter() - t0) * 1000.0)

    clean = _iqr_filter(np.asarray(timings))
    return {
        "providers": sess.get_providers(),
        "imgsz": imgsz,
        "input_shape": tuple(shape),
        "mean_ms": float(clean.mean()),
        "std_ms": float(clean.std()),
        "median_ms": float(np.median(clean)),
        "p95_ms": float(np.percentile(clean, 95)),
        "fps": float(1000.0 / clean.mean()) if clean.mean() > 0 else None,
        "n_after_iqr": int(clean.size),
    }


def window_probabilities(model, model_cfg, train_cfg, arr: np.ndarray,
                         device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """`(starts, P(fall))` for every sliding window of an in-memory keypoint array.

    Exactly `evaluate.score_clips`' arithmetic, but over an array rather than
    the shared cache -- the parity check in notebook 18 extracts with ONNX and
    must not overwrite the PyTorch keypoints everything else is scored against.
    """
    import torch

    from .data import build_features, prepare_sequence, sliding_starts, take_window

    seq = prepare_sequence(arr, None, cfg.PREPROCESS.frame_stride)
    starts = sliding_starts(len(seq), train_cfg.seq_len, cfg.EVAL.sliding_stride)
    windows = np.stack([
        build_features(take_window(seq, train_cfg.seq_len, start=int(s)),
                       model_cfg.features)
        for s in starts
    ])
    model.to(device).eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            model(torch.from_numpy(windows).float().to(device))).cpu().numpy()
    return np.asarray(starts), probs
