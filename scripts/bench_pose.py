"""Time pose model files and recommend the stream strides for each.

Run this on the machine that will run the detection container, before switching
pose models (say yolo26n -> yolo26s or yolo26m). It needs no camera, no Pi and
no running service:

    python scripts/bench_pose.py                      # every .pt/.onnx in demo-cam/models
    python scripts/bench_pose.py demo-cam/models/yolo26s-pose.pt --camera-fps 15
    python scripts/bench_pose.py --imgsz 480 --threads 4
    python scripts/bench_pose.py --video videos/some_clip.mp4

For each file it prints ms per frame (mean and p95, warm-up discarded, IQR
outliers dropped), frames per second, and the `VID_STRIDE` /
`STREAM_FRAME_STRIDE` pair that keeps keypoint rows ~67 ms apart for the given
camera, which is the spacing the fall classifier was trained on. Pick the
strides from here, not by feel: the wrong pair changes the time span of a
60-row window, not just its cost (see demo-cam/v2/README.md, "Row spacing").

Timing uses the settings the live detector reads -- `POSE_IMGSZ` (640),
`POSE_THREADS` (2, ONNX Runtime's pool) and `POSE_DEVICE` (cpu for ONNX, CUDA
when available for .pt) from the environment and demo-cam/.env -- so the
numbers carry over. Each row is labelled with those settings.

Verdicts compare pose fps with the rate the recommended strides demand:

    fits       >= 1.3x   the live loop also tracks, draws, JPEG-encodes and
                         renders the skeleton canvas; 30% is a starting margin,
                         revise it once `/stream/status` fps is compared with a
                         row from here
    marginal   1.0-1.3x
    too slow   < 1.0x    the pose pass cannot keep up; rows drift apart
    failed               the file did not load; the run carries on

This replaces demo-cam v2's old metrics run, which could never read faster than
the camera and silenced fall alerts while it ran. After a switch, check the
live rate with the always-on `fps` field of `/stream/status`.

It deliberately does not import demo-cam's `detector` or `app`: both load the
pose model and read the environment at import time.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "demo-cam" / "models"

TARGET_ROWS_PER_SEC = 15.0                    # training rows: 2 frames apart at 30 fps
TARGET_SPACING_MS = 1000.0 / TARGET_ROWS_PER_SEC
SPACING_TOLERANCE = 0.10                      # warn beyond +-10% of 67 ms
WINDOW_ROWS = 60                              # fallcore.config seq_len: 60 rows = 4 s
FIT_MARGIN = 1.3                              # headroom a "fits" verdict needs


def backend_of(path) -> str:
    return "onnx" if str(path).lower().endswith(".onnx") else "torch"


def recommend_strides(camera_fps: float) -> tuple[int, int]:
    """`(VID_STRIDE, STREAM_FRAME_STRIDE)` for rows closest to 67 ms apart.

    All of the stride goes into `VID_STRIDE`: the README measured posing every
    2nd frame as bit-identical to full rate, and it halves the pose cost.
    """
    return max(1, round(camera_fps / TARGET_ROWS_PER_SEC)), 1


def measure_and_recommend(models, camera_fps: float = 30.0,
                          imgsz: int = 640, threads: int = 2,
                          time_model=None) -> list[dict]:
    """Time each model file and say how to run it on this camera.

    `time_model(path, imgsz, threads)` returns `{"mean_ms", "p95_ms",
    "device"}` and may raise if the file does not load; the default is the
    real one (`make_timer()`). One row per model comes back, in order, whatever happens to
    the others.
    """
    time_model = time_model or make_timer()
    vid_stride, frame_stride = recommend_strides(camera_fps)
    spacing_ms = 1000.0 * vid_stride * frame_stride / camera_fps
    required_fps = camera_fps / vid_stride     # pose runs on every VID_STRIDE-th frame

    warning = None
    if abs(spacing_ms - TARGET_SPACING_MS) > SPACING_TOLERANCE * TARGET_SPACING_MS:
        warning = (f"{camera_fps:g} fps camera: no whole-number stride puts rows "
                   f"near {TARGET_SPACING_MS:.0f} ms; the best is "
                   f"{spacing_ms:.0f} ms, so a {WINDOW_ROWS}-row window spans "
                   f"{WINDOW_ROWS * spacing_ms / 1000:.1f} s instead of "
                   f"{WINDOW_ROWS * TARGET_SPACING_MS / 1000:.0f} s")

    rows = []
    for path in models:
        row = {
            "model": Path(path).name,
            "backend": backend_of(path),
            "imgsz": imgsz,
            "threads": threads,
            "device": None,
            "mean_ms": None,
            "p95_ms": None,
            "fps": None,
            "headroom": None,
            "vid_stride": vid_stride,
            "stream_frame_stride": frame_stride,
            "spacing_ms": spacing_ms,
            "required_fps": required_fps,
            "verdict": "failed",
            "warning": warning,
            "error": None,
        }
        try:
            timing = time_model(path, imgsz, threads)
        except Exception as exc:  # noqa: BLE001 - one bad file must not hide the rest
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue

        fps = 1000.0 / timing["mean_ms"]
        headroom = fps / required_fps
        row.update(
            device=timing.get("device"),
            mean_ms=timing["mean_ms"],
            p95_ms=timing["p95_ms"],
            fps=fps,
            headroom=headroom,
            verdict=("fits" if headroom >= FIT_MARGIN
                     else "marginal" if headroom >= 1.0 else "too slow"),
        )
        rows.append(row)
    return rows


# -- the real timer ------------------------------------------------------------

def make_timer(frame=None, warmup: int = 10, runs: int = 60):
    """A `time_model` that times a file the way the live detector runs it.

    ONNX models load after `use_tuned_sessions(threads)`, exactly as
    detector.py does, and run on `POSE_DEVICE` or cpu; `.pt` models run on
    `POSE_DEVICE` or CUDA when available. Timing is the full Ultralytics predict
    call (`fallcore.onnxpose.predict_timings`), preprocessing and decoding
    included. `frame` defaults to a blank `imgsz` square: YOLO26 is dense, so
    its cost does not depend on content.
    """
    def time_model(path, imgsz, threads):
        import numpy as np

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from ultralytics import YOLO

        from fallcore import onnxpose

        pose_device = os.environ.get("POSE_DEVICE", "")
        if backend_of(path) == "onnx":
            if threads > 0:
                onnxpose.use_tuned_sessions(threads=threads, spinning=False)
            device = pose_device or "cpu"
        else:
            import torch
            device = pose_device or ("cuda" if torch.cuda.is_available() else "cpu")

        model = YOLO(str(path))
        img = frame if frame is not None else np.zeros((imgsz, imgsz, 3), np.uint8)
        t = onnxpose.predict_timings(model, img, imgsz=imgsz, warmup=warmup,
                                     runs=runs, device=device)
        return {"mean_ms": t["mean_ms"], "p95_ms": t["p95_ms"], "device": device}

    return time_model


def first_frame(video: Path):
    import cv2

    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read a frame from {video}")
    return frame


# -- command line --------------------------------------------------------------

def format_table(rows: list[dict], camera_fps: float) -> str:
    def num(v, fmt):
        return "-" if v is None else format(v, fmt)

    head = ("model", "backend", "imgsz", "thr", "device", "mean ms", "p95 ms",
            "fps", "headroom", "VID_STRIDE", "STREAM_FRAME_STRIDE", "row ms",
            "verdict")
    body = [(r["model"], r["backend"], str(r["imgsz"]), str(r["threads"]),
             r["device"] or "-", num(r["mean_ms"], ".1f"), num(r["p95_ms"], ".1f"),
             num(r["fps"], ".1f"),
             num(r["headroom"], ".2f") + ("x" if r["headroom"] is not None else ""),
             str(r["vid_stride"]), str(r["stream_frame_stride"]),
             f"{r['spacing_ms']:.0f}", r["verdict"]) for r in rows]
    widths = [max(len(c) for c in col) for col in zip(head, *body)]

    def line(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    out = [f"camera {camera_fps:g} fps -> pose must sustain "
           f"{rows[0]['required_fps']:.1f} fps" if rows else "", line(head),
           line(["-" * w for w in widths])]
    out += [line(b) for b in body]
    notes = sorted({r["warning"] for r in rows if r["warning"]})
    out += [f"\nWARNING: {n}" for n in notes]
    out += [f"\n{r['model']} failed: {r['error']}" for r in rows if r["error"]]
    return "\n".join(out)


def _load_env() -> None:
    """Read the same .env files as the detection service, without overriding."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / "services" / "detection" / ".env")
    load_dotenv(REPO_ROOT / "demo-cam" / ".env")


def main() -> int:
    _load_env()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", type=Path,
                    help=f"pose model files (default: every .pt/.onnx in {MODELS_DIR})")
    ap.add_argument("--camera-fps", type=float, default=30.0)
    ap.add_argument("--imgsz", type=int,
                    default=int(os.environ.get("POSE_IMGSZ", "640")))
    ap.add_argument("--threads", type=int,
                    default=int(os.environ.get("POSE_THREADS", "2")),
                    help="ONNX Runtime intra-op threads; 0 = ORT default")
    ap.add_argument("--video", type=Path,
                    help="time the first frame of this video instead of a blank one")
    ap.add_argument("--runs", type=int, default=60)
    args = ap.parse_args()

    models = args.models or sorted(p for p in MODELS_DIR.iterdir()
                                   if p.suffix.lower() in (".pt", ".onnx"))
    if not models:
        raise SystemExit(f"no .pt or .onnx files in {MODELS_DIR}")
    frame = first_frame(args.video) if args.video else None

    rows = measure_and_recommend(models, camera_fps=args.camera_fps,
                                 imgsz=args.imgsz, threads=args.threads,
                                 time_model=make_timer(frame, runs=args.runs))
    print(format_table(rows, args.camera_fps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
