"""Replay one video through the v2 decision stack, without camera or Flask.

Feeds every frame through the same pose model and `fallcore.stream` ensemble the
detector uses, printing each scored window (P(fall), rolling mean, latched
alarm) and a summary at the end. Use it to sanity-check a checkpoint set, a
buffer length or a threshold on a recording before pointing v2 at the live
feed -- the same decision code runs here, only the frame source differs.

    python replay.py path\\to\\clip.mp4
    python replay.py clip.mp4 --buffer 4 --threshold 0.5 --max-frames 900
    python replay.py clip.mp4 --pose ..\\models\\yolo26n-pose.pt --imgsz 480

Exit code is 0 whenever the video was readable; the alarm outcome is printed,
not encoded in the status.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from fallcore import config as fcfg
from fallcore import extract as fextract
from fallcore.stream import EnsembleStream

REPO_ROOT = Path(__file__).resolve().parents[2]
# Same default as detector.py: the calibrated pair (one backbone + the head
# fitted on this room). Keep them in step -- the offline replay is what makes a
# live number believable.
DEFAULT_CKPTS = [
    str(REPO_ROOT / "runs" / "checkpoints" / n)
    for n in ("augnone_ms_coords_hn_s99.pt",)
]
DEFAULT_HEAD = str(REPO_ROOT / "runs" / "checkpoints"
                   / "probe_augnone_ms_coords_hn_s99.npz")
POSE_SCALE = os.environ.get("POSE_SCALE", "yolo26n")
_DEFAULT_ONNX = [
    Path(__file__).resolve().parents[1] / "models" / f"{POSE_SCALE}-pose.onnx",
    REPO_ROOT / "runs" / "onnx" / f"{POSE_SCALE}-pose-imgsz640.onnx",
]


def _default_pose() -> str:
    return os.environ.get("POSE_MODEL") or str(
        next((p for p in _DEFAULT_ONNX if p.exists()), _DEFAULT_ONNX[0]))


def _keypoints(result) -> np.ndarray:
    """Same subject conversion as detector/fallcore.extract (largest person)."""
    kp = np.zeros((fcfg.N_KEYPOINTS, 3), dtype=np.float32)
    idx = fextract._largest_person(result)
    keypoints = getattr(result, "keypoints", None)
    if idx is not None and keypoints is not None and keypoints.data is not None \
            and len(keypoints.data) > idx:
        k = keypoints.data[idx].cpu().numpy()
        h, w = result.orig_shape
        kp[:, 0] = np.clip(k[:, 1] / max(h, 1), 0.0, 1.0)
        kp[:, 1] = np.clip(k[:, 0] / max(w, 1), 0.0, 1.0)
        kp[:, 2] = k[:, 2]
    return kp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--pose", default=_default_pose())
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--checkpoints", default=os.pathsep.join(DEFAULT_CKPTS),
                    help="os.pathsep-separated model paths")
    ap.add_argument("--head", default=DEFAULT_HEAD,
                    help="logistic head fitted on this room's clips, or '' for "
                         "the checkpoint's own classifier")
    ap.add_argument("--buffer", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--clear-below", type=float, default=0.2)
    ap.add_argument("--clear-windows", type=int, default=2)
    ap.add_argument("--frame-stride", type=int,
                    default=int(os.environ.get(
                        "STREAM_FRAME_STRIDE",
                        str(fcfg.PREPROCESS.frame_stride))),
                    help="rows between window samples; 2 matches ~30 fps "
                         "training data, use 1 when the pose rate is ~15 fps")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="print machine-readable events")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"no such video: {video}")
        return 1

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    cap.release()

    ckpts = [Path(p) for p in args.checkpoints.split(os.pathsep) if p]
    missing = [str(p) for p in ckpts if not p.exists()]
    if missing:
        print("missing checkpoint(s): " + ", ".join(missing))
        return 1
    head = Path(args.head) if args.head else None
    if head is not None and not head.exists():
        print(f"missing head: {head}")
        return 1
    if str(args.pose).endswith(".onnx") and not Path(args.pose).exists():
        print(f"pose model not found: {args.pose}")
        return 1

    pose = YOLO(str(args.pose))
    ensemble = EnsembleStream(ckpts, heads=[head], buffer=args.buffer,
                              threshold=args.threshold,
                              clear_below=args.clear_below,
                              clear_windows=args.clear_windows,
                              frame_stride=args.frame_stride, device="cpu")
    print(f"pose {Path(args.pose).name} | models {ensemble.classifier.names} | "
          f"buffer {args.buffer} | threshold {args.threshold} | "
          f"frame_stride {args.frame_stride}")
    print(f"file {video.name} @ {fps:.1f} fps")

    events, frame_no = [], 0
    alarm_ever, alarm_first = False, None
    for result in pose.predict(source=str(video), stream=True, imgsz=args.imgsz,
                               device="cpu", verbose=False):
        frame_no += 1
        if args.max_frames and frame_no > args.max_frames:
            break
        if not ensemble.observe(_keypoints(result)):
            continue
        st = ensemble.state()
        t = frame_no / fps
        events.append({"t_s": round(t, 2), "p_fall": round(st["p_fall"], 4),
                       "rolling": round(st["rolling_mean"], 4),
                       "alarm": st["alarm"]})
        if st["alarm"] and not alarm_ever:
            alarm_ever, alarm_first = True, round(t, 2)
        if args.json:
            print(json.dumps(events[-1]))
        else:
            print(f"t={t:6.2f}s  P(fall)={st['p_fall']:.3f}  "
                  f"rolling={st['rolling_mean']:.3f}  "
                  f"{'ALARM' if st['alarm'] else 'clear'}")

    ps = [e["p_fall"] for e in events]
    summary = {
        "video": str(video),
        "frames": frame_no,
        "windows": len(events),
        "max_p_fall": round(max(ps), 4) if ps else None,
        "alarm": alarm_ever,
        "first_alarm_s": alarm_first,
        "models": ensemble.classifier.names,
        "buffer": args.buffer,
        "threshold": args.threshold,
    }
    if not args.json:
        print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
