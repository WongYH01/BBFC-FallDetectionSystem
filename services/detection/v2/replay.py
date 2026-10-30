"""Replay one video through the v2 decision stack, without camera or Flask.

Feeds every frame through the same pose model and the same Fall Decision the
detector builds (`fallcore.fall_decision`), printing each scored window
(P(fall), rolling mean, latched alarm) and a summary at the end. Use it to
sanity-check a buffer length or a threshold on a recording
before pointing v2 at the live feed -- the same decision runs here, read from
the same environment variables, and only the frame source differs. A flag
overrides one setting on top of that.

    python replay.py path\\to\\clip.mp4
    python replay.py clip.mp4 --buffer 4 --threshold 0.5 --max-frames 900

The pose backbone, checkpoint and head are the detector's and are not flags
(`fallcore.pose_backbone`, `fallcore.fall_decision`): a head only reads the
keypoints of the backbone it was fitted on.

Exit code is 0 whenever the video was readable; the alarm outcome is printed,
not encoded in the status.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

from fallcore import pose_backbone
from fallcore.extract import sightings, subject_row
from fallcore.fall_decision import FallDecision

POSE_TRACKER = os.environ.get("POSE_TRACKER", "bytetrack.yaml")


#: Each decision flag and the environment variable it overrides. A flag is that
#: variable for this run, parsed and checked exactly as the detector parses it.
_FLAG_VARS = {
    "buffer": "BUFFER_LEN",
    "threshold": "FALL_THRESHOLD",
    "clear_below": "FALL_CLEAR_BELOW",
    "clear_windows": "FALL_CLEAR_WINDOWS",
    "min_descent": "FALL_MIN_DESCENT",
    "frame_stride": "STREAM_FRAME_STRIDE",
    "subjects": "FALL_SUBJECTS",
}


def _decision(args) -> FallDecision:
    """The detector's Fall Decision: the environment, then the flags given."""
    env = dict(os.environ)
    env.update({var: str(getattr(args, flag))
                for flag, var in _FLAG_VARS.items()
                if getattr(args, flag) is not None})
    return FallDecision.from_env(env)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    # Every decision flag defaults to "not given": the Fall Decision, with the
    # environment's overrides, is the default, exactly as the live detector
    # reads it. A flag replaces one setting on top of that.
    ap.add_argument("--buffer", default=None, help="BUFFER_LEN")
    ap.add_argument("--threshold", default=None, help="FALL_THRESHOLD")
    ap.add_argument("--clear-below", default=None, help="FALL_CLEAR_BELOW")
    ap.add_argument("--clear-windows", default=None, help="FALL_CLEAR_WINDOWS")
    ap.add_argument("--min-descent", default=None,
                    help="FALL_MIN_DESCENT; 0 turns the descent gate off")
    ap.add_argument("--frame-stride", default=None,
                    help="STREAM_FRAME_STRIDE: rows between window samples; 2 "
                         "matches ~30 fps training data, use 1 when the pose "
                         "rate is ~15 fps")
    ap.add_argument("--subjects", default=None,
                    help="FALL_SUBJECTS: 'tracked' decides every tracked person "
                         "separately, 'largest' the largest box alone")
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

    try:
        decision = _decision(args)
    except ValueError as e:
        print(e)
        return 1
    backbone = pose_backbone.DEPLOYED
    missing = [str(p) for p in (backbone.model, *decision.checkpoints, *decision.heads)
               if p is not None and not p.exists()]
    if missing:
        print("missing model file(s): " + ", ".join(missing) +
              " -- run scripts/fetch_weights.py")
        return 1

    pose = YOLO(str(backbone.model))
    ensemble = decision.build_camera(device="cpu")
    tracked = decision.subjects == "tracked"
    print(f"pose {backbone.model.name} | decision {decision}")
    print(f"file {video.name} @ {fps:.1f} fps")

    # The detector tracks (ids are what tie a person's frames together), so
    # the per-person rule must be replayed through the same tracker.
    if tracked:
        results = pose.track(source=str(video), stream=True, imgsz=backbone.imgsz,
                             device="cpu", tracker=POSE_TRACKER, persist=True,
                             verbose=False)
    else:
        results = pose.predict(source=str(video), stream=True, imgsz=backbone.imgsz,
                               device="cpu", verbose=False)
    events, frame_no = [], 0
    alarm_ever, alarm_first = False, None
    for result in results:
        frame_no += 1
        if args.max_frames and frame_no > args.max_frames:
            break
        if tracked:
            frame_input, ensemble.aspect = sightings(result)
        else:
            frame_input, ensemble.aspect = subject_row(result)
        if not ensemble.observe(frame_input):
            continue
        st = ensemble.state()
        t = frame_no / fps
        events.append({"t_s": round(t, 2), "p_fall": round(st["p_fall"], 4),
                       "rolling": round(st["rolling_mean"], 4),
                       "alarm": st["alarm"]})
        if tracked:
            events[-1]["subject"] = st["subject"]
            events[-1]["alarmed_subjects"] = [s["subject"] for s in st["subjects"]
                                              if s["alarm"]]
        if st["alarm"] and not alarm_ever:
            alarm_ever, alarm_first = True, round(t, 2)
        if args.json:
            print(json.dumps(events[-1]))
        else:
            who = (f"  subject={st['subject']} of {len(st['subjects'])}"
                   if tracked else "")
            print(f"t={t:6.2f}s  P(fall)={st['p_fall']:.3f}  "
                  f"rolling={st['rolling_mean']:.3f}  "
                  f"{'ALARM' if st['alarm'] else 'clear'}{who}")

    ps = [e["p_fall"] for e in events]
    summary = {
        "video": str(video),
        "frames": frame_no,
        "windows": len(events),
        "max_p_fall": round(max(ps), 4) if ps else None,
        "alarm": alarm_ever,
        "first_alarm_s": alarm_first,
        "models": ensemble.classifier.names,
        "buffer": decision.buffer,
        "threshold": decision.threshold,
        "min_descent": decision.min_descent,
        "subjects": decision.subjects,
        "decision": str(decision),
    }
    if not args.json:
        print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
