# BBFC Fall Detection System

Real-time fall detection from one RTSP camera. A frozen **YOLO26-Pose** backbone
extracts 17 COCO keypoints per frame; a compact Transformer encoder (1,208,897
parameters) classifies the resulting keypoint *sequence* — a 4 s window — as
Fall / No-Fall. The backbone is never trained, so it is a fixed feature
extractor and the classifier is the only learned part of the pipeline.

Replication of Benabdennour et al., *"Real-Time Human Fall Detection From Video
Using YOLOv11 With Pose Estimation"* (IEEE Access, vol. 14, 2026), with the pose
backbone swapped to **YOLO26** and the decision layer rebuilt for deployment.

## The live system

**[`demo-cam/v2`](demo-cam/v2/README.md)** is what runs on the camera: ONNX pose
on CPU, one window scored per second, a logistic head fitted on the room, a
rolling mean of four windows with a hysteresis latch, and a descent gate.

Measured on the deployment camera's 60 clips (12 actions × 5 subjects),
leave-one-subject-out — fit on four subjects, score the fifth:

| | clip F1 | false alarms | missed falls |
|---|---|---|---|
| the fitted camera position, held out | **0.967** | 1 | 1 |
| the same room re-aimed, zero-shot (52 clips) | **0.909** | 2 | 3 |
| the same, with no fitted head at all | 0.724 | 9 | 7 |

The second row is the honest transfer number: the camera moved to another wall
and a lower angle, and a head nobody re-fitted. The third says what the head is
worth there — it is fitted on one view, but most of its benefit survives the
move. Window AUC is 0.992 on the fitted view and 0.879 on the other, against
0.887 / 0.814 with no head. Transfer to a *different* room is untested: every
clip in this project comes from one room.

## Layout

```
fallcore/        the pipeline as an importable package
  config.py        every hyperparameter, with references to the paper
  extract.py       frozen YOLO26-Pose -> cached (frames, 17, 3) keypoint arrays
  onnxpose.py      the ONNX export of that backbone and its parity checks
  data.py          windowing, padding, the input-feature variants
  model.py         the Transformer classifier (+ BiLSTM / CNN / MLP baselines)
  train.py         AdamW loop, early stopping on validation F1
  evaluate.py      clip and sliding-window protocols, metrics
  calibrate.py     the per-room logistic head and the window kinematics
  stream.py        the live decision: ring buffer, rolling mean, latch, gate
  infer.py         one video file -> verdict + annotated render
  manifest.py      video ingest, label inference, the frozen stratified split
  le2i.py caucafall.py gmdcsa.py omnifall.py picam.py   dataset adapters
demo-cam/
  v2/              the deployed Flask app (see its README)
  v1/              the earlier bounding-box-ratio demo, kept for comparison
  pi/              MediaMTX config + the encoder watchdog for the Pi
scripts/
  fetch_weights.py        pull the deployment weights from Hugging Face
  fit_calibration_head.py fit the room head on that room's own clips
runs/ data/ videos/       checkpoints, corpora and caches   (all gitignored)
```

## Setup

```bash
pip install -e .
python scripts/fetch_weights.py     # ONNX pose + the calibrated pair
```

`fetch_weights.py` pulls three files (~17 MB) from the private Hugging Face repo
`junyuu/fall-detection-bbfc`, pinned to a revision and each verified against its
SHA-256. It reads the token from `demo-cam/.env` only — `FALL_WEIGHTS_TOKEN=hf_…`
with read scope, no other auth path. `--check` verifies local files without one.

## Three things worth knowing

**The keypoint channel order is `(y, x, conf)`, not `(x, y, conf)`.** The paper
stores y first, Ultralytics emits x first, and a transposed cache still trains to
a plausible-looking number.

**The pose scale is part of the model.** The head was fitted on `yolo26n`
keypoints at 640; another scale hands it an input distribution it never saw.

**The head is a commissioning artefact, not a weight.** It is fitted on one
installation's own falls and ADLs (`scripts/fit_calibration_head.py`), and it is
what takes clip F1 from 0.818 to 0.967. Running the checkpoint without one is
supported and scores lower. Moving the camera costs accuracy but not the head:
refit when you can, keep running when you cannot.

These are research artifacts, not a certified medical device.
