# BBFC Fall Detection System

Replication of Benabdennour et al., *"Real-Time Human Fall Detection From Video
Using YOLOv11 With Pose Estimation"* (IEEE Access, vol. 14, 2026), with the pose
backbone swapped to **YOLO26** and a size sweep the original paper only samples
at two points.

**Two-stage pipeline.** A frozen YOLO26-Pose model extracts 17 COCO keypoints
per frame; a compact Transformer encoder (1,208,897 parameters) classifies the
resulting keypoint *sequence* as Fall / No-Fall. The backbone is never trained —
it is a fixed feature extractor, which is what makes the size sweep a clean
measurement of keypoint quality against capacity.

## Layout

```
fallcore/          the pipeline as an importable package
  config.py        every hyperparameter from the paper, with section references
  manifest.py      video ingest, label inference, the frozen stratified split
  extract.py       frozen YOLO26-Pose -> cached (frames, 17, 3) keypoint arrays
  data.py          windowing, padding, the input-feature variants
  model.py         FallDetectorTransformer + BiLSTM / 1D-CNN / MLP baselines
  train.py         AdamW loop, early stopping on validation F1
  evaluate.py      random-clip and sliding-window protocols, metrics
  bench.py         latency / VRAM under the paper's measurement protocol
  interpret.py     gradient saliency, attention capture
  le2i.py          LE2I annotations and the zone-based label transfer
  caucafall.py     CAUCAFall frame folders and the event-semantics zones
  gmdcsa.py        GMDCSA24 second-based annotations over mixed frame rates
  omnifall.py      the OmniFall taxonomy, splits, parquet pose cache and ingest
  picam.py         the deployment camera's zero-shot set and its model registry
  rulebased.py     the demo-cam aspect-ratio rule, ported for offline scoring
  infer.py         one arbitrary video file -> verdict + annotated render
  viz.py           the shared plots
notebooks/         00 - 18; 00 - 11 run in order, 12 - 18 stand alone
scripts/           long-running fetch/extract jobs that do not belong in a cell
videos/            drop a clip here for notebook 12 to score      (gitignored)
runs/              checkpoints, metrics, figures                  (gitignored)
data/              every dataset and every cache                  (gitignored)
```

### Where the data lives

Everything the notebooks read is under `data/`, including the extracted poses.
Nothing is expected in a home directory or a package cache, so the whole corpus
moves with the folder:

```
data/
  raw/                    the Kaggle compilation (Fall/, No_Fall/)
  keypoints/<scale>/      the pose cache, one .npy per video, every corpus
  boxes/<scale>/          person boxes; only notebook 11's rule needs them
  manifest.csv            the Kaggle corpus, one row per clip
  splits.json             the frozen 80/10/10 split
  omnifall_manifest.csv   the OmniFall corpus, one row per video
  le2i/                   LE2I, subset folders directly beneath
  caucafall/              CAUCAFall, Subject.1 .. Subject.10
  gmdcsa24/               GMDCSA24, Subject 1 .. Subject 4
  picam/                  the deployment camera's clips, one folder per action
  omnifall/
    labels/  splits/      OmniFall annotations (small, tracked)
    pose/<ds>/*.parquet   extracted poses, one file per video
    video/<DS>/           raw video fetched outside the companion package
    cache/videos/         what `omnifall prepare <component>` writes
```

A video's `video_id` is built from its folder and file name rather than its
full path, so the keypoint cache survives the datasets being moved or refiled;
that is also why the mis-filed `picam` clips could be corrected without
re-running pose.

Logic lives in `fallcore` and the notebooks are thin drivers, so the seventeen
of them do not each carry a copy of the training loop.

## Setup

`.venv-train` is already provisioned (torch 2.11 + CUDA 12.8, ultralytics 8.4,
JupyterLab) and its editable install already points at `fallcore/`.

```bash
.venv-train/Scripts/python.exe -m jupyter lab notebooks
```

For a fresh environment instead: `pip install -e .`

### Data

1. **Main dataset** — the Kaggle compilation the paper used,
   [`payutch/fall-video-dataset`](https://www.kaggle.com/datasets/payutch/fall-video-dataset)
   (6,988 clips, 3,140 Fall / 3,848 No-Fall). Extract into `data/raw/`.
2. **LE2I** — only needed for notebook 06 (cross-dataset). Extract into
   `data/le2i/`, so that the subset folders (`Coffee_room_01`, `Home_01`,
   `Office`, ...) sit directly beneath it. The Lecture room and Office
   subsets ship no ground truth and are dropped automatically.
3. **CAUCAFall** — only needed for notebook 08. Extract so that
   `data/caucafall/` holds `Subject.1` through `Subject.10`. The adapter reads
   the per-frame PNGs and ignores the bundled `.avi` copies, whose frame counts
   disagree with the labels.
4. **GMDCSA24** — only needed for notebook 09. Extract into `data/gmdcsa24/` so
   that it contains `Subject 1` through `Subject 4`.
5. **The deployment camera** — only needed for notebook 16. Clips recorded off
   the `demo-cam` RTSP feed go in `data/picam/<action>/`, one folder per action,
   and every folder name must appear in `picam.ACTIONS` — an unknown folder is
   an error rather than a silent drop. Subject identity is recovered from the
   capture timestamp, so refiling a clip changes its label and never its
   subject.
6. **OmniFall** — needed for notebooks 14 and 15. Labels and splits come from
   the Hub automatically; the videos come from each component's original
   authors, via the companion package rather than by hand:

   ```bash
   pip install omnifall
   omnifall status                 # per-component size, source, licence
   omnifall prepare OOPS mcfd edf occu le2i caucafall GMDCSA24 -y
   ```

   The package writes into `data/omnifall/cache/` because `config.OMNIFALL_CACHE`
   points there; set `OMNIFALL_ROOT` to move it off a small disk. Then pose and
   ingest whatever landed:

   ```bash
   python scripts/extract_omnifall_pose.py mcfd
   python -c "from fallcore import omnifall; omnifall.ingest()"
   ```

   Two components need care:

   - **UP-Fall** is 1,118 separate Google Drive archives, and Drive throttles.
     One HTTP 503 aborts the run, and re-running is a no-op because
     `is_dataset_prepared` stops at the first `.mp4` it finds — so an
     interrupted run reports "already prepared" forever. Use
     `python scripts/prepare_upfall.py`, which passes `--force` (safe: the
     per-recording skip still applies) and retries with throughput-aware
     backoff. It also quarantines recordings whose frames carry no timestamp —
     the original release ships at least one — and reports which it skipped.
     It resumes for free, so an interrupted fetch costs no re-download.
   - **CMDFall** cannot be automated: email `thanh-hai.tran@mica.edu.vn`, free
     for research. It carries **42,143 of the 48,596 staged segments**, so
     notebook 14 run without it is training on a small fraction of the source
     domain, and its numbers must be read that way.

   `omnifall.coverage()` prints what is actually on disk versus what the labels
   expect, across all three trees under `data/omnifall/`; `omnifall verify --all`
   checks the package cache against the Hub and is the stricter check of the
   two. Note that LE2I, CAUCAFall and GMDCSA24 are *inside* OmniFall, so a model
   trained there has seen notebooks 06/08/09's test material.

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 00 | `setup_and_data_audit` | Environment check, manifest, dataset statistics, **sanity gate against the paper's clip counts** |
| 01 | `extract_keypoints` | Frozen YOLO26 n/s/m over every video → `.npy` cache; freezes the split |
| 02 | `train_transformer` | The paper's architecture; asserts the 1,208,897-parameter count |
| 03 | `yolo_size_sweep` | **The size experiment** — identical classifier, three backbones, three seeds |
| 04 | `ablations` | The paper's 54-config one-variable-at-a-time sweep |
| 05 | `baselines` | Parameter-matched BiLSTM / 1D-CNN / MLP |
| 06 | `cross_dataset_le2i` | Zero-shot transfer with zone-based label assignment |
| 07 | `mechanistic_analysis` | Gradient saliency and attention patterns |
| 08 | `cross_dataset_caucafall` | Second zero-shot domain; adversarial near-fall ADLs and a frame-rate sweep |
| 09 | `cross_dataset_gmdcsa24` | Third zero-shot domain; `Sleeping` as a specificity test, plus a real 15-vs-30 fps split |
| 10 | `hard_negatives` | **Trains new models**: `full`/`coords` x with/without CAUCAFall hard negatives, scored on all four evaluations |
| 11 | `rule_based_baseline` | The `demo-cam` width/height rule on the same four evaluations — is the transformer worth it? |
| 12 | `video_inference` | **Point it at one video file**: verdict, when it fired, and an annotated render of the decision |
| 13 | `detection_gate` | What refusing to score windows with no visible subject costs, measured on all four labelled evaluations |
| 14 | `omnifall_corpus` | **Trains on OmniFall instead of Kaggle** under its own cross-subject protocol; both models scored on both corpora |
| 15 | `multiclass` | Multi-class supervision over OmniFall's 10 activity classes — **a negative result**, with the window-length explanation tested and ruled out too |
| 16 | `picam_zeroshot` | **The deployment camera**: 60 clips, 12 actions x 5 subjects, every model plus the demo-cam rule scored per clip |
| 17 | `deployment_decisions` | Changes only the decision layer: peak vs clip aggregation vs persistence vs the ensemble, plus the causal rolling-buffer sweep, replayed on picam, Kaggle test, LE2I, CAUCAFall and GMDCSA24 |
| 18 | `pose_onnx` | **Exports the frozen pose backbone to ONNX Runtime and measures it**: keypoint and decision parity on picam, CPU latency at 640/480/320, cameras per CPU |

Run 00 → 01 → 02 first. After that, 03–09 are independent. Notebook 10 needs
01 to have cached keypoints for CAUCAFall and GMDCSA24 (notebooks 08 and 09 do
this), and trains its own models rather than scoring notebook 02's. Notebooks 15
to 17 score checkpoints the earlier ones write — 15 needs 10 and 14, 16 and 17
need 02, 10 and 14 — and report which are missing rather than failing. Notebook
18 exports the pose backbone to ONNX Runtime: it needs 01's picam cache and 02's
checkpoint, and the optional dependencies (`pip install -e ".[onnx]"`, already
present in `.venv-train`).

Notebook 02's one departure from Algorithm 1 is the training budget: it runs up
to 300 epochs with patience 50 where the paper writes 100 with patience 25. The
best checkpoint lands at epoch 40, inside both budgets, so the selected weights
are the same either way; `config.FINAL_TRAIN` documents the choice.

Notebooks are committed without stored outputs, so every number in them comes
from the run in front of you rather than from a previous one.

**Notebook 01 is the long pole**: three full passes over ~7,000 videos, mostly
bound by video decoding. It is idempotent — cached videos are skipped, so an
interrupted run resumes for free. Set `LIMIT_VIDEOS = 200` and confirm the
skeleton-overlay check in section 4 looks right before committing to the full run.

## Three things worth knowing

**The keypoint channel order is `(y, x, conf)`, not `(x, y, conf)`.** The paper
stores y first; Ultralytics emits x first. A transposed cache still trains to a
plausible-looking number, so notebook 01 draws a cached skeleton back onto its
source frame as a visual check.

**The confidence filter is backbone-dependent, and that would confound the
sweep.** A larger pose model rescues clips a smaller one gives up on, so
filtering per scale would hand each arm of the experiment a different dataset.
Notebook 01 takes the *intersection* of videos passing the 0.2 threshold at every
scale and freezes one split for the whole project.

**The paper's random temporal crop is a no-op at its final configuration.**
Algorithm 1 builds a 60-frame sequence and then samples a random window of length
`T`; at `T=60` the window is the entire sequence. We reproduce this faithfully by
default. Raising `PREPROCESS.max_frames` above 120 restores genuine random
cropping — a documented departure, not the paper.

## Reference points from the paper

| Quantity | Paper |
|---|---|
| Classifier parameters | 1,208,897 |
| F1 (random clip) | 97.74% |
| F1 (full video sliding window) | 97.92% |
| F1 on LE2I, zero-shot | 91.65% |
| Pose throughput | 76.2 FPS (13.12 ms/frame, Tesla P100) |
| Peak VRAM, full pipeline | 142.4 MB |
| Most influential hyperparameter | model dimension (5.55 F1 points) |
| Least influential | attention heads (0.35 points) |
| YOLOv11 Small vs Nano | +0.6 F1 points for 3.5× the parameters |

That last row is the one notebook 03 revisits. A 0.6-point gap from single runs
is inside the seed-to-seed spread of this model, so the sweep trains each scale
three times and reports mean ± standard deviation — enough to tell an effect from
a coin flip.

## Demo app

`demo-cam/` is a separate, earlier Flask demo doing live bounding-box-ratio fall
detection on an RTSP stream. It shares only the `yolo26n-pose.pt` checkpoint with
this pipeline.
