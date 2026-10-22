# Scheduled training plan

Status: later · 2026-10-21 · builds on [`infra-plan.md`](infra-plan.md)
steps 1–11; nothing here is needed to run the system

## Goal

Retrain on a schedule from the triaged keypoint files, and write the result
as **artifacts only**: candidate weights plus metrics in a bucket. Nothing is
promoted to the live detector automatically; that still goes through the
pinned Hugging Face revision in `scripts/fetch_weights.py`.

## What exists, what does not

| | State |
|---|---|
| Labelled data path | Exists: notification files keypoints under `keypoints/labelled/<genuine\|false_alarm>/<event>.<npz\|parquet>` on triage |
| Keypoint upload from detection | Not yet; a teammate is building it |
| Head fitter | `scripts/fit_calibration_head.py` — seconds on CPU |
| Transformer training | `fallcore/train.py` exists; its driver script and notebooks are gitignored |
| Training entrypoint (bucket in → candidate out) | Not yet; to be written |
| Image with training deps | Not yet: `fallcore.calibrate` / `fallcore.evaluate` need scikit-learn, which the detection image lacks |

## What the base infra already guarantees

Nothing in `infra-plan.md` has to change for this:
- GKE **Standard** on the `REGULAR` channel (1.35): every option below
  works, including the built-in `autopilot-spot` compute class
  (needs ≥1.34.1).
- **e2-standard-4** leaves ~1.5–2 vCPU spare after detection and notification.
- **Workload Identity** is on, so the job gets a KSA and `principal://` grants.
- `keypoints/labelled/` has **no lifecycle rule**; the training set is never
  auto-deleted.
- New resources fit the existing layout: `training.tf` in each Tofu root,
  `infra/k8s/training/`.

## Steps

### T1 — Data (`infra/tofu/base/training.tf`)
- A models bucket, `<project>-bbfc-models`: uniform access, public access
  prevention enforced, no `force_destroy`. A separate bucket keeps the IAM
  simple (no prefix conditions on the clip bucket).
- Upload the original picam set's keypoints there once, under `base-set/`.
  Labelled alerts are only real falls and false alarms; a missed fall never
  becomes an alert. Training on alerts alone is biased.

### T2 — Image
- A training Dockerfile: the detection image's base plus scikit-learn and
  the entrypoint.
- Its CI job reuses `.github/workflows/image.yml`, which then needs a
  Dockerfile-path input.

### T3 — Cluster (`infra/tofu/cluster/training.tf`)
- `addons_config { gcs_fuse_csi_driver_config { enabled = true } }` on the
  cluster (an in-place update). The job then reads and writes plain files;
  the entrypoint needs no GCS client.
- KSA principal `bbfc/training`: `roles/storage.objectViewer` on the clip
  bucket, `roles/storage.objectUser` on the models bucket.

### T4 — Job (`infra/k8s/training/`)
- `serviceaccount.yaml` — KSA `training`.
- `priorityclass.yaml` — `training-low`, value below 0,
  `preemptionPolicy: Never`. The scheduler evicts training before it ever
  squeezes detection.
- `cronjob.yaml`:
  - Weekly schedule, `concurrencyPolicy: Forbid`, `activeDeadlineSeconds`.
  - `priorityClassName: training-low`.
  - Requests ~1 vCPU / 2 Gi, limits ~1.5 vCPU / 4 Gi.
  - GCS FUSE volumes (pod annotation `gke-gcsfuse/volumes: "true"`):
    the clip bucket read-only with `only-dir=keypoints/labelled`, the models
    bucket read-write.
  - Writes `candidates/<run-timestamp>/` with the weights and `metrics.json`.

**If a transformer run outgrows the node:** add
`nodeSelector: { cloud.google.com/compute-class: autopilot-spot }` to the pod.
It then runs off the node, billed per second (~$0.05/h at 2 vCPU / 8 Gi).

**Done when:** a manual `kubectl create job --from=cronjob/training` writes a
candidate and its metrics to the models bucket.

## Alternatives

- **Spot node pool scaling to 0** (e.g. e2-standard-4, ~$0.10/h while up):
  more Tofu (node pool, taints, tolerations); worth it only for GPUs.
- **Cloud Run Job + Cloud Scheduler** (~$0.23/h at 2 vCPU / 8 Gi): runs even
  while the cluster is destroyed. L4 GPUs exist in `asia-southeast1`, but a
  free-trial account cannot use GPUs.
- The CronJob does not run while the cluster is destroyed (infra step 12),
  which is fine for a demo.

## Open questions

- Head refit only, or the transformer too.
- Schedule, and whether each run uses every labelled file or only new ones.
- Where the picam set lives today (`data/` is gitignored and not on the
  machine this plan was written from).

## Unverified

- The `gcs_fuse_csi_driver_config` field name in google provider 8.x.
- Transformer training time on CPU.
