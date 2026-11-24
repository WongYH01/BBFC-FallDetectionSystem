# Environment Variables

Every variable in the deployment `.env` and config examples (not the `demos/` ones), with the file it lives in and what to set. The setup pages tell you when to fill each file in. This page is the lookup.

By the end of this page you know:

- Which file holds each variable on each path.
- What to put in it, and which values to keep.

Needs: nothing. Use it beside [Local deployment](03-local/README.md) or [Kubernetes and Tailscale](05-cloud/02-k8s-and-tailscale.md).

## Where each variable lives

| File | Path | Copied from | Read by |
|------|------|-------------|---------|
| `.env` | Local | `.env.example.local` | Compose, passed on to `notification` |
| `services/detection/.env` | Local | `services/detection/.env.example` | `detection` |
| `infra/k8s/notification/config.env` | Cloud | `infra/k8s/notification/config.env.example` | ConfigMap `notification` |
| `infra/k8s/detection/config.env` | Cloud | `infra/k8s/detection/config.env.example` | ConfigMap `detection` |
| `.env.cloud` | Compose in cloud mode. The guide's cloud path uses GKE instead, so you only need it to run the cloud profile by hand. | `.env.example.cloud` | Compose |

Never commit a `.env` file. On the cloud path the Telegram bot token and the stream password are not in a file: they come from Secret Manager (see [Provision](05-cloud/01-provision.md#5-store-the-secret-values)).

`{INSERT...}` in an example file marks a value you must replace. Angle brackets in the Kubernetes examples, such as `<pi-name>.<tailnet>.ts.net`, mark your own values.

## Notification

Local: `.env` in the repo root. Cloud: `infra/k8s/notification/config.env`.

| Name | Local | Cloud | Value |
|------|-------|-------|-------|
| `COMPOSE_PROFILES` | `local` | not used | Keep. Makes compose start all four services. |
| `POSTGRES_PASSWORD` | Required | `unused` | Local: any password you choose. Cloud logs in with IAM: keep `unused`. |
| `DB_PORT` | Required | not used | Free port on your computer for Postgres. |
| `DB_USERNAME` | not set | `notification@<GCP_PROJECT_ID>.iam` | Cloud: the IAM database user. |
| `SPRING_DATASOURCE_URL` | not set | JDBC URL | Cloud: build it as in [Kubernetes and Tailscale](05-cloud/02-k8s-and-tailscale.md#notification). |
| `TELEGRAM_BOT_TOKEN` | Required | Secret Manager | The [bot token](01-prerequisites/README.md#bot-token). |
| `TELEGRAM_CHAT_GROUP_ID` | Required | Required | The [chat group ID](01-prerequisites/README.md#chat-group-id). |
| `SUPABASE_JWT_SECRET` | Required | not used | [Generate it](01-prerequisites/README.md#supabase). |
| `SUPABASE_ANON_KEY` | Required | not used | [Generate it](01-prerequisites/README.md#supabase). |
| `SUPABASE_SERVICE_KEY` | Required | `unused` | Local: [generate it](01-prerequisites/README.md#supabase). |
| `STORAGE_PORT` | Required | not used | Free port on your computer for Supabase storage. |
| `STORAGE_FILE_SIZE_LIMIT` | `8388608` | not used | Keep. 8 MB. |
| `CLIPS_STORE` | not set | `gcs` | Keep. |
| `CLIPS_GCS_BUCKET` | not set | Required | The `clip_bucket_name` from `terraform.tfvars`, for example `<GCP_PROJECT_ID>-clips`. |
| `CLIPS_ENSURE_BUCKET` | not set | `false` | Keep. |
| `GCLOUD_ADC_PATH` | not used | `.env.cloud` only | Path to your Google application default credentials file. Not used on GKE. |
| `JAVA_OPTS` | not set | `-XX:MaxRAMPercentage=75` | Keep. |

"not set" means the variable is absent from the example file. Compose or the service supplies a default, or the variable is not used on that path.

## Detection

Local: `services/detection/.env`. Cloud: `infra/k8s/detection/config.env`.

| Name | Local | Cloud | Value |
|------|-------|-------|-------|
| `ROOM_ID` | Required | Required | Label for the room the camera watches. On the cloud path it names the room's clip folder in the bucket. |
| `ROOM_NAME` | Required | Required | Room name shown in the alert. |
| `NOTIFICATION_BASE_URL` | `http://notification:8081/api/v1/notification` | same | Keep. The compose or cluster hostname of `notification`. |
| `MEDIAMTX_HOST` | Required | `pi-mediamtx` | Local: the Pi's LAN address. Cloud: keep `pi-mediamtx`, the in-cluster name of the Tailscale egress service. |
| `MEDIAMTX_BROWSER_HOST` | not set | Required | Cloud: the Pi's tailnet name, `<pi-name>.<tailnet>.ts.net`. |
| `MEDIAMTX_RTSP_PORT` | not set | `8554` | Keep. |
| `STREAM_USERNAME` | Required | Required | The user you set on the Pi's MediaMTX. |
| `STREAM_PASSWORD` | Required | Secret Manager | The password of that user. |
| `STREAM_PATH` | not set | `cam` | Keep. Must match the MediaMTX path. |
| `VID_STRIDE` | `1` | not used | Keep for a 15 fps camera. For a 30 fps camera set `2`. |
| `STREAM_FRAME_STRIDE` | `1` | `1` | Keep for a 15 fps camera. |
| `BUFFER_LEN` | `2` | `2` | Keep. |
| `FALL_CLEAR_WINDOWS` | `1` | `1` | Keep. |
| `EVENT_COOLDOWN_SEC` | `60` | not used | Keep. Merges an alarm that drops and returns within 60 s into one alert. |
| `SERVICE_UI_HOST` | set by compose | `0.0.0.0` | Keep. |
| `SPOOL_DIR` | set by compose | `/data/outbox` | Keep. |
| `FALL_CLIPS_ENABLED` | not set | `1` | Keep. |
| `YOLO_VERBOSE` | not set | `False` | Keep. |
| `POSE_THREADS` | not set | `4` | Keep. |

Every other detection setting has a working default. They are listed in `services/detection/v2/README.md`, section "Settings".

## GitHub secrets and variables

Cloud path only. Set them in the repo under Settings, Secrets and variables, Actions, as in [Secrets and variables](01-prerequisites/README.md#github-secrets-and-variables-cloud-path-only).

| Name | Type | Value |
|------|------|-------|
| `GCP_PROJECT_ID` | Variable | Your GCP project ID |
| `GCP_REGION` | Variable | Artifact Registry region, for example `asia-southeast1` |
| `GCP_SERVICE_ACCOUNT` | Variable | `gha-pusher@<GCP_PROJECT_ID>.iam.gserviceaccount.com`, the service account CI impersonates |
| `GCP_WIF_PROVIDER` | Variable | Full resource name of the Workload Identity provider: `projects/<project-number>/locations/global/workloadIdentityPools/github/providers/github` |
| `REFIT_APP_ID`, `REFIT_APP_PRIVATE_KEY` | Secret, optional | ID and private key of the GitHub App for the refit workflow |
| `HF_WRITE_TOKEN` | Secret, optional | Hugging Face token with write access to the weights repo, for the refit workflow's Promotion upload. Downloads need no token |

## Fixed names and versions

Names and versions that several pages repeat. Edit them here first. Each is also fixed in code or config, so a change needs the matching code change too.

| Item | Value |
|------|-------|
| Artifact Registry repo | `bbfc-registry`, images at `<GCP_REGION>-docker.pkg.dev/<GCP_PROJECT_ID>/bbfc-registry/<service>` |
| Service account for CI | `gha-pusher` |
| Workload Identity pool and provider | both named `github` |
| GKE cluster | `bbfc`, namespace `bbfc`, node pool `main`, one `e2-standard-4` node |
| Cloud SQL instance | `bbfc-db` (PostgreSQL 15, private IP only) |
| Secret Manager secrets | `telegram-bot-token`, `detection-stream-password`, `tailscale-operator-oauth` |
| Flux path and branches | `infra/flux`, `main`, image updates on `flux-image-updates` |
| Pi stream | RTSP port `8554`, path `cam`, 1280x720 at 15 fps |
| Detection page and notification API ports | `5005` and `8081` |
| MediaMTX | `v1.19.2` (`linux_arm64`) |
| Tailscale operator chart | `~1.104.0` |
| OpenTofu | 1.13 or newer |
| Region | `GCP_REGION`, for example `asia-southeast1`. The registry and the cluster zone must be in it. |

The upstream project name `fyp-elderly-detection` appears in the repo files that [Flux bootstrap](04-ci-cd-bootstrap/04-flux-bootstrap.md), [Provision](05-cloud/01-provision.md) and [Kubernetes and Tailscale](05-cloud/02-k8s-and-tailscale.md) tell you to replace.

## Next

- Local path: [Local deployment](03-local/README.md)
- Cloud path: [Kubernetes and Tailscale](05-cloud/02-k8s-and-tailscale.md)
