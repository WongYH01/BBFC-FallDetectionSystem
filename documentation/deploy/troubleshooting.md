# Troubleshooting

Symptom index for the deployment guide. Each entry gives the symptom, the cause, a read-only check and the fix. Every fix here is already stated on a step page, collected in one place. Nothing here is new advice. If your symptom is not listed, stop and ask the operator.

Each step page's "If it fails" section links to the matching entry. Entries are grouped by step.

Rules that hold for every fix:

- Never print a secret to check it. Compare hashes, as in [Kubernetes Secret holds the old value](#kubernetes-secret-holds-the-old-value).
- Ask the operator before `tofu apply`, `tofu destroy` or `git push`.
- Never `tofu destroy` in `infra/tofu/base`.

## Pi camera ([step 2](02-pi-camera/README.md))

### Camera list is empty

- Cause: the camera cable is not seated, or the Pi was powered on before the camera was connected.
- Check (Pi, Linux): `rpicam-hello --list-cameras` prints no camera.
- Fix: power the Pi off, connect the camera as in [Connect the camera](02-pi-camera/README.md#2-connect-the-camera), power on, check again. This is a human step.

### Pi hostname does not resolve

- Cause: `<hostname>.local` is not resolved on your network.
- Check: `ssh <username>@<hostname>.local` fails.
- Fix: find the Pi's IP address on your router's device list and use `ssh <username>@<ip-address>`.

### MediaMTX is not running

- Cause: bad `mediamtx.yml` or service file.
- Check (Pi, Linux): `sudo systemctl status mediamtx` does not show `active (running)`. Read the log with `sudo journalctl -u mediamtx.service -f`.
- Fix: correct the config as in [Configure MediaMTX](02-pi-camera/README.md#6-configure-mediamtx) and [Start MediaMTX on boot](02-pi-camera/README.md#7-start-mediamtx-on-boot), then `sudo systemctl restart mediamtx`. Keep `rpiCameraBitrate` at `1000000` or above: an under-fed encoder wedges. The MediaMTX build must be `linux_arm64`, so the OS must be 64-bit.

### ffprobe cannot read the stream

- Cause: wrong address or credentials, MediaMTX down, or (cloud path) your computer is not on the tailnet.
- Check: `ffprobe rtsp://<STREAM_USERNAME>:<STREAM_PASSWORD>@<pi-address>:8554/cam` lists no `Video: h264` stream.
- Fix: use the LAN address on the local path and the `100.x.x.x` Tailscale address on the cloud path. Check MediaMTX with [MediaMTX is not running](#mediamtx-is-not-running).

## Local deployment ([step 3](03-local/README.md))

### Notification is not healthy

- Cause: `detection` waits for `notification` to be `healthy`.
- Check: `docker compose ps` shows `notification` not `healthy`.
- Fix: read `docker compose logs -f notification`. Check that `.env` has no `{INSERT...}` placeholder left, as in [Fill in the two `.env` files](03-local/README.md#2-fill-in-the-two-env-files).

### Test alert returns duplicate

- Cause: the notification service creates at most one Alert per event ID.
- Check: the reply is `duplicate` and no Telegram message arrives.
- Fix: send the test again with a new `eventId`, for example `test-002`.

### No Telegram message arrives

- Cause: the bot is not a member of the group, or `TELEGRAM_CHAT_GROUP_ID` or `TELEGRAM_BOT_TOKEN` is wrong.
- Check: the reply to the test alert is `accepted` but the group stays silent. Cloud path: the message can take up to 20 seconds, because the service waits for a clip.
- Fix: add the bot to the group. Recheck the [bot token](01-prerequisites/README.md#bot-token) and the [chat group ID](01-prerequisites/README.md#chat-group-id).

### First build is slow

- Cause: the detection image downloads the model weights from the public Hugging Face repo during the build.
- Check: `docker compose up -d --build` runs for many minutes.
- Fix: wait. If the build fails on the download, check that the host can reach `huggingface.co`.

## GitHub and CI ([step 4](04-ci-cd-bootstrap/01-github-repo-setup.md))

### Code scanning or dependency review fails

- Cause: CodeQL and Trivy upload to code scanning, which a private repo needs GitHub Advanced Security for. `dependency-review` needs the dependency graph.
- Check: the `ci` run fails in `codeql`, the Trivy scan or `dependency-review`.
- Fix: turn on the dependency graph as in [Check code scanning and the dependency graph](04-ci-cd-bootstrap/01-github-repo-setup.md#3-check-code-scanning-and-the-dependency-graph). This is a human step in the GitHub console.

### CI builds no image

- Cause: `ci` builds only the services whose files changed.
- Check: the `ci` run has no `*-image` job.
- Fix: change `docker-compose.yml`, which builds both services, as in [Run the pipeline](04-ci-cd-bootstrap/03-workflows.md#3-run-the-pipeline). Images are pushed only on a push to `main`.

### Refit fails on Mondays

- Cause: a fork runs the weekly `refit` schedule, and this guide does not set up its bucket or GitHub App.
- Check: `gh workflow list --repo <your-username>/BBFC-FallDetectionSystem --all` shows `refit` as active.
- Fix: `gh workflow disable refit --repo <your-username>/BBFC-FallDetectionSystem`.

## Flux and the registry ([step 4](04-ci-cd-bootstrap/04-flux-bootstrap.md))

### bbfc-apps is not True

- Cause: before the cloud pages are done, `bbfc-apps` lacks the Kubernetes settings and the Tailscale setup.
- Check: `flux get kustomizations`.
- Fix: finish [Kubernetes and Tailscale](05-cloud/02-k8s-and-tailscale.md). Then `flux reconcile kustomization bbfc-apps --with-source`. If the reconcile times out, wait a minute: Flux retries every minute, and the detection image is large.

### SecretSync kind missing

- Cause: after a cluster rebuild, the `SecretSync` kind does not exist yet when Flux first applies.
- Check: `bbfc-apps` fails with `no matches for kind "SecretSync"`.
- Fix: wait: Flux retries every minute. To retry now, `flux reconcile kustomization bbfc-apps --with-source`.

### Flux does not see the new tag

- Cause: the image reflector scans every 5 minutes.
- Check: `flux get image repository` and `flux get image policy`.
- Fix: `flux reconcile image repository detection` and `flux reconcile image repository notification`, then `flux reconcile image update flux-system`.

### Flux deploy key stops working

- Cause: the GitHub token used at bootstrap expired.
- Check: bootstrap fails, or Flux cannot sync the fork.
- Fix: create a new token as in [Create a GitHub token for Flux](04-ci-cd-bootstrap/04-flux-bootstrap.md#3-create-a-github-token-for-flux). This is a human step.

## GKE and Tailscale ([step 5](05-cloud/02-k8s-and-tailscale.md))

### RTSP stream times out

- Cause: the Tailscale policy has no grant from `tag:k8s` to `tag:pi` on `tcp:8554`, or the Pi is not tagged `tag:pi`.
- Check: `curl http://localhost:5005/stream/status` through the port-forward shows `fps` of `0`.
- Fix: add the grant and tag the Pi, as in [Set the Tailscale policy](05-cloud/02-k8s-and-tailscale.md#2-set-the-tailscale-policy). This is a human step in the Tailscale console.

### Camera stream stops on the cluster

- Cause: the Tailscale egress proxy is stuck.
- Check: `kubectl -n tailscale get pods` and `kubectl -n bbfc get svc pi-mediamtx`.
- Fix: `kubectl -n tailscale rollout restart sts/egress-pg`. `egress-pg-0` is `Running` again after a minute.

### Duplicate tailscale-operator machine

- Cause: a reinstall names the new machine `tailscale-operator-1`.
- Check: the Tailscale admin console, Machines, lists an offline `tailscale-operator`.
- Fix: delete the old, offline entry. This is a human step.

### Kubernetes Secret holds the old value

- Cause: GKE syncs the Kubernetes Secret from Secret Manager with a delay.
- Check: the two hashes differ.
  `kubectl -n bbfc get secret notification -o jsonpath='{.data.TELEGRAM_BOT_TOKEN}' | base64 -d | sha256sum`, and `printf '%s' '<NEW_TELEGRAM_BOT_TOKEN>' | sha256sum`. For the stream password use secret `detection` and key `STREAM_PASSWORD`.
- Fix: wait until the hashes match, then restart the workload, as in [Rotate the Telegram bot token](06-operations/README.md#rotate-the-telegram-bot-token).

### kubectl points at the wrong cluster

- Cause: Docker Desktop can switch the kubectl context.
- Check: `kubectl config current-context` is not `gke_<GCP_PROJECT_ID>_<GCP_ZONE>_bbfc`.
- Fix: `kubectl config use-context gke_<GCP_PROJECT_ID>_<GCP_ZONE>_bbfc`.

## Teardown ([step 6](06-operations/teardown.md))

### Outbox check does not print 0

- Cause: detection holds alerts it has not delivered. An error such as `pod not running` is not `0`: the outbox is unchecked.
- Check: the command in [Check the outbox is empty](06-operations/teardown.md#2-check-the-outbox-is-empty).
- Fix: wait and run it again. Fix a pod that is not running first. Do not go on until it prints `0`.

### Disk left after teardown

- Cause: GKE keeps disks when a cluster is deleted, and a disk left over keeps costing money.
- Check: `kubectl get pv` prints something other than `No resources found`.
- Fix: do not destroy the cluster yet. Finish [Delete the workloads](06-operations/teardown.md#4-delete-the-workloads) first.

## Never a fix

Do not send camera pixels off the detection host while troubleshooting, for example by recording the camera or uploading a frame. See [ADR 0001](../../docs/adr/0001-no-camera-pixels-leave-the-detection-host.md).
