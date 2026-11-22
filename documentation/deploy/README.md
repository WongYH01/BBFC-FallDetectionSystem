# Deployment Guide

This guide takes an operator from an empty Raspberry Pi to a running fall-detection system that sends alerts to Telegram. It has two paths: local (docker-compose on your computer) and cloud (GKE, deployed by CI and Flux). Work through the pages in the order below.

By the end of this guide you have:

- A Pi streaming its camera over RTSP.
- Detection and notification running, on your computer or on GKE.
- A `FALL DETECTED` alert, with its skeleton clip, delivered to your Telegram group.

Needs: nothing yet. [Prerequisites](01-prerequisites/README.md) lists the hardware, accounts and tools.

## Pick a path

| | Local | Cloud |
|---|---|---|
| Steps | 1, 2, 3, 6 | 1, 2, 4, 5, 6 |
| Detection and notification run on | Your computer, in Docker | GKE |
| Pi reached over | Same LAN | Tailscale |

Steps 1, 2 and 6 apply to both paths.

## Setup order

1. [Prerequisites](01-prerequisites/README.md): hardware, accounts, tools, and the `.env` and GitHub values you need later.
2. [Pi camera](02-pi-camera/README.md): flash the Pi, connect the camera, serve the stream with MediaMTX.
3. [Local deployment](03-local/README.md) (local path): run the whole stack with `docker compose`.
4. CI/CD bootstrap (cloud path), in this order:
   1. [GitHub repo setup](04-ci-cd-bootstrap/01-github-repo-setup.md)
   2. [GCP auth and registry](04-ci-cd-bootstrap/02-gcp-auth-and-registry.md)
   3. [Workflows](04-ci-cd-bootstrap/03-workflows.md)
   4. [Flux bootstrap](04-ci-cd-bootstrap/04-flux-bootstrap.md)
   5. [Verify pipeline](04-ci-cd-bootstrap/05-verify-pipeline.md)
5. Cloud (cloud path), in this order:
   1. [Provision](05-cloud/01-provision.md)
   2. [Kubernetes and Tailscale](05-cloud/02-k8s-and-tailscale.md)
   3. [Verify](05-cloud/03-verify.md)
6. [Operations](06-operations/README.md): update, restart, rotate secrets. [Teardown](06-operations/teardown.md): remove the cluster and bring it back.

## Reference

- [Environment variables](env-vars.md): every variable in every `.env` and config example.

## Further reading

- [`CONTEXT.md`](../../CONTEXT.md): the project glossary.
- [`CLAUDE.md`](../../CLAUDE.md): conventions for working in the repo.
- Decision records: [0001](../../docs/adr/0001-no-camera-pixels-leave-the-detection-host.md), [0002](../../docs/adr/0002-alert-keypoints-are-kept-for-training.md), [0003](../../docs/adr/0003-camera-heads-are-promoted-without-human-review.md).
