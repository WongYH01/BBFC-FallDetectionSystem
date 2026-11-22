# Agent instructions

Entry point for every coding agent (Claude Code, Gemini, Cursor, Copilot). `CLAUDE.md` imports this file.

## Deploying the system

Follow the [deployment guide](documentation/deploy/README.md). Pick one path, then take the steps in order:

| Path | Steps |
|------|-------|
| Local | 1, 2, 3, 6 |
| Cloud | 1, 2, 4, 5, 6 |

| Step | Page |
|------|------|
| 1 Prerequisites | [documentation/deploy/01-prerequisites/README.md](documentation/deploy/01-prerequisites/README.md) |
| 2 Pi camera | [documentation/deploy/02-pi-camera/README.md](documentation/deploy/02-pi-camera/README.md) |
| 3 Local deployment | [documentation/deploy/03-local/README.md](documentation/deploy/03-local/README.md) |
| 4.1 to 4.5 CI/CD bootstrap | [documentation/deploy/04-ci-cd-bootstrap/](documentation/deploy/04-ci-cd-bootstrap/01-github-repo-setup.md): 01 repo setup, 02 GCP auth and registry, 03 workflows, 04 Flux bootstrap, 05 verify pipeline |
| 5.1 to 5.3 Cloud | [documentation/deploy/05-cloud/](documentation/deploy/05-cloud/01-provision.md): 01 provision, 02 Kubernetes and Tailscale, 03 verify |
| 6 Operations | [documentation/deploy/06-operations/README.md](documentation/deploy/06-operations/README.md), [teardown](documentation/deploy/06-operations/teardown.md) |
| Any value, name or version | [documentation/deploy/env-vars.md](documentation/deploy/env-vars.md) |
| Any symptom | [documentation/deploy/troubleshooting.md](documentation/deploy/troubleshooting.md) |

Every step page has the sections `Needs`, `Who`, `Do`, `Done when`, `If it fails`.

- Read `Needs` and `Who` before you act.
- `Who` is one of: agent can run, human only, or human provides X and the agent continues. Stop at human-only steps. Ask for exactly the values `Who` lists, once.
- Run `Done when` after every step. Do not go on until it matches.
- On failure, follow `If it fails` to the troubleshooting entry. If your symptom is not there, stop and ask.
- Command blocks are labelled with their shell. Use the PowerShell variant on Windows PowerShell. Pi steps run on Linux over SSH.
- A screenshot never holds information the text lacks. Do not depend on images.

## Guardrails

- Never print or commit a secret: tokens, passwords, keys, `.env` files, `config.env` files. Compare hashes to check a secret.
- Ask the operator before any `tofu apply` or `tofu destroy`, and before `git push`.
- Treat [teardown](documentation/deploy/06-operations/teardown.md) as confirm-first. Never `tofu destroy` in `infra/tofu/base`.
- No camera pixels leave the detection host. Do not record, upload or send camera frames while deploying or troubleshooting. See [ADR 0001](docs/adr/0001-no-camera-pixels-leave-the-detection-host.md).
- Use the terms in [CONTEXT.md](CONTEXT.md). Read the [decision records](docs/adr/) that touch your area.

## Folders

- `documentation/`: operator guides. `documentation/deploy/` is the deployment guide and is authoritative.
- `docs/`: decision records (`docs/adr/`), agent tooling notes (`docs/agents/`), and historical planning documents.

## Extending the guide

Keep this file's step table, the path split and the step pages in sync. A new step page uses the five-section template, labels its shells, links its failures to `troubleshooting.md`, and takes repeated values from `env-vars.md`.
