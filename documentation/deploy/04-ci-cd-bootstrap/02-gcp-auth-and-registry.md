# GCP Auth and Registry

The image workflow pushes to Artifact Registry as a Google service account, and it logs in through Workload Identity Federation, so no key file is stored in GitHub. This page creates the registry, the service account and the federation, then sets the four `GCP_*` variables.

By the end of this page you have:

- An Artifact Registry Docker repo named `bbfc-registry`.
- A service account `gha-pusher` that can push to it.
- A Workload Identity pool and provider that let only your fork authenticate as `gha-pusher`.
- The four `GCP_*` variables set on your fork.

## Needs

The finished [GitHub repo setup](01-github-repo-setup.md), a [GCP project](../01-prerequisites/README.md#gcp-cloud-only) with billing enabled, and [gcloud](../01-prerequisites/README.md#cloud-only) installed.

## Who

agent can run the `gcloud` commands, after the human signs in.

- Human only: `gcloud auth login` opens a browser.
- Human provides: `<GCP_PROJECT_ID>`, `<GCP_REGION>` and the GitHub username of the fork. See [Names and versions](../env-vars.md#fixed-names-and-versions).
- Ask first before setting repository variables.

Shell: bash. On PowerShell use the variant noted in step 1.

## Do

### 1. Log in and pick the project

```bash
gcloud auth login
gcloud config set project <GCP_PROJECT_ID>
```

Set these shell variables. The commands below use them. Use the same region everywhere: the `GCP_REGION` variable and the registry must match.

```bash
export PROJECT_ID=<GCP_PROJECT_ID>
export REGION=<GCP_REGION>            # for example asia-southeast1
export GITHUB_REPO=<your-username>/BBFC-FallDetectionSystem
export PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
```

On PowerShell, set them with `$env:PROJECT_ID = "..."` and write `$env:PROJECT_ID` where the commands below write `$PROJECT_ID`.

### 2. Enable the APIs

```bash
gcloud services enable artifactregistry.googleapis.com iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com
```

### 3. Create the registry

The name must be `bbfc-registry`. The workflow and the Flux config both use it.

```bash
gcloud artifacts repositories create bbfc-registry \
  --repository-format=docker \
  --location="$REGION"
```

In the console, open Artifact Registry, Repositories. `bbfc-registry` is listed with format Docker and your region.

![Artifact Registry repositories](../../images/04-images/gcp-artifact-registry.jpg)

The OpenTofu `base` root in the [Cloud pages](../05-cloud/01-provision.md) imports this repo and adds its cleanup policy. Do not add a cleanup policy here.

### 4. Create the service account

The name must be `gha-pusher`. The OpenTofu `base` root refers to it by this name.

```bash
gcloud iam service-accounts create gha-pusher --display-name="GitHub Actions pusher"

gcloud artifacts repositories add-iam-policy-binding bbfc-registry \
  --location="$REGION" \
  --member="serviceAccount:gha-pusher@$PROJECT_ID.iam.gserviceaccount.com" \
  --role=roles/artifactregistry.writer
```

In the console, open IAM & Admin, Service Accounts. `gha-pusher` is listed and enabled.

![Service accounts list](../../images/04-images/gcp-service-accounts.jpg)

### 5. Create the Workload Identity pool and provider

The attribute condition lets only your fork authenticate.

```bash
gcloud iam workload-identity-pools create github \
  --location=global \
  --display-name="GitHub"

gcloud iam workload-identity-pools providers create-oidc github \
  --location=global \
  --workload-identity-pool=github \
  --issuer-uri=https://token.actions.githubusercontent.com \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='$GITHUB_REPO'"
```

Allow identities from your fork to impersonate `gha-pusher`:

```bash
gcloud iam service-accounts add-iam-policy-binding \
  "gha-pusher@$PROJECT_ID.iam.gserviceaccount.com" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/$GITHUB_REPO"
```

In the console, open IAM & Admin, Workload Identity Federation. The pool `github` is listed with 1 provider.

![Workload Identity pools](../../images/04-images/gcp-wif-pool.jpg)

### 6. Set the GitHub variables

Print the provider's full resource name:

```bash
gcloud iam workload-identity-pools providers describe github \
  --location=global \
  --workload-identity-pool=github \
  --format='value(name)'
```

Open your fork, then Settings, Secrets and variables, Actions, the Variables tab, and add four repository variables:

Set `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_SERVICE_ACCOUNT` and `GCP_WIF_PROVIDER`, with the values in [GitHub secrets and variables](../env-vars.md#github-secrets-and-variables). `GCP_WIF_PROVIDER` is the output of the previous command.

These are variables, not secrets. Click New repository variable on the Variables tab, enter a name and value, and click Add variable. Repeat for each row.

![New variable form](../../images/04-images/new-variable.jpg)

Or set them with the GitHub CLI:

```bash
gh variable set GCP_PROJECT_ID --repo "$GITHUB_REPO" --body "$PROJECT_ID"
gh variable set GCP_REGION --repo "$GITHUB_REPO" --body "$REGION"
gh variable set GCP_SERVICE_ACCOUNT --repo "$GITHUB_REPO" --body "gha-pusher@$PROJECT_ID.iam.gserviceaccount.com"
gh variable set GCP_WIF_PROVIDER --repo "$GITHUB_REPO" --body "projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/github"
```

## Done when

Read-only:

```bash
gcloud artifacts repositories describe bbfc-registry --location="$REGION" --format='value(format)'
gcloud iam workload-identity-pools providers describe github --location=global --workload-identity-pool=github --format='value(state)'
gh variable list --repo "$GITHUB_REPO"
```

Expected: `DOCKER`, then `ACTIVE`, then a list of the four `GCP_*` variables. On PowerShell write `$env:REGION` and `$env:GITHUB_REPO`.

## If it fails

- [Troubleshooting index](../troubleshooting.md)

## Next

[Workflows](03-workflows.md)
