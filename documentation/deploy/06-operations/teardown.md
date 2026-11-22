# Tear Down and Bring Back the Cluster

This page destroys the GKE cluster `bbfc` and builds it again, with the data kept. Only the `cluster` OpenTofu root is destroyed. Cloud SQL, the clip bucket, the secrets and the registry are in the `base` root, which you never destroy.

By the end of this page you have:

- The `bbfc` cluster destroyed, with no disk or Tailscale machine left behind.
- Cloud SQL `bbfc-db` and the clip bucket untouched.
- Optionally, Cloud SQL stopped while the cluster is down.
- The cluster built again, with `notification` and `detection` running and the stream connected.

Needs: a running cloud deployment ([Verify](../05-cloud/03-verify.md)), and the `config.env` files in `infra/k8s/notification` and `infra/k8s/detection`. They are gitignored: keep them. Run the commands in bash (Git Bash on Windows), from the repo root.

## 1. Point kubectl at the cluster

Docker Desktop can switch the kubectl context. Switch it back:

```bash
kubectl config use-context gke_<GCP_PROJECT_ID>_<GCP_ZONE>_bbfc
kubectl get nodes
```

One node shows `Ready`.

## 2. Check the outbox is empty

Detection keeps alerts it has not delivered yet on the cluster's disk. Destroying the cluster loses them. The command must print `0`:

```bash
kubectl -n bbfc exec detection-0 -- python -c "from service.spool import Outbox; print(len(Outbox('/data/outbox').pending()))"
```

If it prints another number, wait and run it again. An error such as `pod not running` is not `0`: the outbox is unchecked, so fix that first.

## 3. Suspend Flux

Flux recreates what you delete, so suspend it:

```bash
flux suspend kustomization bbfc-apps
flux suspend image update flux-system
flux get kustomizations
```

`bbfc-apps` shows `SUSPENDED True`. If it does not, also run `flux suspend kustomization flux-system`.

## 4. Delete the workloads

```bash
kubectl delete -k infra/k8s
kubectl get pv
kubectl -n tailscale get pods
```

This deletes the `bbfc` namespace, with:

- The detection PVC and its disk. GKE keeps disks when a cluster is deleted, so a disk left here keeps costing money.
- The ConfigMaps you made by hand.
- `pi-mediamtx`, which makes the Tailscale operator remove the proxy machine from your tailnet.

Continue only when `kubectl get pv` prints `No resources found` and the `tailscale` namespace has only the operator pod left.

## 5. Destroy the cluster

```bash
cd infra/tofu/cluster
tofu plan -destroy
tofu destroy
cd ../../..
```

The plan lists only `google_container_cluster.bbfc` and `google_container_node_pool.main`. Type `yes` at the prompt. Flux goes with the cluster: nothing needs cleaning in git.

Never run `tofu destroy` in `infra/tofu/base`: it deletes Cloud SQL with all alerts, and the clip bucket.

In the console, open Kubernetes Engine, Clusters. `bbfc` is gone. Open SQL: `bbfc-db` still shows a green tick.

## 6. Clean up Tailscale

In the Tailscale admin console, open Machines and remove the offline `tailscale-operator`.

## 7. Optional: stop Cloud SQL

While the cluster is down nothing uses the database. Stopping it saves the instance cost. Storage is still billed.

```bash
gcloud sql instances patch bbfc-db --activation-policy=NEVER --project <GCP_PROJECT_ID>
```

## 8. Bring the cluster back: start Cloud SQL

Skip this step if you did not stop Cloud SQL.

```bash
gcloud sql instances patch bbfc-db --activation-policy=ALWAYS --project <GCP_PROJECT_ID>
```

## 9. Create the cluster

```bash
cd infra/tofu/cluster
tofu init
tofu apply
cd ../../..
gcloud container clusters get-credentials bbfc --zone <GCP_ZONE> --project <GCP_PROJECT_ID> --dns-endpoint
kubectl get nodes
```

The apply takes about 10 minutes. One node shows `Ready`.

## 10. Install the Tailscale operator

The OAuth client is still in Secret Manager. Install the operator as in [step 4 of Kubernetes and Tailscale](../05-cloud/02-k8s-and-tailscale.md#4-install-the-tailscale-operator). Its first block ends with `--wait`: run it all.

In the Tailscale admin console, Machines lists `tailscale-operator` with the tag `tag:k8s-operator`.

## 11. Create the namespace and the ConfigMaps

```bash
kubectl apply -f infra/k8s/namespace.yml
kubectl -n bbfc create configmap notification --from-env-file=infra/k8s/notification/config.env --dry-run=client -o yaml | kubectl apply -f -
kubectl -n bbfc create configmap detection --from-env-file=infra/k8s/detection/config.env --dry-run=client -o yaml | kubectl apply -f -
```

## 12. Install Flux

Create a GitHub token as in [Flux bootstrap](../04-ci-cd-bootstrap/04-flux-bootstrap.md#3-create-a-github-token-for-flux). The old token must not be expired: the deploy key stops working with it.

```bash
export GITHUB_TOKEN=<your-token>
flux check --pre
flux bootstrap github \
  --owner=<your-username> \
  --repository=BBFC-FallDetectionSystem \
  --branch=main \
  --path=infra/flux \
  --personal \
  --components-extra=image-reflector-controller,image-automation-controller
git pull origin main
unset GITHUB_TOKEN
```

`flux check --pre` passes. Flux deploys `infra/k8s` from `main` with the image tags in `infra/k8s/kustomization.yml`.

If `bbfc-apps` fails with `no matches for kind "SecretSync"`, Flux retries every minute. To retry now:

```bash
flux reconcile kustomization bbfc-apps --with-source
```

## 13. Check

```bash
flux get kustomizations
flux get image policy
kubectl -n bbfc rollout status deploy/notification
kubectl -n bbfc rollout status sts/detection
kubectl get --raw "/api/v1/namespaces/bbfc/pods/detection-0:5005/proxy/stream/status"
```

The kustomizations are `True`, both rollouts finish, and the stream status shows `"connected":true`.

Then run the alert check in [Verify](../05-cloud/03-verify.md#4-check-that-an-alert-reaches-telegram). Earlier alerts are still in Cloud SQL and their clips are still in the bucket.

## Next

[Operations](README.md)
