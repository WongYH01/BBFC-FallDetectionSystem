# Cluster runbook

PowerShell, from the repo root. `base` is never destroyed; only `cluster` is.

Make sure kubectl points at GKE (Docker Desktop can switch it):
```powershell
kubectl config use-context gke_fyp-elderly-detection_asia-southeast1-a_bbfc
```

## Tear down

1. Outbox empty: must print `0`, otherwise wait for pending alerts to deliver.
   ```powershell
   kubectl -n bbfc exec detection-0 -- python -c "from service.spool import Outbox; print(len(Outbox('/data/outbox').pending()))"
   ```
2. Delete the workloads. Removes the Pi proxy machine from the tailnet and the
   detection disk (GKE keeps disks when a cluster is deleted). Continue only
   when `get pv` shows `No resources found`.
   ```powershell
   kubectl delete -k infra/tofu/k8s
   kubectl get pv
   ```
3. Destroy the cluster.
   ```powershell
   tofu -chdir=infra/tofu/cluster destroy
   ```
4. Tailscale admin → [Machines](https://console.tailscale.com/admin/machines)
   → remove the offline `tailscale-operator`.
5. Optional: stop Cloud SQL (storage is still billed).
   ```powershell
   gcloud sql instances patch bbfc-db --activation-policy=NEVER --project fyp-elderly-detection
   ```

## Bring up

1. If Cloud SQL was stopped, start it.
   ```powershell
   gcloud sql instances patch bbfc-db --activation-policy=ALWAYS --project fyp-elderly-detection
   ```
2. Create the cluster, then fetch credentials.
   ```powershell
   tofu -chdir=infra/tofu/cluster apply
   gcloud container clusters get-credentials bbfc --zone asia-southeast1-a --project fyp-elderly-detection --dns-endpoint
   ```
3. Install the Tailscale operator. First time on a machine, run
   `helm repo add tailscale https://pkgs.tailscale.com/helmcharts` and
   `helm repo update` before this.
   ```powershell
   $o = (gcloud secrets versions access latest --secret=tailscale-operator-oauth --project fyp-elderly-detection) -join "" | ConvertFrom-Json
   helm upgrade --install tailscale-operator tailscale/tailscale-operator --version 1.104.2 --namespace tailscale --create-namespace --set-string "oauth.clientId=$($o.client_id)" --set-string "oauth.clientSecret=$($o.client_secret)" --wait
   Remove-Variable o
   ```
4. Optional: deploy the newest images instead of the pinned digests.
   ```powershell
   $reg = "asia-southeast1-docker.pkg.dev/fyp-elderly-detection/bbfc-registry"
   Push-Location infra/tofu/k8s
   foreach ($s in "notification", "detection") {
     $d = gcloud artifacts docker images describe "$reg/${s}:latest" --format="value(image_summary.digest)"
     kustomize edit set image "$s=$reg/$s@$d"
   }
   Pop-Location
   ```
5. Deploy. `notification/config.env` and `detection/config.env` are
   gitignored: recreate them first on a fresh clone. If it fails with
   `no matches for kind "SecretSync"`, wait a minute and rerun.
   ```powershell
   kubectl apply -k infra/tofu/k8s
   ```
6. Check: both rollouts finish and the stream shows `"connected":true`.
   ```powershell
   kubectl -n bbfc rollout status deploy/notification
   kubectl -n bbfc rollout status sts/detection
   kubectl get --raw "/api/v1/namespaces/bbfc/pods/detection-0:5005/proxy/stream/status"
   ```
7. Web UI at http://localhost:5005 while this runs:
   ```powershell
   kubectl -n bbfc port-forward pod/detection-0 5005:5005
   ```
