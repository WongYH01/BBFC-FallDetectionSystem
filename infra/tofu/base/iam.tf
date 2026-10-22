resource "google_service_account" "gke_nodes" {
  account_id   = "gke-nodes"
  display_name = "GKE Nodes"
}

resource "google_project_iam_member" "gke_nodes" {
  member  = google_service_account.gke_nodes.member
  project = var.project_id
  role    = "roles/container.defaultNodeServiceAccount"
}

resource "google_artifact_registry_repository_iam_member" "gke_nodes" {
  member     = google_service_account.gke_nodes.member
  repository = google_artifact_registry_repository.registry.name
  role       = "roles/artifactregistry.reader"
  project    = google_artifact_registry_repository.registry.project
  location   = google_artifact_registry_repository.registry.location
}

resource "google_service_account" "notification" {
  account_id   = "notification"
  display_name = "Notification service"
}

resource "google_project_iam_member" "notification" {
  for_each = toset(["roles/cloudsql.client", "roles/cloudsql.instanceUser"])
  project  = var.project_id
  role     = each.value
  member   = google_service_account.notification.member
}
