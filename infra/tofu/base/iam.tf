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

resource "google_service_account_iam_member" "notification_wi" {
  member             = "serviceAccount:${var.project_id}.svc.id.goog[bbfc/notification]"
  role               = "roles/iam.workloadIdentityUser"
  service_account_id = google_service_account.notification.name
}

resource "google_secret_manager_secret_iam_member" "ksa" {
  for_each = {
    "telegram-bot-token"        = "notification"
    "detection-stream-password" = "detection"
  }

  member    = "principal://iam.googleapis.com/projects/${var.project_number}/locations/global/workloadIdentityPools/${var.project_id}.svc.id.goog/subject/ns/bbfc/sa/${each.value}"
  role      = "roles/secretmanager.secretAccessor"
  secret_id = google_secret_manager_secret.secret[each.key].secret_id

}