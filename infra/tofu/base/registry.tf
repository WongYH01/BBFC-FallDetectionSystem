import {
  to = google_artifact_registry_repository.registry
  id = "projects/${var.project_id}/locations/${var.region}/repositories/bbfc-registry"
}

resource "google_artifact_registry_repository" "registry" {
  location               = var.region
  repository_id          = "bbfc-registry"
  format                 = "DOCKER"
  cleanup_policy_dry_run = false

  lifecycle {
    prevent_destroy = true
  }

  cleanup_policies {
    id     = "Keep 5 most recent version"
    action = "KEEP"
    most_recent_versions {
      keep_count = 5
    }
  }

  cleanup_policies {
    id     = "Delete older than 15 days"
    action = "DELETE"
    condition {
      tag_state  = "ANY"
      older_than = "1296000s" # 15 days (15 * 24 * 60 * 60)
    }
  }
}