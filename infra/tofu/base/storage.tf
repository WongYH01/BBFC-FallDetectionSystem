resource "google_storage_bucket" "clips" {
  name                        = var.clip_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
}

resource "google_storage_bucket_iam_member" "notification" {
  bucket = google_storage_bucket.clips.name
  member = google_service_account.notification.member
  role   = "roles/storage.objectAdmin"
}

# The weekly Refit on GitHub Actions mirrors keypoints/ (labelled Alerts and
# the pinned base data) and nothing else: no write, no skeleton clips. A list
# has no object name to match, so it is allowed by the prefix it asks for.
resource "google_storage_bucket_iam_member" "actions_keypoints" {
  bucket = google_storage_bucket.clips.name
  member = "serviceAccount:gha-pusher@${var.project_id}.iam.gserviceaccount.com"
  role   = "roles/storage.objectViewer"

  condition {
    title      = "keypoints-only"
    expression = "resource.name.startsWith(\"projects/_/buckets/${google_storage_bucket.clips.name}/objects/keypoints/\") || api.getAttribute(\"storage.googleapis.com/objectListPrefix\", \"\").startsWith(\"keypoints/\")"
  }
}
