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