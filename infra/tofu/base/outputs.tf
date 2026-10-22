output "project_id" {
  value = var.project_id
}

output "project_number" {
  value = var.project_number
}

output "region" {
  value = var.region
}

output "zone" {
  value = var.zone
}

output "network" {
  value = google_compute_network.vpc.id
}

output "subnetwork" {
  value = google_compute_subnetwork.subnet.id
}

output "sql_connection_name" {
  value = google_sql_database_instance.db.connection_name
}

output "clip_bucket_name" {
  value = google_storage_bucket.clips.name
}

output "notification_gsa_email" {
  value = google_service_account.notification.email
}

output "gke_nodes_gsa_email" {
  value = google_service_account.gke_nodes.email
}
