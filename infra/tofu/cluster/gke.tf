resource "google_container_cluster" "bbfc" {
  name                     = "bbfc"
  location                 = local.base.zone
  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false

  network    = local.base.network
  subnetwork = local.base.subnetwork

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  release_channel {
    channel = "REGULAR"
  }

  workload_identity_config {
    workload_pool = "${local.base.project_id}.svc.id.goog"
  }

  secret_sync_config {
    enabled = true
  }

  control_plane_endpoints_config {
    dns_endpoint_config {
      allow_external_traffic = true
    }
  }

  maintenance_policy {
    recurring_window {
      start_time = "2026-01-01T18:00:00Z"
      end_time   = "2026-01-01T22:00:00Z"
      recurrence = "FREQ=DAILY"
    }
  }
}

resource "google_container_node_pool" "main" {
  name       = "main"
  cluster    = google_container_cluster.bbfc.id
  location   = local.base.zone
  node_count = 1

  node_config {
    machine_type    = "e2-standard-4"
    disk_size_gb    = 50
    service_account = local.base.gke_nodes_gsa_email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}