resource "google_sql_database_instance" "db" {
  name             = "bbfc-db"
  database_version = "POSTGRES_15"
  region           = var.region

  # deletion_protection = true
  # lifecycle {
  #   prevent_destroy = true
  # }

  settings {
    tier    = "db-f1-micro"
    edition = "ENTERPRISE"

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
    }

    database_flags {
      name  = "cloudsql.iam_authentication"
      value = "on"
    }

    backup_configuration {
      enabled = true
    }
  }

  depends_on = [google_service_networking_connection.sql_psa]
}

resource "google_sql_user" "notification" {
  name     = trimsuffix(google_service_account.notification.email, ".gserviceaccount.com")
  instance = google_sql_database_instance.db.name
  type     = "CLOUD_IAM_SERVICE_ACCOUNT"
}


