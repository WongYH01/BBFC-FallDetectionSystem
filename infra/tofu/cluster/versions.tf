terraform {
  required_version = ">= 1.13"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.0"
    }
  }

  backend "gcs" {
    bucket = "fyp-elderly-detection-tofu-state"
    prefix = "bbfc/cluster"
  }
}

provider "google" {
  project = local.base.project_id
  region  = local.base.region
}
