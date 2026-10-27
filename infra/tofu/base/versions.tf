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
    prefix = "bbfc/base"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}