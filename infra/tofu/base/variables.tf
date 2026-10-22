variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "project_number" {
  description = "GCP project number, for principal:// members"
  type        = string
}

variable "region" {
  description = "Region for all regional resources"
  type        = string
}

variable "zone" {
  description = "Zone for the GKE cluster"
  type        = string
}

variable "clip_bucket_name" {
  description = "Bucket for skeleton clips and keypoint files"
  type        = string
}