terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.6.0"
    }
  }
}

provider "google" {
  credentials = file(var.gcp_key_path_src)
  project     = var.project
  region      = var.region
}

################################################################################
# Enable and provision GCS
################################################################################
resource "google_project_service" "cloud_storage" {
  project = var.project
  service = "storage-api.googleapis.com"
  # prevents disabling on destroy (optional)
  disable_on_destroy = false
}

resource "google_storage_bucket" "models-dev-bucket" {
  name          = var.gcs_models_dev_bucket_name
  location      = var.location
  force_destroy = true


  lifecycle_rule {
    condition {
      age = 1
    }
    action {
      type = "AbortIncompleteMultipartUpload"
    }
  }
}

################################################################################
# Enable and provision GAR
################################################################################
resource "google_project_service" "artifact_registry" {
  project = var.project
  service = "artifactregistry.googleapis.com"
  # prevents disabling on destroy (optional)
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "forecasting_repo" {
  depends_on    = [google_project_service.artifact_registry]
  project       = var.project
  location      = var.region
  repository_id = var.repo_name
  description   = "Artifact Registry for Stocks Forecasting Inference"
  format        = "DOCKER"

  docker_config {
    immutable_tags = false
  }
}

################################################################################
# Enable Cloud Run
################################################################################
resource "google_project_service" "run" {
  project = var.project
  service = "run.googleapis.com"

  # Keep the API on even after destroy
  disable_on_destroy = false
}
