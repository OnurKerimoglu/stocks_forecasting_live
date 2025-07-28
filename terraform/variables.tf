variable "gcp_key_path_src" {sensitive=true}
variable "project" {
  description = "Project"
  default     = "stocks-forecasting-466906"
}

variable "zone" {
  description = "Zone"
  #Update the below to your desired region
  default     = "europe-west1-b"
}

variable "region" {
  description = "Region"
  #Update the below to your desired region
  default     = "europe-west1"
}

variable "location" {
  description = "Project Location"
  #Update the below to your desired location
  default     = "EU"
}

variable "gcs_data_monitoring_bucket_name" {
  description = "My Storage Bucket Name"
  #Update the below to a unique bucket name
  default     = "stocks-forecasting-data-monitoring"
}

variable "gcs_models_test_bucket_name" {
  description = "My Storage Bucket Name"
  #Update the below to a unique bucket name
  default     = "stocks-forecasting-models-test"
}

variable "gcs_models_dev_bucket_name" {
  description = "My Storage Bucket Name"
  #Update the below to a unique bucket name
  default     = "stocks-forecasting-models-dev"
}

variable "gcs_models_prod_bucket_name" {
  description = "My Storage Bucket Name"
  #Update the below to a unique bucket name
  default     = "stocks-forecasting-models-prod"
}

variable "repo_name" {
  description = "The Artifact Registry repository name"
  default     = "stocks-forecasting-repo"
  type        = string
}
