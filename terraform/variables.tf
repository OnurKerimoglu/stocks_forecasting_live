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

variable "gcp_key_path_src" {sensitive=true}

variable "repo_name" {
  description = "The Artifact Registry repository name"
  default     = "stocks-forecasting-repo"
  type        = string
}
