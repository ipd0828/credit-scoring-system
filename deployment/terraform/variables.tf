# variables.tf - исправленная версия без дубликатов

variable "yandex_cloud_id" {
  description = "Yandex Cloud ID"
  type        = string
  sensitive   = true
}

variable "yandex_folder_id" {
  description = "Yandex Folder ID"
  type        = string
  sensitive   = true
}

variable "cluster_name" {
  description = "Kubernetes cluster name"
  type        = string
  default     = "credit-scoring"  #
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging or prod."
  }
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "k8s_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.30"
}

variable "node_disk_size" {
  description = "Node disk size in GB"
  type        = number
  default     = 50
}

variable "min_node_count" {
  description = "Minimum node count"
  type        = number
  default     = 2
}

variable "max_node_count" {
  description = "Maximum node count"
  type        = number
  default     = 3
}

variable "node_cores" {
  description = "Number of CPU cores per node"
  type        = number
  default     = 2
}

variable "node_memory" {
  description = "RAM per node in GB"
  type        = number
  default     = 4
}

variable "preemptible_nodes" {
  description = "Use preemptible nodes (cheaper)"
  type        = bool
  default     = true
}

variable "use_existing_network" {
  description = "Use existing default network"
  type        = bool
  default     = true
}

variable "s3_access_key" {
  description = "Access key for S3 storage (leave empty to use service account)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "s3_secret_key" {
  description = "Secret key for S3 storage (leave empty to use service account)"
  type        = string
  sensitive   = true
  default     = ""
}