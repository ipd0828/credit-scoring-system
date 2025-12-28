terraform {
  required_version = ">= 1.5.0"

  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = ">= 0.95.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.23.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.11.0"
    }
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = ">= 1.14.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.5.1"
    }
    null = {
      source  = "hashicorp/null"
      version = ">= 3.2.1"
    }
  }

  # Временно закомментируем этот блок:
  # backend "s3" {
  #   endpoint   = "storage.yandexcloud.net"
  #   bucket     = "credit-scoring-terraform-state"
  #   region     = "ru-central1"
  #   key        = "terraform.tfstate"
  #
  #
  #   skip_region_validation      = true
  #   skip_credentials_validation = true
  #   skip_metadata_api_check     = true
  # }
}

provider "yandex" {
  service_account_key_file = "yandex-key.json"
  cloud_id                 = var.yandex_cloud_id
  folder_id                = var.yandex_folder_id
  zone                     = "ru-central1-a"
}

provider "random" {}
provider "null" {}