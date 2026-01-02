terraform {
  required_providers {
    yandex = {
      source = "yandex-cloud/yandex"
    }
  }
}

# Локальные переменные
locals {
  folder_id = "b1gtp2vtkrplf280eu4g"
  cloud_id  = "b1g8vpvl76pkjcno9uqo"
}

# Провайдер
provider "yandex" {
  cloud_id  = local.cloud_id
  folder_id = local.folder_id
  zone      = "ru-central1-a"
  service_account_key_file = "../../yandex-key.json"
}

# 1. Service Account для кластера
resource "yandex_iam_service_account" "k8s_sa" {
  name        = "k8s-cluster-sa"
  description = "Service account for Kubernetes cluster"
}

# 2. Роли для Service Account
resource "yandex_resourcemanager_folder_iam_member" "k8s_clusters_agent" {
  folder_id = local.folder_id
  role      = "k8s.clusters.agent"
  member    = "serviceAccount:${yandex_iam_service_account.k8s_sa.id}"
}

resource "yandex_resourcemanager_folder_iam_member" "vpc_public_admin" {
  folder_id = local.folder_id
  role      = "vpc.publicAdmin"
  member    = "serviceAccount:${yandex_iam_service_account.k8s_sa.id}"
}

resource "yandex_resourcemanager_folder_iam_member" "images_puller" {
  folder_id = local.folder_id
  role      = "container-registry.images.puller"
  member    = "serviceAccount:${yandex_iam_service_account.k8s_sa.id}"
}

# 3. KMS Key для шифрования
resource "yandex_kms_symmetric_key" "kms_key" {
  name              = "k8s-cluster-key"
  default_algorithm = "AES_128"
  rotation_period   = "8760h"
}

resource "yandex_resourcemanager_folder_iam_member" "encrypter_decrypter" {
  folder_id = local.folder_id
  role      = "kms.keys.encrypterDecrypter"
  member    = "serviceAccount:${yandex_iam_service_account.k8s_sa.id}"
}

# 4. Сеть (используем существующую default)
data "yandex_vpc_network" "default" {
  name = "default"
}

# 5. Подсеть (уже создана: e9b9vc6avhf0b990ohn7)
resource "yandex_vpc_subnet" "k8s_subnet" {
  name           = "k8s-subnet"
  zone           = "ru-central1-a"
  network_id     = data.yandex_vpc_network.default.id
  v4_cidr_blocks = ["192.168.100.0/24"]
}

# 6. НОВАЯ Security Group в правильной сети (default)
resource "yandex_vpc_security_group" "k8s_sg_default" {
  name        = "k8s-sg-default"
  description = "Security group for Kubernetes cluster in default network"
  network_id  = data.yandex_vpc_network.default.id

  # Минимальные обязательные правила
  ingress {
    protocol          = "TCP"
    description       = "Load balancer health checks"
    predefined_target = "loadbalancer_healthchecks"
    from_port         = 0
    to_port           = 65535
  }

  ingress {
    protocol          = "ANY"
    description       = "Master-node and node-node communication"
    predefined_target = "self_security_group"
    from_port         = 0
    to_port           = 65535
  }

  ingress {
    protocol          = "ANY"
    description       = "Pod-pod and service-service communication"
    v4_cidr_blocks    = [yandex_vpc_subnet.k8s_subnet.v4_cidr_blocks[0]]
    from_port         = 0
    to_port           = 65535
  }

  ingress {
    description    = "Kubernetes API"
    protocol       = "TCP"
    port           = 6443
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description    = "SSH"
    protocol       = "TCP"
    port           = 22
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description    = "All outgoing traffic"
    protocol       = "ANY"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
}

# 7. Kubernetes Cluster (еще не создан - остановите создание!)
resource "yandex_kubernetes_cluster" "main" {
  name        = "credit-scoring-cluster"
  description = "Kubernetes cluster for credit scoring system"
  network_id  = data.yandex_vpc_network.default.id

  cluster_ipv4_range       = "10.10.0.0/16"
  service_ipv4_range       = "10.11.0.0/16"
  node_ipv4_cidr_mask_size = 24

  master {
    version = "1.31"
    public_ip = true
    # Важно: используем НОВУЮ security group в правильной сети
    security_group_ids = [yandex_vpc_security_group.k8s_sg_default.id]

    master_location {
      zone      = "ru-central1-a"
      subnet_id = yandex_vpc_subnet.k8s_subnet.id
    }
  }

  service_account_id      = yandex_iam_service_account.k8s_sa.id
  node_service_account_id = yandex_iam_service_account.k8s_sa.id

  labels = {
    project     = "credit-scoring"
    environment = "dev"
  }

  release_channel = "REGULAR"

  # KMS провайдер для шифрования
  kms_provider {
    key_id = yandex_kms_symmetric_key.kms_key.id
  }

  # Зависимости
  depends_on = [
    yandex_resourcemanager_folder_iam_member.k8s_clusters_agent,
    yandex_resourcemanager_folder_iam_member.vpc_public_admin,
    yandex_resourcemanager_folder_iam_member.images_puller,
    yandex_resourcemanager_folder_iam_member.encrypter_decrypter
  ]
}

# 8. Node Group (будет создана после кластера)
resource "yandex_kubernetes_node_group" "workers" {
  cluster_id = yandex_kubernetes_cluster.main.id
  name       = "worker-nodes"
  version    = "1.31"

  instance_template {
    platform_id = "standard-v2"

    resources {
      cores         = 2
      memory        = 2
      core_fraction = 100
    }

    boot_disk {
      type = "network-hdd"
      size = 30
    }

    network_interface {
      subnet_ids = [yandex_vpc_subnet.k8s_subnet.id]
      nat        = true
    }
  }

  scale_policy {
    fixed_scale {
      size = 1
    }
  }

  allocation_policy {
    location {
      zone = "ru-central1-a"
    }
  }

  maintenance_policy {
    auto_upgrade = true
    auto_repair  = true
  }
}

# 9. Container Registry (уже создан: crplfl2mkk2krsnl7vnv)
resource "yandex_container_registry" "main" {
  name = "credit-scoring-registry"
}