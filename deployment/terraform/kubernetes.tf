# kubernetes.tf - полная версия со всеми правами

# Managed Kubernetes cluster

# Create service account for Kubernetes
resource "yandex_iam_service_account" "k8s_sa" {
  name        = "k8s-credit-scoring-sa"
  description = "Service account for Kubernetes cluster"
  folder_id   = var.yandex_folder_id
}

# ==============================================
# ВСЕ НЕОБХОДИМЫЕ ПРАВА ДЛЯ SERVICE ACCOUNT
# ==============================================

# 1. БАЗОВЫЕ ПРАВА (ОБЯЗАТЕЛЬНО)
resource "yandex_resourcemanager_folder_iam_member" "editor" {
  folder_id = var.yandex_folder_id
  role      = "editor"
  member    = "serviceAccount:${yandex_iam_service_account.k8s_sa.id}"
}

# 2. ПРАВА ДЛЯ KUBERNETES КЛАСТЕРА
resource "yandex_resourcemanager_folder_iam_member" "k8s_admin" {
  folder_id = var.yandex_folder_id
  role      = "k8s.admin"
  member    = "serviceAccount:${yandex_iam_service_account.k8s_sa.id}"
}

# 3. ПРАВА ДЛЯ ВЫЧИСЛИТЕЛЬНЫХ РЕСУРСОВ (НОДЫ)
resource "yandex_resourcemanager_folder_iam_member" "compute_admin" {
  folder_id = var.yandex_folder_id
  role      = "compute.admin"
  member    = "serviceAccount:${yandex_iam_service_account.k8s_sa.id}"
}

# 4. ПРАВА ДЛЯ VPC (СЕТЬ)
resource "yandex_resourcemanager_folder_iam_member" "vpc_user" {
  folder_id = var.yandex_folder_id
  role      = "vpc.user"
  member    = "serviceAccount:${yandex_iam_service_account.k8s_sa.id}"
}

resource "yandex_resourcemanager_folder_iam_member" "vpc_public_admin" {
  folder_id = var.yandex_folder_id
  role      = "vpc.publicAdmin"
  member    = "serviceAccount:${yandex_iam_service_account.k8s_sa.id}"
}

# 5. ПРАВА ДЛЯ CONTAINER REGISTRY
resource "yandex_resourcemanager_folder_iam_member" "images_puller" {
  folder_id = var.yandex_folder_id
  role      = "container-registry.images.puller"
  member    = "serviceAccount:${yandex_iam_service_account.k8s_sa.id}"
}

# 6. ПРАВА ДЛЯ STORAGE (ЕСЛИ НУЖНЫ BUCKET'Ы)
resource "yandex_resourcemanager_folder_iam_member" "storage_admin" {
  folder_id = var.yandex_folder_id
  role      = "storage.admin"
  member    = "serviceAccount:${yandex_iam_service_account.k8s_sa.id}"
}

# ==============================================
# ОСТАЛЬНЫЕ РЕСУРСЫ
# ==============================================

# Create static key for service account
resource "yandex_iam_service_account_static_access_key" "k8s_sa_key" {
  service_account_id = yandex_iam_service_account.k8s_sa.id
  description        = "Static key for Kubernetes SA"
}

# KMS key for encrypting secrets in Kubernetes
resource "yandex_kms_symmetric_key" "k8s_secrets_key" {
  name              = "k8s-secrets-key"
  description       = "KMS key for encrypting Kubernetes secrets"
  default_algorithm = "AES_256"
  rotation_period   = "8760h"  # 1 year

  labels = {
    environment = var.environment
  }
}

# Create Managed Kubernetes cluster
resource "yandex_kubernetes_cluster" "credit_scoring_k8s" {
  name               = "${var.cluster_name}-${var.environment}-2"
  description        = "Kubernetes cluster for credit scoring (${var.environment})"
  network_id         = var.use_existing_network ? data.yandex_vpc_network.existing[0].id : yandex_vpc_network.credit_scoring_network[0].id
  release_channel = "STABLE"
  cluster_ipv4_range = "172.20.0.0/16"
  service_ipv4_range = "172.21.0.0/16"

  # ДОБАВЬТЕ ЭТОТ БЛОК:
  timeouts {
    create = "45m"  # 30 минут на создание
    update = "30m"  # 15 минут на обновление
    delete = "30m"  # 15 минут на удаление
  }

  master {
    version   = var.k8s_version
    public_ip = true

    master_location {
      zone      = yandex_vpc_subnet.subnet_a.zone
      subnet_id = yandex_vpc_subnet.subnet_a.id
    }

    security_group_ids = [
      yandex_vpc_security_group.k8s_master.id
    ]
  }

  service_account_id      = yandex_iam_service_account.k8s_sa.id
  node_service_account_id = yandex_iam_service_account.k8s_sa.id

  kms_provider {
    key_id = yandex_kms_symmetric_key.k8s_secrets_key.id
  }

  labels = {
    environment = var.environment
    project     = "credit-scoring"
  }

  # ВАЖНО: Зависимости от ВСЕХ IAM permissions
  depends_on = [
    yandex_resourcemanager_folder_iam_member.editor,
    yandex_resourcemanager_folder_iam_member.k8s_admin,
    yandex_resourcemanager_folder_iam_member.compute_admin,
    yandex_resourcemanager_folder_iam_member.vpc_user,
    yandex_resourcemanager_folder_iam_member.vpc_public_admin,
    yandex_resourcemanager_folder_iam_member.images_puller,
    yandex_resourcemanager_folder_iam_member.storage_admin,
    yandex_vpc_subnet.subnet_a,
    yandex_vpc_subnet.subnet_b,
    yandex_vpc_subnet.subnet_d
  ]
}

# Create node group for CPU workload
resource "yandex_kubernetes_node_group" "cpu_nodes" {
  cluster_id  = yandex_kubernetes_cluster.credit_scoring_k8s.id
  name        = "cpu-node-group"
  description = "CPU nodes for credit scoring"

  instance_template {
    platform_id = "standard-v2"

    resources {
      memory = var.node_memory * 1024  # Convert GB to MB
      cores  = var.node_cores
    }

    boot_disk {
      type = "network-hdd"
      size = var.node_disk_size
    }

    scheduling_policy {
      preemptible = var.preemptible_nodes
    }

    network_interface {
      subnet_ids = [yandex_vpc_subnet.subnet_a.id]
      security_group_ids = [
        yandex_vpc_security_group.k8s_nodes.id,
        yandex_vpc_security_group.application.id
      ]
    }

    container_runtime {
      type = "containerd"
    }

    metadata = {
      ssh-keys = "ubuntu:${file("~/.ssh/id_rsa.pub")}"
    }
  }

  scale_policy {
    fixed_scale {
      size = var.min_node_count
    }
  }

  allocation_policy {
    location {
      zone = yandex_vpc_subnet.subnet_a.zone
    }
  }

  maintenance_policy {
    auto_upgrade = true
    auto_repair  = true

    maintenance_window {
      day        = "saturday"
      start_time = "22:00"
      duration   = "3h"
    }
  }

  labels = {
    environment = var.environment
    node-type   = "cpu"
  }
}