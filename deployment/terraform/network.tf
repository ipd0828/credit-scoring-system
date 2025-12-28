# network.tf - исправленная версия с доступными зонами

# Network configuration
data "yandex_vpc_network" "existing" {
  count = var.use_existing_network ? 1 : 0
  name  = "default"
}

# Create new network if needed
resource "yandex_vpc_network" "credit_scoring_network" {
  count       = var.use_existing_network ? 0 : 1
  name        = "credit-scoring-${var.environment}"
  description = "Network for credit scoring (${var.environment})"
  labels = {
    environment = var.environment
    project     = "credit-scoring"
  }
}

# Subnets in available zones
resource "yandex_vpc_subnet" "subnet_a" {
  name           = "credit-scoring-subnet-a"
  zone           = "ru-central1-a"
  network_id     = var.use_existing_network ? data.yandex_vpc_network.existing[0].id : yandex_vpc_network.credit_scoring_network[0].id
  v4_cidr_blocks = ["10.0.1.0/24"]

  labels = {
    environment = var.environment
    zone        = "ru-central1-a"
  }
}

resource "yandex_vpc_subnet" "subnet_b" {
  name           = "credit-scoring-subnet-b"
  zone           = "ru-central1-b"
  network_id     = var.use_existing_network ? data.yandex_vpc_network.existing[0].id : yandex_vpc_network.credit_scoring_network[0].id
  v4_cidr_blocks = ["10.0.2.0/24"]

  labels = {
    environment = var.environment
    zone        = "ru-central1-b"
  }
}

resource "yandex_vpc_subnet" "subnet_d" {
  name           = "credit-scoring-subnet-d"
  zone           = "ru-central1-d"  # Используем доступную зону вместо c
  network_id     = var.use_existing_network ? data.yandex_vpc_network.existing[0].id : yandex_vpc_network.credit_scoring_network[0].id
  v4_cidr_blocks = ["10.0.3.0/24"]

  labels = {
    environment = var.environment
    zone        = "ru-central1-d"
  }
}

# Security Groups
resource "yandex_vpc_security_group" "k8s_master" {
  name        = "k8s-master-security-group"
  description = "Security group for Kubernetes master"
  network_id  = var.use_existing_network ? data.yandex_vpc_network.existing[0].id : yandex_vpc_network.credit_scoring_network[0].id

  ingress {
    protocol       = "TCP"
    description    = "SSH for administration"
    port           = 22
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    protocol       = "TCP"
    description    = "Kubernetes API"
    port           = 6443
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    protocol       = "ANY"
    description    = "All outgoing traffic"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  labels = {
    environment = var.environment
  }
}

resource "yandex_vpc_security_group" "k8s_nodes" {
  name        = "k8s-nodes-security-group"
  description = "Security group for Kubernetes worker nodes"
  network_id  = var.use_existing_network ? data.yandex_vpc_network.existing[0].id : yandex_vpc_network.credit_scoring_network[0].id

  ingress {
    protocol       = "ANY"
    description    = "All traffic between nodes"
    v4_cidr_blocks = [
      yandex_vpc_subnet.subnet_a.v4_cidr_blocks[0],
      yandex_vpc_subnet.subnet_b.v4_cidr_blocks[0],
      yandex_vpc_subnet.subnet_d.v4_cidr_blocks[0]
    ]
  }

  ingress {
    protocol       = "TCP"
    description    = "NodePort range"
    from_port      = 30000
    to_port        = 32767
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    protocol       = "ANY"
    description    = "All outgoing traffic"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  labels = {
    environment = var.environment
  }
}

resource "yandex_vpc_security_group" "application" {
  name        = "application-security-group"
  description = "Security group for application"
  network_id  = var.use_existing_network ? data.yandex_vpc_network.existing[0].id : yandex_vpc_network.credit_scoring_network[0].id

  ingress {
    protocol       = "TCP"
    description    = "HTTP for application"
    port           = 80
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    protocol       = "TCP"
    description    = "HTTPS for application"
    port           = 443
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    protocol       = "TCP"
    description    = "FastAPI application"
    port           = 8000
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    protocol       = "TCP"
    description    = "Streamlit frontend"
    port           = 8501
    v4_cidr_blocks = ["0.0.0.0/0"]
  }

  labels = {
    environment = var.environment
  }
}