# Output values
output "kubernetes_cluster_name" {
  description = "Kubernetes cluster name"
  value       = yandex_kubernetes_cluster.main.name
}

output "kubernetes_cluster_endpoint" {
  description = "Kubernetes cluster endpoint"
  value       = yandex_kubernetes_cluster.main.master[0].external_v4_endpoint
  sensitive   = true
}

output "container_registry_id" {
  description = "Container registry ID"
  value       = yandex_container_registry.main.id
}

output "vpc_network_id" {
  description = "VPC network ID"
  value       = data.yandex_vpc_network.default.id
}

output "vpc_subnet_id" {
  description = "VPC subnet ID"
  value       = yandex_vpc_subnet.k8s_subnet.id
}

output "service_account_id" {
  description = "Service Account ID"
  value       = yandex_iam_service_account.k8s_sa.id
}

output "kms_key_id" {
  description = "KMS Key ID"
  value       = yandex_kms_symmetric_key.kms_key.id
}