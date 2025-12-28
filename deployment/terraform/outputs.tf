# outputs.tf - временная упрощенная версия

output "kubernetes_cluster_id" {
  description = "Kubernetes cluster ID"
  value       = try(yandex_kubernetes_cluster.credit_scoring_k8s.id, "not created")
}

output "kubernetes_cluster_name" {
  description = "Kubernetes cluster name"
  value       = try(yandex_kubernetes_cluster.credit_scoring_k8s.name, "not created")
}

output "kubernetes_cluster_endpoint" {
  description = "Kubernetes cluster endpoint"
  value       = try(yandex_kubernetes_cluster.credit_scoring_k8s.master[0].external_v4_endpoint, "not created")
  sensitive   = true
}

output "service_account_id" {
  description = "Service account ID for Kubernetes"
  value       = try(yandex_iam_service_account.k8s_sa.id, "not created")
  sensitive   = true
}

output "network_subnets" {
  description = "Created subnets"
  value = {
    subnet_a = try(yandex_vpc_subnet.subnet_a.v4_cidr_blocks[0], "not created")
    subnet_b = try(yandex_vpc_subnet.subnet_b.v4_cidr_blocks[0], "not created")
    subnet_d = try(yandex_vpc_subnet.subnet_d.v4_cidr_blocks[0], "not created")
  }
}

# Временно комментируем вывод bucket'ов пока они не созданы
# output "storage_buckets" {
#   description = "Created storage buckets"
#   value = {
#     terraform_state = try(yandex_storage_bucket.terraform_state.bucket, "not created")
#     ml_models       = try(yandex_storage_bucket.ml_models.bucket, "not created")
#   }
# }