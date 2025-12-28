# main.tf
data "yandex_client_config" "client" {}

resource "yandex_container_registry" "credit_scoring_registry" {
  name      = "credit-scoring-registry"
  folder_id = var.yandex_folder_id
}