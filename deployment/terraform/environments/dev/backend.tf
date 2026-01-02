# Конфигурация remote state (опционально, можно использовать локальный)
# terraform {
#   backend "s3" {
#     endpoint   = "storage.yandexcloud.net"
#     bucket     = "credit-scoring-terraform-state"
#     region     = "ru-central1"
#     key        = "dev/terraform.tfstate"
#     access_key = "YOUR_ACCESS_KEY"
#     secret_key = "YOUR_SECRET_KEY"
#
#     skip_region_validation      = true
#     skip_credentials_validation = true
#     skip_metadata_api_check     = true
#   }
# }