# Развертывание инфраструктуры кредитного скоринга в Yandex Cloud

## Предварительные требования

1. **Установите инструменты:**
   ```bash
   # Terraform
   wget https://releases.hashicorp.com/terraform/1.5.0/terraform_1.5.0_linux_amd64.zip
   unzip terraform_1.5.0_linux_amd64.zip
   sudo mv terraform /usr/local/bin/
   
   # Yandex Cloud CLI
   curl -sSL https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash
   yc init