# Скрипт развертывания в Minikube
Write-Host "Starting Kubernetes deployment..." -ForegroundColor Green

# Шаг 1: Загружаем образы в Minikube
Write-Host "Loading Docker images to Minikube..." -ForegroundColor Cyan
minikube image load credit-scoring-api:test
minikube image load credit-scoring-frontend:test

# Шаг 2: Развертываем в Kubernetes
Write-Host "pplying Kubernetes manifests..." -ForegroundColor Cyan
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml

# Шаг 3: Ждем запуска
Write-Host "Waiting for pods to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Шаг 4: Проверяем статус
Write-Host "`Pods status:" -ForegroundColor Green
kubectl get pods -n credit-scoring

Write-Host "`Services:" -ForegroundColor Green
kubectl get svc -n credit-scoring

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "✅ Deployment complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`n🌐 How to access:" -ForegroundColor White
Write-Host "1. API: kubectl port-forward -n credit-scoring service/credit-scoring-api 8000:8000" -ForegroundColor Gray
Write-Host "2. Frontend: kubectl port-forward -n credit-scoring service/credit-scoring-frontend 8501:8501" -ForegroundColor Gray
Write-Host "`n📊 Dashboard: minikube dashboard" -ForegroundColor Gray
Write-Host "📝 API Docs: http://localhost:8000/docs (after port-forward)" -ForegroundColor Gray
Write-Host "🎨 Frontend: http://localhost:8501 (after port-forward)" -ForegroundColor Gray