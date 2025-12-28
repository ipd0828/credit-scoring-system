Write-Host "🌐 Accessing Credit Scoring System..." -ForegroundColor Green

# Проверить все компоненты
Write-Host "📋 System status:" -ForegroundColor Cyan
kubectl get all -n credit-scoring

# Запустить port-forward для обоих сервисов
Write-Host "`n🔗 Starting port-forward for all services..." -ForegroundColor Yellow
Write-Host "   API:       http://localhost:8000" -ForegroundColor White
Write-Host "   API Docs:  http://localhost:8000/docs" -ForegroundColor White
Write-Host "   Frontend:  http://localhost:8501" -ForegroundColor White
Write-Host "`n   Press Ctrl+C to stop all" -ForegroundColor Gray

# Запустить оба port-forward в фоне
$apiJob = Start-Job -ScriptBlock {
    kubectl port-forward -n credit-scoring service/credit-scoring-api 8000:8000
}

$frontendJob = Start-Job -ScriptBlock {
    kubectl port-forward -n credit-scoring service/credit-scoring-frontend 8501:8501
}

# Показать статус
Write-Host "`n✅ Port-forward started. Jobs running:" -ForegroundColor Green
Get-Job

# Ждать команды остановки
Write-Host "`n⏳ Press Enter to stop..." -ForegroundColor Yellow
Read-Host

# Остановить
Stop-Job -Job $apiJob, $frontendJob
Remove-Job -Job $apiJob, $frontendJob
Write-Host "❌ Port-forward stopped." -ForegroundColor Red