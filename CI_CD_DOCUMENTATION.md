# CI/CD Документация для проекта кредитного скоринга

## 🚀 Обзор

Проект кредитного скоринга включает полную автоматизацию CI/CD пайплайна с использованием GitHub Actions, Docker, MLflow и системы мониторинга.

## 📁 Структура CI/CD

```
.github/workflows/          # GitHub Actions workflows
├── ci-cd.yml              # Основной CI/CD пайплайн

scripts/deployment/         # Скрипты развертывания
├── docker_build.sh        # Сборка Docker образов (Linux/Mac)
├── docker_build.bat       # Сборка Docker образов (Windows)
├── docker_run.sh          # Запуск в Docker (Linux/Mac)
└── docker_run.bat         # Запуск в Docker (Windows)

scripts/monitoring/         # Скрипты мониторинга
├── model_monitoring.py    # Мониторинг моделей
└── data_quality_monitor.py # Мониторинг качества данных

scripts/model_training/     # ML скрипты с MLflow
├── mlflow_tracking.py     # Интеграция с MLflow
├── train_models.py        # Обучение моделей
├── hyperparameter_tuning.py # Подбор параметров
└── validation.py          # Валидация моделей

.pre-commit-config.yaml    # Pre-commit hooks
scripts/setup_pre_commit.py # Настройка pre-commit
```

## 🔄 CI/CD Пайплайн

### 1. **Code Quality** (Качество кода)
- **Black** - форматирование Python кода
- **isort** - сортировка импортов
- **flake8** - линтинг кода
- **mypy** - проверка типов
- **bandit** - проверка безопасности

### 2. **Testing** (Тестирование)
- Unit тесты с pytest
- Покрытие кода с coverage
- Интеграционные тесты

### 3. **Data Validation** (Валидация данных)
- Проверка наличия данных
- Валидация структуры данных
- Проверка качества данных

### 4. **Model Training** (Обучение моделей)
- Автоматическое обучение при push в main
- Еженедельное переобучение по расписанию
- Интеграция с MLflow для отслеживания экспериментов

### 5. **Deployment** (Развертывание)
- Развертывание в staging
- Smoke тесты
- Развертывание в production

### 6. **Monitoring** (Мониторинг)
- Мониторинг качества модели
- Мониторинг дрифта данных
- Алерты при деградации

## 🐳 Docker Контейнеризация

### Сборка образа
```bash
# Linux/Mac
./scripts/deployment/docker_build.sh

# Windows
scripts\deployment\docker_build.bat

# С тестированием
./scripts/deployment/docker_build.sh latest Dockerfile.ml --test
```

### Запуск пайплайна
```bash
# Linux/Mac
./scripts/deployment/docker_run.sh

# Windows
scripts\deployment\docker_run.bat

# С параметрами
./scripts/deployment/docker_run.sh --data-dir ./data --models-dir ./models --steps eda,preprocessing
```

### Docker Compose
```yaml
# docker-compose.yml уже настроен для полного стека
docker-compose up -d
```

## 📊 MLflow Интеграция

### Настройка MLflow
```python
from scripts.model_training.mlflow_tracking import setup_mlflow_experiment

# Создание эксперимента
tracker = setup_mlflow_experiment("credit-scoring")

# Логирование эксперимента
with tracker.start_run(run_name="my_experiment") as run:
    tracker.log_metrics({"accuracy": 0.95})
    tracker.log_model(model, "my_model")
```

### MLflow UI
```bash
# Запуск MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Доступ: http://localhost:5000
```

## 🔧 Pre-commit Hooks

### Установка
```bash
# Автоматическая установка
python scripts/setup_pre_commit.py

# Только установка без проверок
python scripts/setup_pre_commit.py --install-only

# Обновление hooks
python scripts/setup_pre_commit.py --update
```

### Ручной запуск
```bash
# Все файлы
pre-commit run --all-files

# Конкретный hook
pre-commit run black
pre-commit run flake8
```

## 📈 Мониторинг

### Мониторинг моделей
```bash
# Запуск мониторинга модели
python scripts/monitoring/model_monitoring.py

# Результаты сохраняются в monitoring/reports/
```

### Мониторинг качества данных
```bash
# Запуск мониторинга данных
python scripts/monitoring/data_quality_monitor.py

# Результаты сохраняются в monitoring/reports/
```

## 🚨 Алерты и уведомления

### Настройка алертов
1. **Email уведомления** - настройте в GitHub Actions secrets
2. **Slack уведомления** - добавьте webhook URL
3. **Telegram боты** - можно добавить через API

### Типы алертов
- **Data Drift** - дрифт данных
- **Performance Degradation** - деградация производительности
- **Prediction Bias** - смещение предсказаний
- **Data Quality Issues** - проблемы с качеством данных

## 🔄 Автоматизация

### Триггеры CI/CD
1. **Push в main** - полный пайплайн
2. **Pull Request** - проверки качества кода и тесты
3. **Еженедельно** - переобучение модели
4. **Ручной запуск** - через GitHub Actions UI

### Расписание
```yaml
# Еженедельное переобучение
schedule:
  - cron: '0 2 * * 1'  # Понедельник в 2:00 UTC
```

## 📋 Конфигурация

### Переменные окружения
```bash
# .env файл
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_REGISTRY_URI=sqlite:///mlflow.db
DATA_PATH=data/raw/accepted_2007_to_2018Q4.csv
MODEL_PATH=models/trained/best_model.pkl
```

### GitHub Secrets
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password
- `SLACK_WEBHOOK` - Slack webhook URL
- `EMAIL_PASSWORD` - Email password для уведомлений

## 🛠️ Разработка

### Локальная разработка
```bash
# Установка зависимостей
pip install -r requirements.txt

# Настройка pre-commit
python scripts/setup_pre_commit.py

# Запуск пайплайна локально
python scripts/run_pipeline.py

# Запуск в Docker
./scripts/deployment/docker_run.sh --interactive
```

### Тестирование
```bash
# Unit тесты
pytest tests/unit/

# Интеграционные тесты
pytest tests/integration/

# E2E тесты
pytest tests/e2e/

# С покрытием
pytest --cov=scripts --cov-report=html
```

## 📊 Метрики и KPI

### Качество кода
- Покрытие тестами > 80%
- Отсутствие критических уязвимостей
- Соответствие стандартам кодирования

### Качество данных
- Процент пропусков < 10%
- Количество выбросов < 5%
- Отсутствие дубликатов

### Качество модели
- ROC-AUC > 0.8
- F1-score > 0.7
- Отсутствие значительного дрифта

## 🔍 Troubleshooting

### Частые проблемы

1. **Docker не запускается**
   ```bash
   # Проверьте, что Docker запущен
   docker --version
   docker ps
   ```

2. **MLflow не работает**
   ```bash
   # Проверьте подключение
   mlflow ui --host 0.0.0.0 --port 5000
   ```

3. **Pre-commit не работает**
   ```bash
   # Переустановите hooks
   pre-commit uninstall
   pre-commit install
   ```

4. **Тесты не проходят**
   ```bash
   # Запустите с verbose
   pytest -v --tb=short
   ```

### Логи
- **CI/CD логи** - в GitHub Actions
- **Приложение логи** - в `logs/`
- **MLflow логи** - в MLflow UI
- **Docker логи** - `docker logs <container_name>`

## 📚 Полезные команды

### GitHub Actions
```bash
# Просмотр workflow
gh workflow list
gh workflow view ci-cd

# Запуск workflow
gh workflow run ci-cd
```

### Docker
```bash
# Список образов
docker images

# Список контейнеров
docker ps -a

# Очистка
docker system prune -a
```

### MLflow
```bash
# Список экспериментов
mlflow experiments list

# Скачать модель
mlflow models download -r <run_id> -d ./model
```

## 🎯 Следующие шаги

1. **Настройка production окружения**
2. **Интеграция с Kubernetes**
3. **Настройка мониторинга в реальном времени**
4. **A/B тестирование моделей**
5. **Автоматическое переобучение**

---

**Примечание**: Этот CI/CD пайплайн обеспечивает полную автоматизацию процесса разработки и развертывания ML моделей с соблюдением лучших практик DevOps и MLOps.
