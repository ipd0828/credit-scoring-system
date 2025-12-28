# Dockerfile.ml - Для ML пайплайна с DVC
FROM python:3.12-slim as builder

# Установка системных зависимостей для ML
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .

# Установка зависимостей в виртуальное окружение
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# --- Production stage ---
FROM python:3.12-slim

# Установка DVC с S3 поддержкой
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копирование виртуального окружения из builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Создание пользователя приложения
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Копирование кода
COPY . .

# Настройка DVC
RUN mkdir -p /app/.dvc && \
    chown -R appuser:appuser /app

USER appuser

# Команда по умолчанию
CMD ["python", "scripts/run_pipeline.py --use-custom-features"]