# Dockerfile.api - Многостадийный для FastAPI
FROM python:3.12-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# --- Production stage ---
FROM python:3.12-slim as production

WORKDIR /app

# Копирование зависимостей из builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Копирование кода
COPY api/ ./api/
COPY models/ ./models/

# Создание пользователя
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health', timeout=2)"

EXPOSE 8000

CMD ["uvicorn", "api.credit_scoring_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
