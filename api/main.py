"""Основное FastAPI приложение."""

import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import structlog

from api.routes import credit_scoring, health, metrics
from config.settings import get_settings

# Настройка структурированного логирования
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Менеджер жизненного цикла приложения."""
    # Запуск
    logger.info("Запуск API кредитного скоринга")
    settings = get_settings()
    
    # Инициализация модели и других ресурсов
    # Здесь вы бы загрузили вашу ML модель
    logger.info(f"API запускается на {settings.api_host}:{settings.api_port}")
    
    yield
    
    # Остановка
    logger.info("Остановка API кредитного скоринга")


# Создание FastAPI приложения
app = FastAPI(
    title="API Кредитного Скоринга",
    description="API для кредитного скоринга и предсказания одобрения займов",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Добавление middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Настроить соответствующим образом для production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Настроить соответствующим образом для production
)


# Middleware для измерения времени обработки запросов
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Добавление времени обработки в заголовки ответа."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Глобальный обработчик исключений
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Глобальный обработчик исключений."""
    logger.error("Необработанное исключение", exc_info=exc, path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Внутренняя ошибка сервера",
            "message": "Произошла неожиданная ошибка"
        }
    )


# Подключение роутеров
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(credit_scoring.router, prefix="/api/v1", tags=["credit-scoring"])
app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])


@app.get("/")
async def root():
    """Корневой endpoint."""
    return {
        "message": "API Кредитного Скоринга",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug
    )
