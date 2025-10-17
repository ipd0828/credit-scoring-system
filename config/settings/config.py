"""Настройки конфигурации приложения."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения."""
    
    # Конфигурация API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Конфигурация базы данных
    database_url: str = Field(env="DATABASE_URL")
    database_host: str = Field(default="localhost", env="DATABASE_HOST")
    database_port: int = Field(default=5432, env="DATABASE_PORT")
    database_name: str = Field(default="credit_scoring_db", env="DATABASE_NAME")
    database_user: str = Field(default="user", env="DATABASE_USER")
    database_password: str = Field(default="password", env="DATABASE_PASSWORD")
    
    # Конфигурация модели
    model_path: str = Field(default="models/trained/credit_scoring_model.pkl", env="MODEL_PATH")
    model_version: str = Field(default="1.0.0", env="MODEL_VERSION")
    model_threshold: float = Field(default=0.5, env="MODEL_THRESHOLD")
    
    # Безопасность
    secret_key: str = Field(env="SECRET_KEY")
    jwt_secret_key: str = Field(env="JWT_SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Мониторинг
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    grafana_port: int = Field(default=3000, env="GRAFANA_PORT")
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    
    # Внешние API
    external_credit_api_url: Optional[str] = Field(default=None, env="EXTERNAL_CREDIT_API_URL")
    external_credit_api_key: Optional[str] = Field(default=None, env="EXTERNAL_CREDIT_API_KEY")
    
    # Логирование
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Флаги функций
    enable_model_monitoring: bool = Field(default=True, env="ENABLE_MODEL_MONITORING")
    enable_automatic_retraining: bool = Field(default=False, env="ENABLE_AUTOMATIC_RETRAINING")
    enable_a_b_testing: bool = Field(default=False, env="ENABLE_A_B_TESTING")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }


# Глобальный экземпляр настроек
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Получить настройки приложения."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
