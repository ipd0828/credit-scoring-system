"""
Unit тесты для API (api/simple_main.py).

Схема запроса соответствует текущей реализации api/simple_main.py:
признаки, доступные при подаче заявки (limit_bal, sex, marriage_new,
age, pay_new, education_new).
"""

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent.parent))

from api import simple_main
from api.simple_main import app


def _make_model_mock(probability: float = 0.9, prediction_class: int = 1):
    """Создает мок модели с предсказанием и вероятностью."""
    model = MagicMock()
    model.predict.return_value = [prediction_class]
    model.predict_proba.return_value = [[1.0 - probability, probability]]
    return model


def _valid_request() -> dict:
    """Возвращает валидные входные данные для predict endpoint."""
    return {
        "limit_bal": 200000,
        "sex": 2,
        "marriage_new": 1,
        "age": 35,
        "pay_new": 0,
        "education_new": 2,
    }


class TestAPI:
    """Тесты для API endpoints."""

    @pytest.fixture
    def client(self):
        """Создает тестовый клиент FastAPI."""
        return TestClient(app)

    def test_health_check(self, client):
        """Тест health check endpoint."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
        assert "model_status" in data

    def test_predict_endpoint_valid_request(self, client):
        """Тест predict endpoint с валидным запросом."""
        with patch.object(simple_main, "model", _make_model_mock()):
            response = client.post("/api/v1/predict", json=_valid_request())

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "prediction" in data
        assert "processing_time_ms" in data
        assert "request_id" in data
        assert "timestamp" in data

        # Проверяем структуру prediction
        prediction = data["prediction"]
        assert "prediction" in prediction
        assert "probability" in prediction
        assert "confidence" in prediction
        assert "risk_score" in prediction
        assert "model_version" in prediction
        assert "features_importance" in prediction

        # Проверяем типы данных
        assert isinstance(prediction["prediction"], int)
        assert isinstance(prediction["probability"], float)
        assert isinstance(prediction["risk_score"], float)
        assert isinstance(prediction["features_importance"], dict)

    def test_predict_endpoint_high_risk(self, client):
        """Тест predict endpoint с высоким риском (низкая вероятность)."""
        with patch.object(simple_main, "model", _make_model_mock(probability=0.9)):
            response = client.post("/api/v1/predict", json=_valid_request())

        assert response.status_code == 200
        data = response.json()
        prediction = data["prediction"]
        # risk_score = (1 - probability) * 100 -> низкая вероятность, высокий риск
        assert prediction["risk_score"] > 50
        assert prediction["confidence"] == "низкая"

    def test_predict_endpoint_low_risk(self, client):
        """Тест predict endpoint с низким риском (высокая вероятность)."""
        with patch.object(simple_main, "model", _make_model_mock(probability=0.1)):
            response = client.post("/api/v1/predict", json=_valid_request())

        assert response.status_code == 200
        data = response.json()
        prediction = data["prediction"]
        # risk_score = (1 - probability) * 100 -> высокая вероятность, низкий риск
        assert prediction["risk_score"] < 50
        assert prediction["confidence"] == "высокая"

    def test_predict_endpoint_invalid_data(self, client):
        """Тест predict endpoint с невалидными данными (лимит вне диапазона)."""
        with patch.object(simple_main, "model", _make_model_mock()):
            request_data = _valid_request()
            request_data["limit_bal"] = 100  # Вне диапазона 10000-1000000

            response = client.post("/api/v1/predict", json=request_data)

        assert response.status_code == 400

    def test_predict_endpoint_wrong_types(self, client):
        """Тест predict endpoint с невалидными значениями полей."""
        with patch.object(simple_main, "model", _make_model_mock()):
            request_data = _valid_request()
            request_data["sex"] = 5  # Должно быть 1 или 2

            response = client.post("/api/v1/predict", json=request_data)

        assert response.status_code == 400

    def test_predict_endpoint_missing_field(self, client):
        """Тест predict endpoint с отсутствующим обязательным полем."""
        request_data = _valid_request()
        del request_data["age"]

        response = client.post("/api/v1/predict", json=request_data)

        assert response.status_code == 422

    def test_api_docs_endpoint(self, client):
        """Тест endpoint'ов документации."""
        response = client.get("/docs")
        assert response.status_code == 200

        response = client.get("/redoc")
        assert response.status_code == 200

    def test_response_time(self, client):
        """Тест времени ответа predict endpoint."""
        with patch.object(simple_main, "model", _make_model_mock()):
            start_time = time.time()
            response = client.post("/api/v1/predict", json=_valid_request())
            end_time = time.time()

        assert response.status_code == 200
        response_time = end_time - start_time

        # Проверяем, что ответ пришел быстро (менее 1 секунды)
        assert response_time < 1.0

        data = response.json()
        assert data["processing_time_ms"] < 1000  # Менее 1 секунды в миллисекундах


class TestAPIIntegration:
    """Интеграционные тесты для API."""

    @pytest.fixture
    def client(self):
        """Создает тестовый клиент FastAPI."""
        return TestClient(app)

    def test_multiple_requests(self, client):
        """Тест множественных запросов к API."""
        with patch.object(simple_main, "model", _make_model_mock()):
            for _ in range(5):
                response = client.post("/api/v1/predict", json=_valid_request())
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True

    def test_concurrent_requests(self, client):
        """Тест конкурентных запросов к API."""
        with patch.object(simple_main, "model", _make_model_mock()):
            results = []

            def make_request():
                response = client.post("/api/v1/predict", json=_valid_request())
                results.append(response.status_code)

            # Создаем несколько потоков
            threads = []
            for _ in range(3):
                thread = threading.Thread(target=make_request)
                threads.append(thread)
                thread.start()

            # Ждем завершения всех потоков
            for thread in threads:
                thread.join()

            # Проверяем, что все запросы успешны
            assert all(status == 200 for status in results)
            assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__])
