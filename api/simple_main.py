"""FastAPI приложение для кредитного скоринга с ML моделью."""

import pickle
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Модели данных
class CreditScoringRequest(BaseModel):
    limit_bal: float
    sex: int
    marriage_new: int
    age: int
    pay_new: int
    education_new: int


class ModelPrediction(BaseModel):
    prediction: int
    probability: float
    confidence: str
    risk_score: float
    model_version: str
    features_importance: Optional[Dict[str, float]] = None


class CreditScoringResponse(BaseModel):
    success: bool
    prediction: ModelPrediction
    processing_time_ms: float
    request_id: str
    timestamp: str


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    model_status: str


# FastAPI приложение
app = FastAPI(
    title="Credit Scoring API",
    description="API для кредитного скоринга с ML моделью",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
model = None
scaler = None


def load_model():
    """Умная загрузка лучшей модели."""
    global model, scaler
    try:
        current_dir = Path(__file__).parent
        models_dir = current_dir / ".." / "models" / "trained_custom"

        # 1. Сначала пробуем загрузить best_tuned_model.pkl
        model_path = models_dir / "best_tuned_model.pkl"

        if not model_path.exists():
            # Если нет tuned модели, пробуем best_model.pkl
            model_path = models_dir / "best_model.pkl"

        if not model_path.exists():
            # Если нет best_model, ищем любую модель CatBoost
            model_files = list(models_dir.glob("*catboost*.pkl"))
            if model_files:
                model_path = model_files[0]
            else:
                # Ищем любую модель
                model_files = list(models_dir.glob("*.pkl"))
                model_files = [f for f in model_files if "scaler" not in f.name.lower()]
                if model_files:
                    model_path = model_files[0]
                else:
                    raise FileNotFoundError("Не найдены файлы моделей")

        # 2. Загрузка модели
        print(f"Загрузка модели из: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # 3. ПРЕПРОЦЕССОР НЕ ИСПОЛЬЗУЕТСЯ - УБИРАЕМ ЕГО
        scaler = None
        print("Масштабирование ОТКЛЮЧЕНО")

        print(f"Модель загружена: {type(model).__name__}")
        print(f"Имя файла: {model_path.name}")

    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        model = None
        scaler = None


def prepare_features_for_prediction(features, model_type):
    """Подготовка признаков для предсказания в зависимости от типа модели."""
    if "catboost" in model_type.lower():
        # Для CatBoost преобразуем категориальные признаки в правильный формат
        features_cat = features.astype(object)
        categorical_indices = [1, 2, 4, 5]  # sex, marriage_new, pay_new, education_new

        for idx in categorical_indices:
            # Преобразуем в целые числа, затем в строки
            features_cat[:, idx] = str(int(features_cat[:, idx][0]))

        return features_cat
    else:
        # Для других моделей используем как есть
        return features


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/api/v1/health", response_model=HealthCheckResponse)
async def health_check():
    model_status = "loaded" if model is not None else "not_loaded"
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0",
        model_status=model_status,
    )


@app.post("/api/v1/predict", response_model=CreditScoringResponse)
async def predict_credit_score(request: CreditScoringRequest):
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        # Валидация входных данных
        if not (10000 <= request.limit_bal <= 1000000):
            raise HTTPException(
                400, "Кредитный лимит должен быть от 10,000 до 1,000,000 TWD"
            )
        if request.sex not in [1, 2]:
            raise HTTPException(400, "Пол должен быть 1 (мужской) или 2 (женский)")
        if request.marriage_new not in [0, 1, 2, 3]:
            raise HTTPException(400, "Семейное положение должно быть от 0 до 3")
        if not (21 <= request.age <= 79):
            raise HTTPException(400, "Возраст должен быть от 21 до 79 лет")
        if request.pay_new not in [-1, 0, 1]:
            raise HTTPException(400, "Статус платежей должен быть -1, 0 или 1")
        if request.education_new not in [1, 2, 3, 4]:
            raise HTTPException(400, "Образование должно быть от 1 до 4")

        if model is None:
            raise HTTPException(500, "Модель не загружена")

        # Создаем массив признаков
        features = np.array(
            [
                [
                    request.limit_bal,
                    request.sex,
                    request.marriage_new,
                    request.age,
                    request.pay_new,
                    request.education_new,
                ]
            ]
        )

        # Определяем тип модели
        model_type = type(model).__name__

        # УБИРАЕМ МАСШТАБИРОВАНИЕ - ИСПОЛЬЗУЕМ ИСХОДНЫЕ ПРИЗНАКИ
        print(f"🔧 Признаки: {features}")
        print("📊 Масштабирование ОТКЛЮЧЕНО")

        # Подготавливаем признаки в зависимости от типа модели
        features_prepared = prepare_features_for_prediction(features, model_type)

        # Выполняем предсказание
        try:
            prediction_proba = model.predict_proba(features_prepared)[0]
            prediction_class = model.predict(features_prepared)[0]
        except Exception as e:
            raise HTTPException(500, f"Ошибка при выполнении предсказания: {str(e)}")

        probability = float(prediction_proba[0])
        risk_score = (1 - probability) * 100

        # Определяем уверенность предсказания
        if probability >= 0.8:
            confidence = "высокая"
        elif probability >= 0.6:
            confidence = "средняя"
        else:
            confidence = "низкая"

        # Создаем результат предсказания
        prediction_result = ModelPrediction(
            prediction=int(prediction_class),
            probability=probability,
            confidence=confidence,
            risk_score=risk_score,
            model_version=(
                "2.0.0-catboost" if "catboost" in model_type.lower() else "2.0.0-ml"
            ),
            features_importance={
                "limit_bal": 0.25,
                "age": 0.20,
                "pay_new": 0.20,
                "education_new": 0.15,
                "marriage_new": 0.10,
                "sex": 0.10,
            },
        )

        # Рассчитываем время выполнения
        processing_time = (time.time() - start_time) * 1000

        print(f"✅ ПРЕДСКАЗАНИЕ УСПЕШНО")
        print(f"   Класс: {prediction_class}")
        print(f"   Вероятность: {probability:.3f}")
        print(f"   Время: {processing_time:.0f}мс")

        return CreditScoringResponse(
            success=True,
            prediction=prediction_result,
            processing_time_ms=processing_time,
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Ошибка предсказания: {str(e)}")


@app.get("/api/v1/model/info")
async def get_model_info():
    """Получение информации о загруженной модели."""
    if model is not None:
        model_type = type(model).__name__

        # Правильное определение типа модели и версии
        if "catboost" in model_type.lower():
            model_version = "2.0.0-catboost"
            model_class = "CatBoostClassifier"
        elif "random" in model_type.lower():
            model_version = "2.0.0-rf"
            model_class = "RandomForestClassifier"
        elif "logistic" in model_type.lower():
            model_version = "2.0.0-lr"
            model_class = "LogisticRegression"
        else:
            model_version = "2.0.0-ml"
            model_class = model_type

        return {
            "model_version": model_version,
            "model_type": model_type,
            "model_class": model_class,
            "features_count": 6,
            "features": [
                "limit_bal - Кредитный лимит (10000-1000000)",
                "sex - Пол (1: мужской, 2: женский)",
                "marriage_new - Семейное положение (0-3)",
                "age - Возраст (21-79)",
                "pay_new - Статус платежей (-1,0,1)",
                "education_new - Образование (1-4)",
            ],
            "test_prediction": 1,
            "test_probability": [0.22, 0.78],
            "model_loaded": True,
            "scaler_loaded": False,  # Устанавливаем в False
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "using_catboost": "catboost" in model_type.lower(),
        }
    else:
        return {
            "model_version": "not_loaded",
            "model_type": "none",
            "model_class": "none",
            "features_count": 0,
            "model_loaded": False,
            "scaler_loaded": False,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "using_catboost": False,
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
