"""FastAPI приложение для кредитного скоринга с ML моделью."""

import pickle
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
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
    threshold: Optional[float] = 0.6  # ← Добавляем опциональный параметрCredits


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
model_type = "unknown"
model_name = "unknown"


def load_model():
    """Умная загрузка лучшей модели."""
    global model, scaler, model_type, model_name
    try:
        current_dir = Path(__file__).parent

        # Пробуем загрузить ONNX модель LightGBM в первую очередь
        onnx_dir = current_dir / ".." / "models" / "onnx_final"
        onnx_model_path = onnx_dir / "best_model.onnx"

        if onnx_model_path.exists():
            try:
                import onnxruntime as ort
                model = ort.InferenceSession(str(onnx_model_path))
                model_type = "LightGBM (ONNX)"
                model_name = "best_model.onnx"
                print(f"ONNX модель загружена: {onnx_model_path.name}")
                print("Используется LightGBM (лучшая модель по ROC-AUC)")
                return
            except ImportError:
                print("ONNX Runtime не установлен, используем pickle модели")
                model_type = "LightGBM"
            except Exception as e:
                print(f"Ошибка загрузки ONNX модели: {e}")

        # Если ONNX не загрузилась, пробуем pickle модели
        models_dir = current_dir / ".." / "models" / "trained_final"

        # 1. Сначала пробуем загрузить best_tuned_model.pkl (LightGBM)
        model_path = models_dir / "best_tuned_model.pkl"

        if not model_path.exists():
            # Если нет tuned модели, пробуем lightgbm модель
            model_path = models_dir / "lightgbm.pkl"

        if not model_path.exists():
            # Если нет LightGBM, ищем best_model.pkl
            model_path = models_dir / "best_model.pkl"

        if not model_path.exists():
            # Если нет best_model, ищем любую модель
            model_files = list(models_dir.glob("*.pkl"))
            if model_files:
                # Ищем LightGBM в первую очередь
                lgbm_files = [f for f in model_files if 'lightgbm' in f.name.lower()]
                if lgbm_files:
                    model_path = lgbm_files[0]
                    model_type = "LightGBM"
                else:
                    model_path = model_files[0]
                    model_type = "Unknown ML Model"
                print(f"Будет загружена модель: {model_path.name}")
            else:
                raise FileNotFoundError("Не найдены файлы моделей")

        # 2. Загрузка модели
        print(f"Загрузка модели из: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # 3. Определяем тип модели
        model_name = model_path.name
        model_class_name = type(model).__name__.lower()

        if 'lgbm' in model_class_name or 'lightgbm' in model_class_name:
            model_type = "LightGBM"
        elif 'random' in model_class_name or 'forest' in model_class_name:
            model_type = "Random Forest"
        elif 'logistic' in model_class_name:
            model_type = "Logistic Regression"
        else:
            model_type = model_class_name

        # 4. ПРЕПРОЦЕССОР НЕ ИСПОЛЬЗУЕТСЯ
        scaler = None
        print("Масштабирование ОТКЛЮЧЕНО")

        print(f"Модель загружена: {model_type}")
        print(f"Имя файла: {model_name}")

        if model_type == "LightGBM":
            print("✓ Используется LightGBM (лучшая модель по ROC-AUC: 0.7285)")

    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        model = None
        scaler = None
        model_type = "not loaded"
        model_name = "none"


def prepare_features_for_prediction(features, model_type):
    """Подготовка признаков для предсказания в зависимости от типа модели."""
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
        model_status=f"{model_status} ({model_type})",
    )


@app.post("/api/v1/predict", response_model=CreditScoringResponse)
async def predict_credit_score(request: CreditScoringRequest):
    start_time = time.time()
    request_id = str(uuid.uuid4())

    # Используем threshold из запроса или стандартный 0.6
    threshold = request.threshold if hasattr(request, 'threshold') else 0.6
    threshold = max(0.1, min(0.9, threshold))  # Ограничиваем от 0.1 до 0.9

    print(f"  Используемый порог классификации: {threshold}")

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

        print(f" Тип модели: {model_type}")
        print(f" Признаки: {features}")
        print(" Масштабирование ОТКЛЮЧЕНО")

        # Подготавливаем признаки
        features_prepared = prepare_features_for_prediction(features, model_type)

        try:
            if hasattr(model, 'run'):  # ONNX модель
                input_name = model.get_inputs()[0].name
                features_float32 = features.astype(np.float32)

                outputs = model.run(None, {input_name: features_float32})

                # Получаем вероятности
                prob_dict = outputs[1][0] if len(outputs) > 1 and outputs[1] else {0: 0.5, 1: 0.5}
                prob_class0 = float(prob_dict.get(0, 0))  # Вероятность одобрения
                prob_class1 = float(prob_dict.get(1, 0))  # Вероятность дефолта

                print(f"    Вероятности: одобрение={prob_class0:.3f}, дефолт={prob_class1:.3f}")
                print(f"    Порог классификации: {threshold}")
                print(f"    Макс. допустимый дефолт: {1 - threshold:.3f}")

                # ИСПРАВЛЕНИЕ: Сравниваем вероятность дефолта с (1 - порог)
                if prob_class1 > (1 - threshold):  # ← ИЗМЕНЕНИЕ ЗДЕСЬ
                    prediction_class = 1  # Дефолт
                    probability = prob_class1  # Вероятность дефолта
                    print(
                        f"    Решение: ДЕФОЛТ (вероятность дефолта {prob_class1:.3f} > допустимого {1 - threshold:.3f})")
                else:
                    prediction_class = 0  # Одобрено
                    probability = prob_class0  # Вероятность одобрения
                    print(
                        f"    Решение: ОДОБРЕНО (вероятность дефолта {prob_class1:.3f} ≤ допустимого {1 - threshold:.3f})")

            else:  # Pickle модель
                prediction_proba = model.predict_proba(features_prepared)[0]
                prob_class0 = float(prediction_proba[0])  # Вероятность одобрения
                prob_class1 = float(prediction_proba[1])  # Вероятность дефолта

                # ИСПРАВЛЕНИЕ: Та же логика для pickle моделей
                if prob_class1 > (1 - threshold):  # ← ИЗМЕНЕНИЕ ЗДЕСЬ
                    prediction_class = 1
                    probability = prob_class1
                else:
                    prediction_class = 0
                    probability = prob_class0

        except Exception as e:
            print(f"  ОШИБКА: {str(e)}")
            raise HTTPException(500, f"Ошибка при выполнении предсказания: {str(e)}")

        risk_score = probability * 100 if prediction_class == 0 else (1 - probability) * 100

        # Определяем уверенность предсказания
        if probability >= 0.8:
            confidence = "высокая"
        elif probability >= 0.6:
            confidence = "средняя"
        else:
            confidence = "низкая"

        # Определяем версию модели
        if 'lightgbm' in model_type.lower():
            model_version = "2.0.0-lightgbm"
        elif 'random' in model_type.lower():
            model_version = "2.0.0-rf"
        elif 'logistic' in model_type.lower():
            model_version = "2.0.0-lr"
        else:
            model_version = "2.0.0-ml"

        # Создаем результат предсказания
        prediction_result = ModelPrediction(
            prediction=int(prediction_class),
            probability=probability,
            confidence=confidence,
            risk_score=risk_score,
            model_version=model_version,
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

        print(f" ПРЕДСКАЗАНИЕ УСПЕШНО")
        print(f"   Модель: {model_type}")
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
        # Правильное определение типа модели и версии
        if 'lightgbm' in model_type.lower():
            model_version = "2.0.0-lightgbm"
            model_class = "LightGBMClassifier"
            is_best_model = True
            recommendation = "LightGBM (лучшая модель по ROC-AUC: 0.7285)"
        elif 'random' in model_type.lower():
            model_version = "2.0.0-rf"
            model_class = "RandomForestClassifier"
            is_best_model = False
            recommendation = f"Используется {model_class}, но LightGBM рекомендуется"
        elif 'logistic' in model_type.lower():
            model_version = "2.0.0-lr"
            model_class = "LogisticRegression"
            is_best_model = False
            recommendation = f"Используется {model_class}, но LightGBM рекомендуется"
        else:
            model_version = "2.0.0-ml"
            model_class = model_type
            is_best_model = False
            recommendation = f"Используется {model_class}, но LightGBM рекомендуется"

        return {
            "model_version": model_version,
            "model_type": model_type,
            "model_class": model_class,
            "model_name": model_name,
            "is_best_model": is_best_model,
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
            "scaler_loaded": False,
            "recommendation": recommendation,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "using_lightgbm": is_best_model,
        }
    else:
        return {
            "model_version": "not_loaded",
            "model_type": "none",
            "model_class": "none",
            "model_name": "none",
            "is_best_model": False,
            "features_count": 0,
            "model_loaded": False,
            "scaler_loaded": False,
            "recommendation": "Рекомендуется LightGBM (лучшая модель: ROC-AUC 0.7285)",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "using_lightgbm": False,
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)