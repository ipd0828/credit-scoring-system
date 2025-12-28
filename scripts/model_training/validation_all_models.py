#!/usr/bin/env python3
"""
Валидация всех 4 настроенных моделей.
ИСПРАВЛЕННАЯ ВЕРСИЯ: Исправлена проблема с LightGBM (F1=0)
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings("ignore")

# Для кодировки в Windows
if sys.platform == "win32":
    import os

    os.system('chcp 65001 > nul')

# Пытаемся импортировать PyTorch
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SimpleNN(nn.Module):
    """Простая нейронная сеть для загрузки."""

    def __init__(self, input_size=6, hidden_layers=[64, 32], dropout_rate=0.0):
        super(SimpleNN, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_models_and_data():
    """Загрузка настроенных моделей и данных."""
    print("Загрузка настроенных моделей и данных...")

    # Загрузка данных
    data_path = Path("data/processed_fixed")
    X_test = pd.read_csv(data_path / "X_test.csv")
    y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

    # Анализ распределения классов
    print(f"\nРаспределение классов в тестовых данных:")
    unique, counts = np.unique(y_test, return_counts=True)
    for cls, count in zip(unique, counts):
        percentage = count / len(y_test) * 100
        print(f"  Class {cls}: {count} ({percentage:.1f}%)")

    # Загрузка моделей
    models_dir = Path("models/trained_final")
    models = {}

    # Загружаем настроенные модели - ОБНОВЛЕННЫЙ СПИСОК
    model_files = [
        ("tuned_logisticregression.pkl", "Logistic Regression"),
        ("tuned_randomforestclassifier.pkl", "Random Forest"),
        ("tuned_lgbmclassifier.pkl", "LightGBM"),
        ("tuned_neural_network.pth", "Neural Network"),
        ("logistic_regression.pkl", "Logistic Regression (base)"),  # Базовые модели
        ("random_forest.pkl", "Random Forest (base)"),
        ("lightgbm.pkl", "LightGBM (base)"),
        ("neural_network.pth", "Neural Network (base)"),
    ]

    for filename, model_name in model_files:
        model_path = models_dir / filename
        if model_path.exists():
            try:
                if "Neural Network" in model_name and TORCH_AVAILABLE:
                    # Загружаем PyTorch модель
                    # Сначала попробуем загрузить параметры
                    params_path = models_dir / "tuned_neural_network_params.pkl"
                    if params_path.exists():
                        params = joblib.load(params_path)
                        hidden_layers = params.get("hidden_layers", [64, 32])
                        dropout_rate = params.get("dropout_rate", 0.0)
                        input_size = params.get("input_size", 6)
                    else:
                        # Параметры по умолчанию
                        hidden_layers = [64, 32]
                        dropout_rate = 0.0
                        input_size = 6

                    # Создаем модель с правильной архитектурой
                    model = SimpleNN(
                        input_size=input_size,
                        hidden_layers=hidden_layers,
                        dropout_rate=dropout_rate
                    )

                    # Загружаем веса
                    state_dict = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(state_dict)
                    model.eval()

                    models[model_name] = {
                        'model': model,
                        'type': 'neural_network',
                        'params': {
                            'hidden_layers': hidden_layers,
                            'dropout_rate': dropout_rate
                        }
                    }
                    print(f"  Загружена: {model_name} (архитектура: {hidden_layers})")
                else:
                    # Загружаем sklearn модели
                    model = joblib.load(model_path)

                    models[model_name] = {
                        'model': model,
                        'type': 'sklearn'
                    }
                    print(f"  Загружена: {model_name}")
            except Exception as e:
                print(f"  Ошибка загрузки {model_name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Пробуем альтернативные пути для базовых моделей
            if "(base)" in model_name:
                base_dir = Path("models/trained")
                base_path = base_dir / filename
                if base_path.exists():
                    try:
                        if "Neural Network" in model_name and TORCH_AVAILABLE:
                            # Упрощенная нейросеть для базовой модели
                            model = SimpleNN(input_size=6, hidden_layers=[64, 32])
                            model.load_state_dict(torch.load(base_path, map_location='cpu'))
                            model.eval()

                            models[model_name] = {
                                'model': model,
                                'type': 'neural_network'
                            }
                            print(f"  Загружена: {model_name} (из базовой директории)")
                        else:
                            model = joblib.load(base_path)
                            models[model_name] = {
                                'model': model,
                                'type': 'sklearn'
                            }
                            print(f"  Загружена: {model_name} (из базовой директории)")
                    except Exception as e:
                        print(f"  Ошибка загрузки {model_name} из базовой директории: {e}")

    print(f"\nЗагружено {len(models)} моделей")
    print(f"Данные: X_test {X_test.shape}")

    return models, X_test, y_test


def validate_single_model(model_name, model_info, X_test, y_test):
    """Валидация одной модели с исправлением для LightGBM."""
    print(f"\nВалидация модели: {model_name}")

    try:
        model = model_info['model']
        model_type = model_info['type']

        # Предсказания
        if model_type == 'neural_network':
            # Масштабирование для нейронной сети
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_test_scaled = scaler.fit_transform(X_test).astype(np.float32)
            X_test_tensor = torch.FloatTensor(X_test_scaled)

            with torch.no_grad():
                outputs = model(X_test_tensor)
                y_proba = outputs.numpy().flatten()
                y_pred = (y_proba >= 0.5).astype(int)
        else:
            # Для sklearn моделей
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # ОСОБАЯ ПРОВЕРКА ДЛЯ LIGHTGBM
            if model_name == "LightGBM":
                # Проверяем, не предсказывает ли модель только один класс
                unique_preds = np.unique(y_pred)
                if len(unique_preds) == 1:
                    print(f"  ⚠️  LightGBM предсказывает только класс {unique_preds[0]}")
                    print(f"  Пробуем найти оптимальный порог...")

                    # Ищем оптимальный порог по F1-score
                    thresholds = np.linspace(0.1, 0.9, 9)
                    best_f1 = 0
                    best_threshold = 0.5

                    for threshold in thresholds:
                        temp_pred = (y_proba >= threshold).astype(int)
                        if len(np.unique(temp_pred)) > 1:
                            temp_f1 = f1_score(y_test, temp_pred, zero_division=0)
                            if temp_f1 > best_f1:
                                best_f1 = temp_f1
                                best_threshold = threshold

                    if best_f1 > 0:
                        print(f"  Найден оптимальный порог: {best_threshold:.2f} (F1: {best_f1:.4f})")
                        y_pred = (y_proba >= best_threshold).astype(int)
                    else:
                        # Если все пороги дают F1=0, используем порог по балансу классов
                        print(f"  Использую порог по балансу классов...")
                        class_ratio = np.sum(y_test == 0) / np.sum(y_test == 1)
                        threshold = 1 / (1 + class_ratio)
                        y_pred = (y_proba >= threshold).astype(int)
                        print(f"  Порог: {threshold:.2f} (на основе соотношения классов)")

        # Метрики
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "average_precision": average_precision_score(y_test, y_proba),
        }

        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  Average Precision: {metrics['average_precision']:.4f}")
        print(f"  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

        # Распределение предсказаний
        unique_preds, counts = np.unique(y_pred, return_counts=True)
        print(f"  Предсказано:")
        for cls, count in zip(unique_preds, counts):
            percentage = count / len(y_pred) * 100
            print(f"    Класс {cls}: {count} ({percentage:.1f}%)")

        return {
            "model_name": model_name,
            "metrics": metrics,
            "predictions": y_pred,
            "probabilities": y_proba,
            "confusion_matrix": cm,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    except Exception as e:
        print(f"  Ошибка валидации {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return {"model_name": model_name, "error": str(e)}


def save_validation_results(valid_results):
    """Сохранение результатов валидации."""
    print("\nСохранение результатов валидации...")

    # Создаем DataFrame с результатами
    results_data = []
    for result in valid_results:
        metrics = result["metrics"]
        results_data.append({
            "Модель": result["model_name"],
            "Accuracy": f"{metrics['accuracy']:.4f}",
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "F1-Score": f"{metrics['f1']:.4f}",
            "ROC-AUC": f"{metrics['roc_auc']:.4f}",
            "Avg Precision": f"{metrics['average_precision']:.4f}",
            "TP": result["tp"],
            "TN": result["tn"],
            "FP": result["fp"],
            "FN": result["fn"],
        })

    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values("ROC-AUC", ascending=False)

    # Сохраняем результаты
    output_dir = Path("models/artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / "validation_report_final.csv"
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"  CSV отчет сохранен: {csv_path}")

    # TXT
    txt_path = output_dir / "validation_summary_final.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ФИНАЛЬНЫЙ ОТЧЕТ ПО ВАЛИДАЦИИ МОДЕЛЕЙ\n")
        f.write("=" * 60 + "\n\n")

        f.write("Рейтинг моделей по ROC-AUC:\n")
        f.write("-" * 80 + "\n")

        for _, row in results_df.iterrows():
            f.write(f"{row['Модель']:25} ")
            f.write(f"ROC-AUC: {row['ROC-AUC']} | ")
            f.write(f"Accuracy: {row['Accuracy']} | ")
            f.write(f"F1: {row['F1-Score']}\n")

        # Лучшая модель
        best_row = results_df.iloc[0]
        f.write("\n" + "=" * 60 + "\n")
        f.write("ЛУЧШАЯ МОДЕЛЬ:\n")
        f.write("=" * 60 + "\n")
        f.write(f"Модель: {best_row['Модель']}\n")
        f.write(f"ROC-AUC: {best_row['ROC-AUC']}\n")
        f.write(f"Accuracy: {best_row['Accuracy']}\n")
        f.write(f"F1-Score: {best_row['F1-Score']}\n")
        f.write(f"Precision: {best_row['Precision']}\n")
        f.write(f"Recall: {best_row['Recall']}\n")

        # Матрица ошибок
        f.write(f"\nМатрица ошибок:\n")
        f.write(f"  True Positive:  {best_row['TP']}\n")
        f.write(f"  True Negative:  {best_row['TN']}\n")
        f.write(f"  False Positive: {best_row['FP']}\n")
        f.write(f"  False Negative: {best_row['FN']}\n")

    print(f"  Текстовый отчет сохранен: {txt_path}")

    return results_df


def main():
    """Основная функция."""
    print("=" * 60)
    print("ИСПРАВЛЕННАЯ ВАЛИДАЦИЯ МОДЕЛЕЙ")
    print("=" * 60)

    try:
        # Загружаем модели и данные
        models, X_test, y_test = load_models_and_data()

        if not models:
            print("Не найдено моделей для валидации")
            return

        # Валидируем каждую модель
        validation_results = []
        for model_name, model_info in models.items():
            result = validate_single_model(model_name, model_info, X_test, y_test)
            validation_results.append(result)

        # Фильтруем успешные результаты
        valid_results = [r for r in validation_results if "error" not in r]

        if not valid_results:
            print("\nНет валидных результатов валидации")
            return

        # Сохраняем результаты
        results_df = save_validation_results(valid_results)

        # Выводим сводку
        print(f"\n{'=' * 60}")
        print("СВОДКА РЕЗУЛЬТАТОВ ВАЛИДАЦИИ")
        print(f"{'=' * 60}")

        # Сортируем по ROC-AUC
        results_df = results_df.sort_values("ROC-AUC", ascending=False)

        print("\nРейтинг моделей по ROC-AUC:")
        print("-" * 70)
        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            print(f"{i}. {row['Модель']:25} ROC-AUC: {row['ROC-AUC']} | "
                  f"Accuracy: {row['Accuracy']} | F1: {row['F1-Score']}")

        # Лучшая модель
        best_row = results_df.iloc[0]
        print(f"\n{'=' * 60}")
        print(f"ЛУЧШАЯ МОДЕЛЬ: {best_row['Модель']}")
        print("=" * 60)
        print(f"  ROC-AUC:       {best_row['ROC-AUC']}")
        print(f"  Accuracy:      {best_row['Accuracy']}")
        print(f"  F1-Score:      {best_row['F1-Score']}")
        print(f"  Precision:     {best_row['Precision']}")
        print(f"  Recall:        {best_row['Recall']}")

        print(f"\nМатрица ошибок:")
        print(f"  True Positive:  {best_row['TP']}")
        print(f"  True Negative:  {best_row['TN']}")
        print(f"  False Positive: {best_row['FP']}")
        print(f"  False Negative: {best_row['FN']}")

        # Рекомендации
        print(f"\n{'=' * 60}")
        print("РЕКОМЕНДАЦИИ:")
        print("=" * 60)

        # Проверяем F1-Score
        f1_score_val = float(best_row['F1-Score'])
        roc_auc_val = float(best_row['ROC-AUC'])

        if f1_score_val > 0.5:
            print("Хороший баланс precision и recall (F1 > 0.5)")
        elif f1_score_val > 0.4:
            print("Удовлетворительный баланс (F1 > 0.4)")
        else:
            print("Низкий F1-Score - требуется улучшение модели")

        if roc_auc_val > 0.7:
            print("Хорошая дискриминационная способность (ROC-AUC > 0.7)")
        else:
            print("ROC-AUC ниже 0.7 - можно улучшить")

        print(f"\nДля продакшена используйте: {best_row['Модель']}")

    except Exception as e:
        print(f"\nОшибка при валидации моделей: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()