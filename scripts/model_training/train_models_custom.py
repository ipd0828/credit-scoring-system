"""
Скрипт для обучения моделей кредитного скоринга на кастомных признаках.
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings("ignore")

# Пытаемся импортировать CatBoost
try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost не установлен. Установите: pip install catboost")


def load_processed_data():
    """Загрузка обработанных данных для кастомных признаков."""
    print("Загрузка обработанных данных...")

    data_path = Path("data/processed_custom")

    if not data_path.exists():
        raise FileNotFoundError(f"Папка с данными не найдена: {data_path}")

    # Проверяем наличие файлов
    required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    for file in required_files:
        if not (data_path / file).exists():
            raise FileNotFoundError(f"Файл не найден: {data_path / file}")

    X_train = pd.read_csv(data_path / "X_train.csv")
    X_test = pd.read_csv(data_path / "X_test.csv")
    y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
    y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

    print(f"Загружено:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")

    # Выводим информацию о признаках
    print(f"\nИспользуемые признаки ({len(X_train.columns)}):")
    for feature in X_train.columns:
        print(f"  - {feature}")

    # УБИРАЕМ загрузку scalers - они больше не нужны
    # scalers = None
    # scalers_path = data_path / "artifacts" / "scalers.pkl"
    # if scalers_path.exists():
    #     scalers = joblib.load(scalers_path)
    #     print("Загружены scalers")

    return X_train, X_test, y_train, y_test  # Убрали scalers


def prepare_data_for_catboost(X_train, X_test):
    """Подготовка данных для CatBoost - преобразование категориальных признаков."""
    print("Подготовка данных для CatBoost...")

    X_train_cat = X_train.copy()
    X_test_cat = X_test.copy()

    # Категориальные признаки
    categorical_features = ['sex', 'marriage_new', 'pay_new', 'education_new']

    # Преобразуем в целые числа, затем в строки
    for feature in categorical_features:
        if feature in X_train_cat.columns:
            X_train_cat[feature] = X_train_cat[feature].astype(int).astype(str)
            X_test_cat[feature] = X_test_cat[feature].astype(int).astype(str)

    print("Категориальные признаки преобразованы для CatBoost")
    return X_train_cat, X_test_cat, categorical_features


def create_models(X_train, y_train):
    """Создание моделей для обучения."""
    print("Создание моделей...")

    # Рассчитываем веса классов для дисбаланса
    class_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    print(f"Дисбаланс классов: {class_ratio:.2f}:1")

    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            class_weight='balanced'
        ),
    }

    # Добавляем CatBoost если доступен
    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            random_state=42,
            verbose=False,
            thread_count=-1,
            # БАЛАНСИРОВКА КЛАССОВ - ВАЖНО!
            auto_class_weights='Balanced',
            loss_function='Logloss',
            eval_metric='AUC',
            iterations=100,
            learning_rate=0.1,
            depth=6
        )

    print(f"Создано {len(models)} моделей:")
    for name in models.keys():
        print(f"  - {name}")

    return models


def evaluate_model(model, X_test, y_test, model_name):
    """Оценка модели на тестовых данных."""
    # Предсказания
    y_pred = model.predict(X_test)

    # Для CatBoost используем predict_proba с учетом особенностей
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        # Проверяем форму predict_proba
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1]
        else:
            y_proba = y_proba
    else:
        y_proba = model.predict(X_test)

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Кросс-валидация
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_test, y_test, cv=cv, scoring='roc_auc')

    print(f"{model_name} завершён:")
    print(f"  Точность (Accuracy): {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std()
    }


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Обучение и оценка моделей."""
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ")
    print("=" * 60)

    # Подготавливаем данные для CatBoost
    X_train_cat, X_test_cat, categorical_features = prepare_data_for_catboost(X_train, X_test)

    models = create_models(X_train, y_train)
    results = {}

    for name, model in models.items():
        print(f"\nОбучение {name}...")

        try:
            # Обучение модели
            if name == "CatBoost":
                # Получаем индексы категориальных признаков
                cat_features_indices = [i for i, col in enumerate(X_train_cat.columns) if col in categorical_features]

                model.fit(
                    X_train_cat, y_train,
                    cat_features=cat_features_indices,
                    verbose=False,
                    plot=False
                )

                # Оценка на подготовленных данных для CatBoost
                results[name] = evaluate_model(model, X_test_cat, y_test, name)
            else:
                # Для остальных моделей используем исходные данные
                model.fit(X_train, y_train)
                results[name] = evaluate_model(model, X_test, y_test, name)

            # Сохраняем модель
            models_dir = Path("models/trained_custom")
            models_dir.mkdir(parents=True, exist_ok=True)

            model_path = models_dir / f"{name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, model_path)
            print(f"  Модель сохранена: {model_path}")

        except Exception as e:
            print(f"  Ошибка при обучении {name}: {e}")
            continue

    return models, results


def save_results(results):
    """Сохранение результатов обучения."""
    print("\nСохранение результатов...")

    if not results:
        print("  Нет результатов для сохранения")
        return None

    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('roc_auc', ascending=False)

    # Сохраняем результаты
    output_dir = Path("models/trained_custom")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "model_results.csv"
    results_df.to_csv(results_path)
    print(f"  Результаты сохранены: {results_path}")

    # Сохраняем лучшую модель
    best_model_name = results_df.index[0]
    best_model_path = output_dir / f"{best_model_name.lower().replace(' ', '_')}.pkl"
    final_best_path = output_dir / "best_model.pkl"

    # Копируем лучшую модель
    import shutil
    shutil.copy2(best_model_path, final_best_path)
    print(f"  Лучшая модель сохранена: {final_best_path}")
    print(f"  Лучшая модель: {best_model_name} (ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f})")

    return results_df


def main():
    """Основная функция."""
    try:
        # Загрузка данных
        X_train, X_test, y_train, y_test = load_processed_data()

        # Обучение и оценка моделей
        models, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

        # Сохранение результатов
        results_df = save_results(results)

        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ МОДЕЛЕЙ ЗАВЕРШЕНО УСПЕШНО")
        print("=" * 60)

        if results_df is not None:
            print(f"\nЛучшая модель: {results_df.index[0]}")
            print(f"ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f}")
            print(f"Точность (Accuracy): {results_df.iloc[0]['accuracy']:.4f}")
            print(f"F1-score: {results_df.iloc[0]['f1']:.4f}")

            # Выводим все метрики для лучшей модели
            best_model_metrics = results_df.iloc[0]
            print(f"\nДетальные метрики лучшей модели:")
            print(f"  Precision: {best_model_metrics['precision']:.4f}")
            print(f"  Recall: {best_model_metrics['recall']:.4f}")
            print(f"  CV AUC: {best_model_metrics['cv_auc_mean']:.4f} ± {best_model_metrics['cv_auc_std']:.4f}")
        else:
            print("Нет успешно обученных моделей")

    except Exception as e:
        print(f"Ошибка: {e}")
        raise


if __name__ == "__main__":
    main()