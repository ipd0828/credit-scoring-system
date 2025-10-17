"""
Скрипт для обучения моделей кредитного скоринга.

Этот скрипт выполняет:
1. Загрузку обработанных данных
2. Обучение различных моделей машинного обучения
3. Оценку качества моделей
4. Сохранение обученных моделей
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib

# Импорты для визуализации
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# from sklearn.svm import SVC  # Удалено для упрощения
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Импорты для машинного обучения
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

# Импорт MLflow
from .simple_mlflow_tracking import setup_mlflow_experiment

warnings.filterwarnings("ignore")


def load_processed_data(
    data_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]:
    """
    Загружает обработанные данные и препроцессор.

    Args:
        data_dir: Папка с обработанными данными

    Returns:
        Tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    data_path = Path(data_dir)

    print("Загрузка обработанных данных...")

    # Загружаем данные
    X_train = pd.read_csv(data_path / "X_train.csv")
    X_test = pd.read_csv(data_path / "X_test.csv")
    y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
    y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

    # Загружаем препроцессор
    preprocessor = joblib.load(data_path / "preprocessor.pkl")

    print(f"Загружено:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test, preprocessor


def create_models(X_train) -> Dict[str, Pipeline]:
    """
    Создает словарь моделей для обучения.

    Args:
        X_train: Обучающие данные для определения типов признаков

    Returns:
        Dict[str, Pipeline]: Словарь с моделями
    """
    print("\nСоздание моделей...")

    # Определяем типы признаков
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Создаем препроцессор
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    # Создаем модели
    models = {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=50, max_depth=10, random_state=42
                    ),
                ),
            ]
        ),
    }

    print(f"Создано {len(models)} моделей:")
    for name in models.keys():
        print(f"  - {name}")

    return models


def train_and_evaluate_models(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    use_mlflow: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Обучает модели и оценивает их качество.

    Args:
        models: Словарь с моделями
        X_train: Обучающие признаки
        X_test: Тестовые признаки
        y_train: Обучающая целевая переменная
        y_test: Тестовая целевая переменная

    Returns:
        Dict: Результаты обучения и оценки
    """
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ")
    print("=" * 60)

    results = {}
    predictions = {}
    probabilities = {}

    # Настройка MLflow
    if use_mlflow:
        tracker = setup_mlflow_experiment("credit-scoring-training")

    # Обучаем каждую модель
    for name, model in tqdm(models.items(), desc="Обучение моделей", unit="модель"):
        print(f"\nОбучение {name}...")

        try:
            # Обучение
            model.fit(X_train, y_train)

            # Предсказания
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Сохраняем предсказания
            predictions[name] = y_pred
            probabilities[name] = y_proba

            # Вычисляем метрики
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }

            # Кросс-валидация
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5, scoring="roc_auc"
            )
            metrics["cv_auc_mean"] = cv_scores.mean()
            metrics["cv_auc_std"] = cv_scores.std()

            results[name] = metrics

            print(f"{name} завершён:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-score: {metrics['f1']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(
                f"  CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}"
            )

            # Логируем в MLflow
            if use_mlflow:
                try:
                    with tracker.start_run(run_name=name) as run:
                        # Логируем информацию о данных
                        tracker.log_data_info(X_train, X_test, y_train, y_test)

                        # Логируем параметры модели
                        model_params = model.get_params()
                        tracker.log_model_params(model_params)

                        # Логируем метрики
                        tracker.log_metrics(metrics)

                        # Логируем модель
                        tracker.log_model(model, name)

                        print(f"  MLflow run ID: {run.info.run_id}")

                except Exception as mlflow_error:
                    print(f"  Ошибка MLflow для {name}: {mlflow_error}")

        except Exception as e:
            print(f"Ошибка при обучении {name}: {e}")
            results[name] = {"error": str(e)}

    return {
        "results": results,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def create_comparison_plot(
    results: Dict[str, Dict[str, Any]], output_dir: str = "models/artifacts"
) -> None:
    """
    Создает график сравнения моделей.

    Args:
        results: Результаты обучения
        output_dir: Папка для сохранения
    """
    print("\nСоздание графика сравнения моделей...")

    # Подготавливаем данные для графика
    model_names = []
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_values = {metric: [] for metric in metrics}

    for model_name, model_results in results.items():
        if "error" not in model_results:
            model_names.append(model_name)
            for metric in metrics:
                metric_values[metric].append(model_results[metric])

    # Создаем график
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(model_names, metric_values[metric])
        ax.set_title(f"{metric.upper()}")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=45)

        # Добавляем значения на столбцы
        for bar, value in zip(bars, metric_values[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    # Удаляем лишний subplot
    axes[5].remove()

    plt.tight_layout()

    # Сохраняем график
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"График сохранен: {output_path / 'model_comparison.png'}")


def create_roc_curves(
    probabilities: Dict[str, np.ndarray],
    y_test: pd.Series,
    output_dir: str = "models/artifacts",
) -> None:
    """
    Создает ROC-кривые для всех моделей.

    Args:
        probabilities: Словарь с вероятностями предсказаний
        y_test: Тестовая целевая переменная
        output_dir: Папка для сохранения
    """
    print("\nСоздание ROC-кривых...")

    plt.figure(figsize=(10, 8))

    for name, y_proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    # Диагональ — случайный классификатор
    plt.plot([0, 1], [0, 1], "k--", label="Случайный классификатор")

    # Оформление
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривые: Сравнение моделей")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Сохраняем график
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"ROC-кривые сохранены: {output_path / 'roc_curves.png'}")


def save_models(
    models: Dict[str, Pipeline],
    results: Dict[str, Dict[str, Any]],
    output_dir: str = "models/trained",
) -> None:
    """
    Сохраняет обученные модели и результаты.

    Args:
        models: Словарь с обученными моделями
        results: Результаты обучения
        output_dir: Папка для сохранения
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nСохранение моделей в {output_path}...")

    # Сохраняем каждую модель
    for name, model in models.items():
        if "error" not in results.get(name, {}):
            model_path = output_path / f"{name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, model_path)
            print(f"  {name} -> {model_path}")

    # Сохраняем результаты
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_path / "model_results.csv")
    print(f"  Результаты -> {output_path / 'model_results.csv'}")

    # Сохраняем лучшую модель
    if not results_df.empty:
        best_model_name = results_df["roc_auc"].idxmax()
        best_model = models[best_model_name]
        joblib.dump(best_model, output_path / "best_model.pkl")
        print(
            f"  Лучшая модель ({best_model_name}) -> {output_path / 'best_model.pkl'}"
        )


def print_final_results(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Выводит финальные результаты сравнения моделей.

    Args:
        results: Результаты обучения
    """
    print("\n" + "=" * 60)
    print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)

    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results).T

    if results_df.empty:
        print("Нет результатов для отображения.")
        return

    # Удаляем столбцы с ошибками
    results_df = results_df.drop(columns=["error"], errors="ignore")

    # Сортируем по ROC-AUC
    results_df = results_df.sort_values("roc_auc", ascending=False)

    print("\nСравнение моделей:")
    print(results_df.round(4))

    # Выводим лучшую модель
    if "roc_auc" in results_df.columns:
        best_model = results_df["roc_auc"].idxmax()
        best_auc = results_df.loc[best_model, "roc_auc"]
        print(f"\nЛучшая модель: {best_model} (ROC-AUC: {best_auc:.4f})")


def main():
    """Основная функция для запуска обучения моделей."""
    # Загружаем данные
    X_train, X_test, y_train, y_test, preprocessor = load_processed_data()

    # Создаем модели
    models = create_models(X_train)

    # Обучаем и оцениваем модели
    training_results = train_and_evaluate_models(
        models, X_train, X_test, y_train, y_test
    )

    # Создаем визуализации
    create_comparison_plot(training_results["results"])
    create_roc_curves(training_results["probabilities"], y_test)

    # Сохраняем модели
    save_models(models, training_results["results"])

    # Выводим финальные результаты
    print_final_results(training_results["results"])

    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ ЗАВЕРШЕНО УСПЕШНО")
    print("=" * 60)


if __name__ == "__main__":
    main()
