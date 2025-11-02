"""
Скрипт для валидации моделей кредитного скоринга на кастомных признаках.
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
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings("ignore")


def prepare_data_for_catboost(X_data):
    """Подготовка данных для CatBoost - преобразование категориальных признаков."""
    X_prepared = X_data.copy()

    # Категориальные признаки
    categorical_features = ["sex", "marriage_new", "pay_new", "education_new"]

    # Преобразуем в целые числа, затем в строки
    for feature in categorical_features:
        if feature in X_prepared.columns:
            X_prepared[feature] = X_prepared[feature].astype(int).astype(str)

    return X_prepared


def load_models_and_data():
    """Загрузка моделей и данных для валидации."""
    print("Загрузка моделей и данных...")

    # Загрузка данных
    data_path = Path("data/processed_custom")
    X_test = pd.read_csv(data_path / "X_test.csv")
    y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

    # Загрузка моделей
    models_dir = Path("models/trained_custom")
    models = {}

    # Загружаем все модели .pkl
    model_files = list(models_dir.glob("*.pkl"))

    for model_file in model_files:
        model_name = model_file.stem

        # Пропускаем служебные файлы
        if any(skip in model_name for skip in ["scaler", "preprocessor", "feature"]):
            continue

        try:
            model = joblib.load(model_file)
            models[model_name] = model
            print(f"  Загружена модель: {model_name}")
        except Exception as e:
            print(f"  Ошибка загрузки {model_name}: {e}")

    print(f"\nЗагружено {len(models)} моделей")
    print(f"Данные: X_test {X_test.shape}")

    return models, X_test, y_test


def validate_single_model(model_name, model, X_test, y_test):
    """Валидация одной модели на тестовых данных."""
    print(f"\nВалидация модели: {model_name}")

    try:
        # Для CatBoost моделей используем подготовленные данные
        if "catboost" in model_name.lower():
            X_test_prepared = prepare_data_for_catboost(X_test)
            y_pred = model.predict(X_test_prepared)
            y_proba = model.predict_proba(X_test_prepared)[:, 1]
        else:
            # Для остальных моделей используем исходные данные
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        # Основные метрики
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

        # ROC и PR кривые
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  Average Precision: {metrics['average_precision']:.4f}")

        return {
            "model_name": model_name,
            "metrics": metrics,
            "predictions": y_pred,
            "probabilities": y_proba,
            "confusion_matrix": cm,
            "roc_curve": (fpr, tpr, roc_thresholds),
            "pr_curve": (precision, recall, pr_thresholds),
        }

    except Exception as e:
        print(f"  Ошибка валидации {model_name}: {e}")
        return {"model_name": model_name, "error": str(e)}


def create_validation_plots(validation_results, output_dir="models/artifacts_custom"):
    """Создает графики для анализа валидации."""
    print("\nСоздание графиков валидации...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Фильтруем успешные результаты
    valid_results = [r for r in validation_results if "error" not in r]

    if not valid_results:
        print("Нет валидных результатов для создания графиков")
        return

    # 1. Сравнение метрик
    plt.figure(figsize=(14, 8))
    model_names = [r["model_name"] for r in valid_results]
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    x = np.arange(len(model_names))
    width = 0.15

    for i, metric in enumerate(metrics):
        values = [r["metrics"][metric] for r in valid_results]
        plt.bar(x + i * width, values, width, label=metric, alpha=0.8)

    plt.xlabel("Модели")
    plt.ylabel("Значения метрик")
    plt.title("Сравнение метрик моделей")
    plt.xticks(x + width * 2, model_names, rotation=45, ha="right")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_path / "validation_metrics_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. ROC кривые
    plt.figure(figsize=(10, 8))
    for result in valid_results:
        fpr, tpr, _ = result["roc_curve"]
        roc_auc = result["metrics"]["roc_auc"]
        plt.plot(
            fpr, tpr, label=f"{result['model_name']} (AUC = {roc_auc:.3f})", linewidth=2
        )

    plt.plot([0, 1], [0, 1], "k--", label="Случайный классификатор", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривые: Сравнение моделей")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "validation_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Precision-Recall кривые
    plt.figure(figsize=(10, 8))
    for result in valid_results:
        precision, recall, _ = result["pr_curve"]
        ap = result["metrics"]["average_precision"]
        plt.plot(
            recall,
            precision,
            label=f"{result['model_name']} (AP = {ap:.3f})",
            linewidth=2,
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall кривые: Сравнение моделей")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "validation_pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Матрицы ошибок
    n_models = len(valid_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for i, result in enumerate(valid_results):
        cm = result["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
        axes[i].set_title(f'{result["model_name"]}\nМатрица ошибок')
        axes[i].set_xlabel("Предсказанный класс")
        axes[i].set_ylabel("Истинный класс")

    plt.tight_layout()
    plt.savefig(
        output_path / "validation_confusion_matrices.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("  Графики валидации сохранены")


def generate_validation_report(
    validation_results, output_dir="models/artifacts_custom"
):
    """Генерирует детальный отчет о валидации."""
    print("\nГенерация отчета о валидации...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Фильтруем успешные результаты
    valid_results = [r for r in validation_results if "error" not in r]

    if not valid_results:
        print("Нет валидных результатов для создания отчета")
        return

    # Создаем детальный отчет
    report_data = []

    for result in valid_results:
        row = {"Модель": result["model_name"]}
        row.update(
            {
                "Accuracy": result["metrics"]["accuracy"],
                "Precision": result["metrics"]["precision"],
                "Recall": result["metrics"]["recall"],
                "F1-score": result["metrics"]["f1"],
                "ROC-AUC": result["metrics"]["roc_auc"],
                "Avg Precision": result["metrics"]["average_precision"],
            }
        )
        report_data.append(row)

    # Создаем DataFrame
    report_df = pd.DataFrame(report_data)
    report_df = report_df.sort_values("ROC-AUC", ascending=False)

    # Сохраняем отчет
    report_path = output_path / "validation_report.csv"
    report_df.to_csv(report_path, index=False)

    # Создаем текстовый отчет
    text_report_path = output_path / "validation_report.txt"
    with open(text_report_path, "w", encoding="utf-8") as f:
        f.write("ОТЧЕТ О ВАЛИДАЦИИ МОДЕЛЕЙ КРЕДИТНОГО СКОРИНГА (КАСТОМНЫЕ ПРИЗНАКИ)\n")
        f.write("=" * 70 + "\n\n")

        f.write("СВОДНАЯ ТАБЛИЦА МЕТРИК:\n")
        f.write("-" * 50 + "\n")
        f.write(report_df.to_string(index=False, float_format="%.4f"))
        f.write("\n\n")

        f.write("ЛУЧШАЯ МОДЕЛЬ:\n")
        f.write("-" * 50 + "\n")
        best_model = report_df.iloc[0]
        f.write(f"Название: {best_model['Модель']}\n")
        f.write(f"ROC-AUC: {best_model['ROC-AUC']:.4f}\n")
        f.write(f"Accuracy: {best_model['Accuracy']:.4f}\n")
        f.write(f"Precision: {best_model['Precision']:.4f}\n")
        f.write(f"Recall: {best_model['Recall']:.4f}\n")
        f.write(f"F1-score: {best_model['F1-score']:.4f}\n")
        f.write(f"Average Precision: {best_model['Avg Precision']:.4f}\n\n")

        f.write("РЕКОМЕНДАЦИИ:\n")
        f.write("-" * 50 + "\n")
        if best_model["Recall"] < 0.5:
            f.write("⚠️  Низкий Recall - модель плохо обнаруживает дефолты\n")
        if best_model["Precision"] < 0.4:
            f.write("⚠️  Низкий Precision - много ложных срабатываний\n")
        if best_model["ROC-AUC"] > 0.7:
            f.write("✅ Хорошее качество модели (ROC-AUC > 0.7)\n")
        else:
            f.write("⚠️  ROC-AUC ниже 0.7 - требуется улучшение модели\n")

    print(f"Отчет сохранен:")
    print(f"  CSV: {report_path}")
    print(f"  TXT: {text_report_path}")


def print_validation_summary(validation_results):
    """Выводит краткую сводку результатов валидации."""
    print("\n" + "=" * 60)
    print("СВОДКА ВАЛИДАЦИИ МОДЕЛЕЙ")
    print("=" * 60)

    # Фильтруем успешные результаты
    valid_results = [r for r in validation_results if "error" not in r]

    if not valid_results:
        print("Нет валидных результатов валидации")
        return

    # Создаем сводную таблицу
    summary_data = []
    for result in valid_results:
        row = {
            "Модель": result["model_name"],
            "Accuracy": f"{result['metrics']['accuracy']:.4f}",
            "Precision": f"{result['metrics']['precision']:.4f}",
            "Recall": f"{result['metrics']['recall']:.4f}",
            "F1-score": f"{result['metrics']['f1']:.4f}",
            "ROC-AUC": f"{result['metrics']['roc_auc']:.4f}",
            "Avg Precision": f"{result['metrics']['average_precision']:.4f}",
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("ROC-AUC", ascending=False)

    print("\nСравнение моделей:")
    print(summary_df.to_string(index=False))

    # Выводим лучшую модель
    best_model = summary_df.iloc[0]
    print(f"\nЛучшая модель: {best_model['Модель']}")
    print(f"  ROC-AUC: {best_model['ROC-AUC']}")
    print(f"  Accuracy: {best_model['Accuracy']}")
    print(f"  F1-score: {best_model['F1-score']}")


def main():
    """Основная функция для запуска валидации моделей."""
    try:
        # Загружаем модели и данные
        models, X_test, y_test = load_models_and_data()

        if not models:
            print("Не найдено моделей для валидации")
            return

        # Валидируем каждую модель
        validation_results = []

        for model_name, model in models.items():
            result = validate_single_model(model_name, model, X_test, y_test)
            validation_results.append(result)

        # Создаем графики
        create_validation_plots(validation_results)

        # Генерируем отчет
        generate_validation_report(validation_results)

        # Выводим сводку
        print_validation_summary(validation_results)

        print("\n" + "=" * 60)
        print("ВАЛИДАЦИЯ МОДЕЛЕЙ ЗАВЕРШЕНА УСПЕШНО")
        print("=" * 60)

    except Exception as e:
        print(f"Ошибка при валидации моделей: {e}")
        raise


if __name__ == "__main__":
    main()
