"""
Скрипт для валидации моделей кредитного скоринга.

Этот скрипт выполняет:
1. Загрузку обученных моделей
2. Валидацию на тестовых данных
3. Анализ производительности моделей
4. Создание отчетов о валидации
5. Сравнение различных моделей
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib

# Импорты для визуализации
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Импорты для машинного обучения
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_models_and_data(
    models_dir: str = "models/trained", data_dir: str = "data/processed"
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Загружает обученные модели и тестовые данные.

    Args:
        models_dir: Папка с обученными моделями
        data_dir: Папка с обработанными данными

    Returns:
        Tuple: (модели, X_train, X_test, y_train, y_test)
    """
    models_path = Path(models_dir)
    data_path = Path(data_dir)

    print("Загрузка моделей и данных...")

    # Загружаем данные
    X_train = pd.read_csv(data_path / "X_train.csv")
    X_test = pd.read_csv(data_path / "X_test.csv")
    y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
    y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

    # Загружаем модели
    models = {}
    model_files = list(models_path.glob("*.pkl"))

    for model_file in model_files:
        model_name = model_file.stem
        try:
            model = joblib.load(model_file)
            models[model_name] = model
            print(f"  Загружена модель: {model_name}")
        except Exception as e:
            print(f"  Ошибка загрузки {model_name}: {e}")

    print(f"\nЗагружено {len(models)} моделей")
    print(f"Данные: X_train {X_train.shape}, X_test {X_test.shape}")

    return models, X_train, X_test, y_train, y_test


def validate_single_model(
    model_name: str, model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:
    """
    Валидирует одну модель на тестовых данных.

    Args:
        model_name: Название модели
        model: Обученная модель
        X_test: Тестовые признаки
        y_test: Тестовая целевая переменная

    Returns:
        Dict: Результаты валидации
    """
    print(f"\nВалидация модели: {model_name}")

    try:
        # Предсказания
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Основные метрики
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "average_precision": average_precision_score(y_test, y_proba),
            "log_loss": log_loss(y_test, y_proba),
        }

        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Дополнительные метрики
        metrics.update(
            {
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
                "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
                "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
            }
        )

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


def cross_validate_model(
    model: Any, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Выполняет кросс-валидацию модели.

    Args:
        model: Модель для валидации
        X: Признаки
        y: Целевая переменная
        cv_folds: Количество фолдов

    Returns:
        Dict: Результаты кросс-валидации
    """
    print(f"  Кросс-валидация ({cv_folds} фолдов)...")

    try:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Различные метрики для кросс-валидации
        scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        cv_results = {}

        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            cv_results[metric] = {
                "mean": scores.mean(),
                "std": scores.std(),
                "scores": scores,
            }

        return cv_results

    except Exception as e:
        print(f"  Ошибка кросс-валидации: {e}")
        return {"error": str(e)}


def create_validation_plots(
    validation_results: List[Dict[str, Any]], output_dir: str = "models/artifacts"
) -> None:
    """
    Создает графики для анализа валидации.

    Args:
        validation_results: Результаты валидации
        output_dir: Папка для сохранения
    """
    print("\nСоздание графиков валидации...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Фильтруем успешные результаты
    valid_results = [r for r in validation_results if "error" not in r]

    if not valid_results:
        print("Нет валидных результатов для создания графиков")
        return

    # 1. Сравнение метрик
    create_metrics_comparison_plot(valid_results, output_path)

    # 2. ROC кривые
    create_roc_curves_plot(valid_results, output_path)

    # 3. Precision-Recall кривые
    create_pr_curves_plot(valid_results, output_path)

    # 4. Матрицы ошибок
    create_confusion_matrices_plot(valid_results, output_path)


def create_metrics_comparison_plot(
    validation_results: List[Dict[str, Any]], output_path: Path
) -> None:
    """Создает график сравнения метрик."""
    print("  Создание графика сравнения метрик...")

    # Подготавливаем данные
    model_names = [r["model_name"] for r in validation_results]
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_values = {metric: [] for metric in metrics}

    for result in validation_results:
        for metric in metrics:
            metric_values[metric].append(result["metrics"][metric])

    # Создаем график
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
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

    plt.suptitle("Сравнение метрик валидации", fontsize=16)
    plt.tight_layout()

    plot_path = output_path / "validation_metrics_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"    График сохранен: {plot_path}")


def create_roc_curves_plot(
    validation_results: List[Dict[str, Any]], output_path: Path
) -> None:
    """Создает график ROC кривых."""
    print("  Создание ROC кривых...")

    plt.figure(figsize=(10, 8))

    for result in validation_results:
        fpr, tpr, _ = result["roc_curve"]
        auc = result["metrics"]["roc_auc"]
        plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {auc:.3f})")

    # Диагональ — случайный классификатор
    plt.plot([0, 1], [0, 1], "k--", label="Случайный классификатор")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривые: Сравнение моделей")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plot_path = output_path / "validation_roc_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"    ROC кривые сохранены: {plot_path}")


def create_pr_curves_plot(
    validation_results: List[Dict[str, Any]], output_path: Path
) -> None:
    """Создает график Precision-Recall кривых."""
    print("  Создание Precision-Recall кривых...")

    plt.figure(figsize=(10, 8))

    for result in validation_results:
        precision, recall, _ = result["pr_curve"]
        ap = result["metrics"]["average_precision"]
        plt.plot(recall, precision, label=f"{result['model_name']} (AP = {ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall кривые: Сравнение моделей")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plot_path = output_path / "validation_pr_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"    PR кривые сохранены: {plot_path}")


def create_confusion_matrices_plot(
    validation_results: List[Dict[str, Any]], output_path: Path
) -> None:
    """Создает график матриц ошибок."""
    print("  Создание матриц ошибок...")

    n_models = len(validation_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for i, result in enumerate(validation_results):
        cm = result["confusion_matrix"]

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i])
        axes[i].set_title(f"{result['model_name']}")
        axes[i].set_xlabel("Предсказанный класс")
        axes[i].set_ylabel("Истинный класс")

    plt.suptitle("Матрицы ошибок", fontsize=16)
    plt.tight_layout()

    plot_path = output_path / "validation_confusion_matrices.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"    Матрицы ошибок сохранены: {plot_path}")


def generate_validation_report(
    validation_results: List[Dict[str, Any]], output_dir: str = "models/artifacts"
) -> None:
    """
    Генерирует детальный отчет о валидации.

    Args:
        validation_results: Результаты валидации
        output_dir: Папка для сохранения
    """
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
        row = {"model_name": result["model_name"]}
        row.update(result["metrics"])
        report_data.append(row)

    # Создаем DataFrame
    report_df = pd.DataFrame(report_data)
    report_df = report_df.sort_values("roc_auc", ascending=False)

    # Сохраняем отчет
    report_path = output_path / "validation_report.csv"
    report_df.to_csv(report_path, index=False)

    # Создаем текстовый отчет
    text_report_path = output_path / "validation_report.txt"
    with open(text_report_path, "w", encoding="utf-8") as f:
        f.write("ОТЧЕТ О ВАЛИДАЦИИ МОДЕЛЕЙ КРЕДИТНОГО СКОРИНГА\n")
        f.write("=" * 60 + "\n\n")

        f.write("СВОДНАЯ ТАБЛИЦА МЕТРИК:\n")
        f.write("-" * 40 + "\n")
        f.write(report_df.to_string(index=False, float_format="%.4f"))
        f.write("\n\n")

        f.write("ЛУЧШАЯ МОДЕЛЬ:\n")
        f.write("-" * 40 + "\n")
        best_model = report_df.iloc[0]
        f.write(f"Название: {best_model['model_name']}\n")
        f.write(f"ROC-AUC: {best_model['roc_auc']:.4f}\n")
        f.write(f"Accuracy: {best_model['accuracy']:.4f}\n")
        f.write(f"Precision: {best_model['precision']:.4f}\n")
        f.write(f"Recall: {best_model['recall']:.4f}\n")
        f.write(f"F1-score: {best_model['f1']:.4f}\n")
        f.write(f"Average Precision: {best_model['average_precision']:.4f}\n")
        f.write(f"Log Loss: {best_model['log_loss']:.4f}\n")

        f.write("\n\nДЕТАЛЬНЫЙ АНАЛИЗ КАЖДОЙ МОДЕЛИ:\n")
        f.write("-" * 40 + "\n")

        for result in valid_results:
            f.write(f"\n{result['model_name']}:\n")
            f.write(f"  Accuracy: {result['metrics']['accuracy']:.4f}\n")
            f.write(f"  Precision: {result['metrics']['precision']:.4f}\n")
            f.write(f"  Recall: {result['metrics']['recall']:.4f}\n")
            f.write(f"  F1-score: {result['metrics']['f1']:.4f}\n")
            f.write(f"  ROC-AUC: {result['metrics']['roc_auc']:.4f}\n")
            f.write(
                f"  Average Precision: {result['metrics']['average_precision']:.4f}\n"
            )
            f.write(f"  Log Loss: {result['metrics']['log_loss']:.4f}\n")
            f.write(f"  Specificity: {result['metrics']['specificity']:.4f}\n")
            f.write(f"  Sensitivity: {result['metrics']['sensitivity']:.4f}\n")
            f.write(
                f"  False Positive Rate: {result['metrics']['false_positive_rate']:.4f}\n"
            )
            f.write(
                f"  False Negative Rate: {result['metrics']['false_negative_rate']:.4f}\n"
            )

    print(f"Отчет сохранен:")
    print(f"  CSV: {report_path}")
    print(f"  TXT: {text_report_path}")


def print_validation_summary(validation_results: List[Dict[str, Any]]) -> None:
    """
    Выводит краткую сводку результатов валидации.

    Args:
        validation_results: Результаты валидации
    """
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
    # Загружаем модели и данные
    models, X_train, X_test, y_train, y_test = load_models_and_data()

    if not models:
        print("Не найдено моделей для валидации")
        return

    # Валидируем каждую модель
    validation_results = []

    for model_name, model in models.items():
        # Валидация на тестовых данных
        result = validate_single_model(model_name, model, X_test, y_test)
        validation_results.append(result)

        # Кросс-валидация (опционально)
        if "error" not in result:
            cv_results = cross_validate_model(model, X_train, y_train)
            if "error" not in cv_results:
                result["cross_validation"] = cv_results

    # Создаем графики
    create_validation_plots(validation_results)

    # Генерируем отчет
    generate_validation_report(validation_results)

    # Выводим сводку
    print_validation_summary(validation_results)

    print("\n" + "=" * 60)
    print("ВАЛИДАЦИЯ МОДЕЛЕЙ ЗАВЕРШЕНА УСПЕШНО")
    print("=" * 60)


if __name__ == "__main__":
    main()
