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
    """Создает график сравнения метрик без Tkinter."""
    print("  Создание графика сравнения метрик...")

    # Используем неинтерактивный бэкенд
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        # Подготавливаем данные
        model_names = [r["model_name"] for r in validation_results]
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        metric_values = {metric: [] for metric in metrics}

        for result in validation_results:
            for metric in metrics:
                metric_values[metric].append(result["metrics"][metric])

        # Создаем упрощенный график
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Используем только ROC-AUC для сравнения
        roc_auc_scores = metric_values['roc_auc']

        # Создаем bar plot
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightpink', 'lightcyan']
        bars = ax.bar(model_names, roc_auc_scores, color=colors[:len(model_names)])
        ax.set_ylabel('ROC-AUC Score')
        ax.set_title('Сравнение ROC-AUC моделей')
        ax.set_ylim(0, 1)

        # Добавляем значения на столбцы
        for bar, score in zip(bars, roc_auc_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=9)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plot_path = output_path / "validation_metrics_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"    График сохранен: {plot_path}")

    except Exception as e:
        print(f"    Ошибка при создании графика сравнения метрик: {e}")


def create_roc_curves_plot(
        validation_results: List[Dict[str, Any]], output_path: Path
) -> None:
    """Создает график ROC кривых без Tkinter."""
    print("  Создание ROC кривых...")

    # Используем неинтерактивный бэкенд
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc

    try:
        plt.figure(figsize=(10, 8))

        for result in validation_results:
            fpr, tpr, _ = result["roc_curve"]
            roc_auc = result["metrics"]["roc_auc"]
            plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {roc_auc:.3f})")

        # Диагональ — случайный классификатор
        plt.plot([0, 1], [0, 1], "k--", label="Случайный классификатор")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC-кривые: Сравнение моделей")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plot_path = output_path / "validation_roc_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"    ROC кривые сохранены: {plot_path}")

    except Exception as e:
        print(f"    Ошибка при создании ROC кривых: {e}")


def create_pr_curves_plot(
        validation_results: List[Dict[str, Any]], output_path: Path
) -> None:
    """Создает график Precision-Recall кривых без Tkinter."""
    print("  Создание Precision-Recall кривых...")

    # Используем неинтерактивный бэкенд
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
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
        plt.close()

        print(f"    PR кривые сохранены: {plot_path}")

    except Exception as e:
        print(f"    Ошибка при создании PR кривых: {e}")


def create_confusion_matrices_plot(
        validation_results: List[Dict[str, Any]], output_path: Path
) -> None:
    """Создает график матриц ошибок без Tkinter."""
    print("  Создание матриц ошибок...")

    # Используем неинтерактивный бэкенд
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    try:
        n_models = len(validation_results)

        # Для 1-2 моделей создаем горизонтальное расположение, для большего - вертикальное
        if n_models <= 2:
            fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        else:
            fig, axes = plt.subplots(n_models, 1, figsize=(6, 4 * n_models))

        if n_models == 1:
            axes = [axes]

        for i, result in enumerate(validation_results):
            cm = result["confusion_matrix"]

            if n_models <= 2:
                ax = axes[i]
            else:
                ax = axes[i]

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"{result['model_name']}")
            ax.set_xlabel("Предсказанный класс")
            ax.set_ylabel("Истинный класс")

        plt.tight_layout()

        plot_path = output_path / "validation_confusion_matrices.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"    Матрицы ошибок сохранены: {plot_path}")

    except Exception as e:
        print(f"    Ошибка при создании матриц ошибок: {e}")


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


def create_data_validation_suite():
    """
    Создает набор ожиданий (expectations) для валидации данных кредитного скоринга.
    """
    try:
        import great_expectations as gx
        from great_expectations.core.expectation_configuration import ExpectationConfiguration

        print("\nСоздание набора валидации данных Great Expectations...")

        # Создаем файловый контекст вместо ephemeral
        context = gx.get_context(mode="file")

        # Создание набора ожиданий
        expectation_suite_name = "credit_scoring_data_suite"

        # Удаляем существующий набор, если есть
        try:
            context.delete_expectation_suite(expectation_suite_name)
            print(f"Удален существующий набор '{expectation_suite_name}'")
        except:
            pass  # Набора не существует, это нормально

        # Создаем новый набор
        suite = context.add_expectation_suite(expectation_suite_name)
        print(f"Создан новый набор '{expectation_suite_name}'")

        # Определение ожиданий для данных
        expectations = [
            # Базовые проверки существования колонок
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "LIMIT_BAL"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "AGE"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": "target"}
            ),

            # Проверки на отсутствие NULL значений в ключевых колонках
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "LIMIT_BAL"}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "target"}
            ),

            # Проверки диапазонов значений
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "AGE", "min_value": 18, "max_value": 100}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "LIMIT_BAL", "min_value": 0, "max_value": 1000000}
            ),

            # Проверки категориальных значений
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={"column": "target", "value_set": [0, 1]}
            ),
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={"column": "SEX", "value_set": [1, 2]}
            ),

            # Дополнительные проверки для надежности
            ExpectationConfiguration(
                expectation_type="expect_table_row_count_to_be_between",
                kwargs={"min_value": 1, "max_value": 1000000}
            )
        ]

        # Добавление ожиданий в набор
        for exp in expectations:
            suite.add_expectation(exp, send_usage_event=False)

        # Сохранение набора
        context.save_expectation_suite(suite)

        # Проверяем, что набор действительно сохранен
        saved_suite = context.get_expectation_suite(expectation_suite_name)
        print(f"Проверка: набор '{expectation_suite_name}' успешно сохранен")
        print(f"Количество ожиданий в наборе: {len(saved_suite.expectations)}")

        print("OK Набор валидации данных создан успешно!")
        return saved_suite

    except ImportError:
        print("ERROR: Great Expectations не установлен!")
        print("Установите: pip install great-expectations")
        raise
    except Exception as e:
        print(f"ERROR: Ошибка при создании набора валидации: {e}")
        raise


def validate_data_quality(df: pd.DataFrame, suite_name: str = "credit_scoring_data_suite"):
    """
    Выполняет валидацию качества данных с помощью Great Expectations.
    """
    try:
        import great_expectations as gx
        from great_expectations.core.expectation_configuration import ExpectationConfiguration

        print(f"\nВалидация качества данных с помощью {suite_name}...")

        # Используем файловый контекст
        context = gx.get_context(mode="file")

        # Создаем набор ожиданий непосредственно перед валидацией
        try:
            context.delete_expectation_suite(suite_name)
        except:
            pass

        suite = context.add_expectation_suite(suite_name)

        # Определение ожиданий для данных
        expectations = [
            ExpectationConfiguration("expect_column_to_exist", {"column": "LIMIT_BAL"}),
            ExpectationConfiguration("expect_column_to_exist", {"column": "AGE"}),
            ExpectationConfiguration("expect_column_to_exist", {"column": "target"}),
            ExpectationConfiguration("expect_column_values_to_not_be_null", {"column": "LIMIT_BAL"}),
            ExpectationConfiguration("expect_column_values_to_not_be_null", {"column": "target"}),
            ExpectationConfiguration("expect_column_values_to_be_between",
                                     {"column": "AGE", "min_value": 18, "max_value": 100}),
            ExpectationConfiguration("expect_column_values_to_be_between",
                                     {"column": "LIMIT_BAL", "min_value": 0, "max_value": 1000000}),
            ExpectationConfiguration("expect_column_values_to_be_in_set", {"column": "target", "value_set": [0, 1]}),
            ExpectationConfiguration("expect_column_values_to_be_in_set", {"column": "SEX", "value_set": [1, 2]}),
        ]

        for exp in expectations:
            suite.add_expectation(exp, send_usage_event=False)

        context.save_expectation_suite(suite)
        print(f"Создан набор валидации с {len(expectations)} ожиданиями")

        # Создаем временный datasource
        datasource_name = "temp_pandas_datasource"

        # Удаляем существующий datasource если есть
        try:
            context.delete_datasource(datasource_name)
        except:
            pass

        # Создаем новый datasource
        datasource = context.sources.add_pandas(datasource_name)

        # Создаем data asset
        data_asset_name = "temp_dataframe_asset"
        data_asset = datasource.add_dataframe_asset(data_asset_name)

        # Создаем batch request
        batch_request = data_asset.build_batch_request(dataframe=df)

        # Создаем валидатор
        validator = context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name,
        )

        # Выполнение валидации
        validation_result = validator.validate()

        # Анализ результатов
        success = validation_result.success
        statistics = validation_result.statistics

        print(f"Результаты валидации:")
        print(f"  Успешно: {success}")
        print(f"  Всего проверок: {statistics.get('evaluated_expectations', 0)}")
        print(f"  Успешных: {statistics.get('successful_expectations', 0)}")
        print(f"  Неудачных: {statistics.get('unsuccessful_expectations', 0)}")

        if not success:
            print("\nWARNING: Обнаружены проблемы с данными:")
            unsuccessful_count = 0
            for result in validation_result.results:
                if not result.success:
                    unsuccessful_count += 1
                    exp_type = result.expectation_config.expectation_type
                    column = result.expectation_config.kwargs.get('column', 'N/A')
                    print(f"  {unsuccessful_count}. {exp_type} (колонка: {column})")
        else:
            # Используем ASCII символ вместо Unicode для совместимости с кодировками
            print("  Все проверки данных пройдены успешно!")

        # Очищаем временный datasource
        try:
            context.delete_datasource(datasource_name)
        except:
            pass

        return validation_result

    except ImportError:
        print("ERROR: Great Expectations не установлен!")
        print("Установите: pip install great-expectations")
        raise
    except Exception as e:
        print(f"ERROR: Ошибка при валидации данных: {e}")
        raise


def generate_data_quality_report(validation_results, output_dir: str = "models/artifacts"):
    """
    Генерирует отчет о качестве данных.
    """
    if not validation_results:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_data = {
        "validation_success": validation_results.success,
        "total_expectations": validation_results.statistics.get("evaluated_expectations", 0),
        "successful_expectations": validation_results.statistics.get("successful_expectations", 0),
        "unsuccessful_expectations": validation_results.statistics.get("unsuccessful_expectations", 0),
        "failed_checks": []
    }

    # Собираем информацию о неудачных проверках
    if hasattr(validation_results, 'results'):
        for result in validation_results.results:
            if not result.success:
                failed_check = {
                    "expectation_type": result.expectation_config.expectation_type,
                    "column": result.expectation_config.kwargs.get("column", "N/A"),
                    "message": str(getattr(result, 'exception_info', {}).get('exception_message', 'N/A'))
                }
                report_data["failed_checks"].append(failed_check)

    # Сохраняем отчет
    report_df = pd.DataFrame([report_data])
    report_path = output_path / "data_quality_report.csv"

    try:
        report_df.to_csv(report_path, index=False, encoding='utf-8')
        print(f"  CSV отчет сохранен: {report_path}")
    except Exception as e:
        print(f"  Ошибка при сохранении CSV отчета: {e}")
        # Пробуем сохранить без проблемных символов
        report_df.to_csv(report_path, index=False, encoding='utf-8', errors='ignore')

    # Создаем текстовый отчет с правильной кодировкой
    text_report_path = output_path / "data_quality_report.txt"
    try:
        with open(text_report_path, "w", encoding='utf-8') as f:
            f.write("ОТЧЕТ О КАЧЕСТВЕ ДАННЫХ\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Результат валидации: {'УСПЕШНО' if report_data['validation_success'] else 'НЕУДАЧА'}\n")
            f.write(f"Всего проверок: {report_data['total_expectations']}\n")
            f.write(f"Успешных: {report_data['successful_expectations']}\n")
            f.write(f"Неудачных: {report_data['unsuccessful_expectations']}\n\n")

            if report_data['failed_checks']:
                f.write("НЕУДАЧНЫЕ ПРОВЕРКИ:\n")
                f.write("-" * 30 + "\n")
                for i, check in enumerate(report_data['failed_checks'], 1):
                    f.write(f"{i}. Тип проверки: {check['expectation_type']}\n")
                    f.write(f"   Колонка: {check['column']}\n")
                    f.write(f"   Сообщение: {check['message']}\n\n")

        print(f"  TXT отчет сохранен: {text_report_path}")
    except Exception as e:
        print(f"  Ошибка при сохранении TXT отчета: {e}")


def main():
    """Основная функция для запуска валидации моделей."""
    # Создаем папки для артефактов
    Path("models/artifacts").mkdir(parents=True, exist_ok=True)
    Path("models/validation").mkdir(parents=True, exist_ok=True)

    # Загружаем модели и данные
    models, X_train, X_test, y_train, y_test = load_models_and_data()

    if not models:
        print("Не найдено моделей для валидации")
        return

    # Валидация качества данных с помощью Great Expectations (обязательная)
    print("\n=== ВАЛИДАЦИЯ КАЧЕСТВА ДАННЫХ ===")

    data_validation_success = False
    data_validation_results = None

    try:
        # Объединяем признаки и целевую переменную для валидации
        train_data = X_train.copy()
        train_data['target'] = y_train.values

        # Выполняем валидацию данных (набор создается внутри функции)
        data_validation_results = validate_data_quality(train_data)
        data_validation_success = True

    except Exception as e:
        print(f"Ошибка при валидации данных: {e}")
        data_validation_success = False

    # Генерируем отчет о качестве данных, если есть результаты
    if data_validation_results:
        try:
            generate_data_quality_report(data_validation_results)
        except Exception as e:
            print(f"Ошибка при генерации отчета о качестве данных: {e}")

    # Проверяем успешность валидации данных
    if data_validation_success and data_validation_results and data_validation_results.success:
        print("\nВалидация данных завершена успешно! Продолжаем с валидацией моделей...")
    else:
        print("\nВалидация данных не прошла. Прерываем выполнение.")
        return

    # Продолжаем с валидацией моделей (только если валидация данных прошла успешно)
    print("\n=== ВАЛИДАЦИЯ МОДЕЛЕЙ ===")

    validation_results = []

    for model_name, model in models.items():
        # Валидация на тестовых данных
        result = validate_single_model(model_name, model, X_test, y_test)
        validation_results.append(result)

        # Кросс-валидация (опционально)
        if "error" not in result:
            try:
                cv_results = cross_validate_model(model, X_train, y_train)
                if "error" not in cv_results:
                    result["cross_validation"] = cv_results
            except Exception as e:
                print(f"Ошибка при кросс-валидации модели {model_name}: {e}")

    # Создаем графики
    try:
        create_validation_plots(validation_results)
    except Exception as e:
        print(f"Ошибка при создании графиков: {e}")

    # Генерируем отчет
    try:
        generate_validation_report(validation_results)
    except Exception as e:
        print(f"Ошибка при генерации отчета: {e}")

    # Выводим сводку
    try:
        print_validation_summary(validation_results)
    except Exception as e:
        print(f"Ошибка при выводе сводки: {e}")

    print("\n" + "=" * 60)
    print("ВАЛИДАЦИЯ МОДЕЛЕЙ ЗАВЕРШЕНА")
    print("=" * 60)


if __name__ == "__main__":
    main()
