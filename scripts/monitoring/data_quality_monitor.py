"""
Скрипт для мониторинга качества данных в проекте кредитного скоринга.

Этот скрипт выполняет:
1. Мониторинг пропущенных значений
2. Мониторинг выбросов
3. Мониторинг распределений
4. Мониторинг целостности данных
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Импорты для статистического анализа
from scipy import stats
from scipy.stats import zscore

warnings.filterwarnings("ignore")

# Настройка логирования
# Создаем папку logs, если её нет
import os
from pathlib import Path

# Создаем папку logs относительно корня проекта
project_root = Path(__file__).parent.parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(logs_dir / "data_quality_monitoring.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """
    Класс для мониторинга качества данных.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация монитора качества данных.

        Args:
            config: Конфигурация мониторинга
        """
        self.config = config or self._default_config()
        logger.info("Монитор качества данных инициализирован")

    def _default_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию по умолчанию."""
        return {
            "missing_threshold": 0.1,  # Порог для пропущенных значений (10%)
            "outlier_threshold": 3.0,  # Порог для выбросов (z-score)
            "correlation_threshold": 0.95,  # Порог для высокой корреляции
            "duplicate_threshold": 0.05,  # Порог для дубликатов (5%)
            "min_unique_ratio": 0.1,  # Минимальное соотношение уникальных значений
            "max_skewness": 2.0,  # Максимальная асимметрия
            "max_kurtosis": 3.0,  # Максимальный эксцесс
        }

    def check_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Проверяет пропущенные значения в данных.

        Args:
            data: DataFrame для анализа

        Returns:
            Результаты проверки пропущенных значений
        """
        logger.info("Проверка пропущенных значений...")

        missing_info = {
            "total_missing": data.isnull().sum().sum(),
            "missing_percentage": (
                data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            )
            * 100,
            "columns_with_missing": {},
            "critical_columns": [],
            "timestamp": datetime.now().isoformat(),
        }

        try:
            missing_counts = data.isnull().sum()
            missing_percentages = (missing_counts / len(data)) * 100

            for col in data.columns:
                missing_count = missing_counts[col]
                missing_pct = missing_percentages[col]

                missing_info["columns_with_missing"][col] = {
                    "count": int(missing_count),
                    "percentage": round(missing_pct, 2),
                }

                # Критические колонки с высоким процентом пропусков
                if missing_pct > self.config["missing_threshold"] * 100:
                    missing_info["critical_columns"].append(
                        {"column": col, "missing_percentage": round(missing_pct, 2)}
                    )

            logger.info(
                f"Пропущенных значений: {missing_info['total_missing']} "
                f"({missing_info['missing_percentage']:.2f}%)"
            )

        except Exception as e:
            logger.error(f"Ошибка при проверке пропущенных значений: {e}")
            missing_info["error"] = str(e)

        return missing_info

    def check_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Проверяет выбросы в числовых данных.

        Args:
            data: DataFrame для анализа

        Returns:
            Результаты проверки выбросов
        """
        logger.info("Проверка выбросов...")

        outlier_info = {
            "columns_with_outliers": {},
            "total_outliers": 0,
            "critical_columns": [],
            "timestamp": datetime.now().isoformat(),
        }

        try:
            numeric_columns = data.select_dtypes(include=[np.number]).columns

            for col in numeric_columns:
                if data[col].notna().sum() > 0:  # Проверяем, что есть непустые значения
                    # Z-score метод
                    z_scores = np.abs(zscore(data[col].dropna()))
                    outliers = z_scores > self.config["outlier_threshold"]

                    outlier_count = outliers.sum()
                    outlier_percentage = (outlier_count / len(data[col].dropna())) * 100

                    outlier_info["columns_with_outliers"][col] = {
                        "count": int(outlier_count),
                        "percentage": round(outlier_percentage, 2),
                        "mean_z_score": round(z_scores.mean(), 2),
                        "max_z_score": round(z_scores.max(), 2),
                    }

                    outlier_info["total_outliers"] += outlier_count

                    # Критические колонки с высоким процентом выбросов
                    if outlier_percentage > 5:  # Более 5% выбросов
                        outlier_info["critical_columns"].append(
                            {
                                "column": col,
                                "outlier_percentage": round(outlier_percentage, 2),
                            }
                        )

            logger.info(f"Общее количество выбросов: {outlier_info['total_outliers']}")

        except Exception as e:
            logger.error(f"Ошибка при проверке выбросов: {e}")
            outlier_info["error"] = str(e)

        return outlier_info

    def check_data_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Проверяет распределения данных.

        Args:
            data: DataFrame для анализа

        Returns:
            Результаты проверки распределений
        """
        logger.info("Проверка распределений данных...")

        distribution_info = {
            "columns_distribution": {},
            "skewed_columns": [],
            "high_kurtosis_columns": [],
            "low_variance_columns": [],
            "timestamp": datetime.now().isoformat(),
        }

        try:
            numeric_columns = data.select_dtypes(include=[np.number]).columns

            for col in numeric_columns:
                if data[col].notna().sum() > 10:  # Минимум 10 значений
                    values = data[col].dropna()

                    # Основные статистики
                    stats_info = {
                        "mean": round(values.mean(), 4),
                        "std": round(values.std(), 4),
                        "min": round(values.min(), 4),
                        "max": round(values.max(), 4),
                        "skewness": round(stats.skew(values), 4),
                        "kurtosis": round(stats.kurtosis(values), 4),
                        "variance": round(values.var(), 4),
                    }

                    distribution_info["columns_distribution"][col] = stats_info

                    # Проверяем асимметрию
                    if abs(stats_info["skewness"]) > self.config["max_skewness"]:
                        distribution_info["skewed_columns"].append(
                            {"column": col, "skewness": stats_info["skewness"]}
                        )

                    # Проверяем эксцесс
                    if abs(stats_info["kurtosis"]) > self.config["max_kurtosis"]:
                        distribution_info["high_kurtosis_columns"].append(
                            {"column": col, "kurtosis": stats_info["kurtosis"]}
                        )

                    # Проверяем низкую дисперсию
                    if stats_info["variance"] < 0.01:  # Очень низкая дисперсия
                        distribution_info["low_variance_columns"].append(
                            {"column": col, "variance": stats_info["variance"]}
                        )

            logger.info(f"Проверено {len(numeric_columns)} числовых колонок")

        except Exception as e:
            logger.error(f"Ошибка при проверке распределений: {e}")
            distribution_info["error"] = str(e)

        return distribution_info

    def check_data_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Проверяет целостность данных.

        Args:
            data: DataFrame для анализа

        Returns:
            Результаты проверки целостности
        """
        logger.info("Проверка целостности данных...")

        integrity_info = {
            "duplicate_rows": data.duplicated().sum(),
            "duplicate_percentage": (data.duplicated().sum() / len(data)) * 100,
            "duplicate_columns": [],
            "constant_columns": [],
            "highly_correlated_columns": [],
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Проверяем дубликаты строк
            logger.info(
                f"Дубликатов строк: {integrity_info['duplicate_rows']} "
                f"({integrity_info['duplicate_percentage']:.2f}%)"
            )

            # Проверяем дубликаты колонок
            for i, col1 in enumerate(data.columns):
                for j, col2 in enumerate(data.columns[i + 1 :], i + 1):
                    if data[col1].equals(data[col2]):
                        integrity_info["duplicate_columns"].append(
                            {"column1": col1, "column2": col2}
                        )

            # Проверяем константные колонки
            for col in data.columns:
                if data[col].nunique() <= 1:
                    integrity_info["constant_columns"].append(col)

            # Проверяем высокую корреляцию
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr().abs()

                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        corr_value = corr_matrix.iloc[i, j]

                        if corr_value > self.config["correlation_threshold"]:
                            integrity_info["highly_correlated_columns"].append(
                                {
                                    "column1": col1,
                                    "column2": col2,
                                    "correlation": round(corr_value, 4),
                                }
                            )

            logger.info(
                f"Константных колонок: {len(integrity_info['constant_columns'])}"
            )
            logger.info(
                f"Высоко коррелированных пар: {len(integrity_info['highly_correlated_columns'])}"
            )

        except Exception as e:
            logger.error(f"Ошибка при проверке целостности: {e}")
            integrity_info["error"] = str(e)

        return integrity_info

    def check_categorical_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Проверяет категориальные данные.

        Args:
            data: DataFrame для анализа

        Returns:
            Результаты проверки категориальных данных
        """
        logger.info("Проверка категориальных данных...")

        categorical_info = {
            "categorical_columns": {},
            "high_cardinality_columns": [],
            "low_cardinality_columns": [],
            "timestamp": datetime.now().isoformat(),
        }

        try:
            categorical_columns = data.select_dtypes(
                include=["object", "category"]
            ).columns

            for col in categorical_columns:
                unique_count = data[col].nunique()
                unique_ratio = unique_count / len(data)

                categorical_info["categorical_columns"][col] = {
                    "unique_count": unique_count,
                    "unique_ratio": round(unique_ratio, 4),
                    "most_common": (
                        data[col].mode().iloc[0] if not data[col].mode().empty else None
                    ),
                    "most_common_count": (
                        data[col].value_counts().iloc[0] if not data[col].empty else 0
                    ),
                }

                # Высокая кардинальность (много уникальных значений)
                if unique_ratio > 0.5:
                    categorical_info["high_cardinality_columns"].append(
                        {"column": col, "unique_ratio": round(unique_ratio, 4)}
                    )

                # Низкая кардинальность (мало уникальных значений)
                if unique_ratio < self.config["min_unique_ratio"]:
                    categorical_info["low_cardinality_columns"].append(
                        {"column": col, "unique_ratio": round(unique_ratio, 4)}
                    )

            logger.info(f"Проверено {len(categorical_columns)} категориальных колонок")

        except Exception as e:
            logger.error(f"Ошибка при проверке категориальных данных: {e}")
            categorical_info["error"] = str(e)

        return categorical_info

    def generate_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Генерирует полный отчет о качестве данных.

        Args:
            data: DataFrame для анализа

        Returns:
            Полный отчет о качестве данных
        """
        logger.info("Генерация отчета о качестве данных...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "data_shape": data.shape,
            "data_types": data.dtypes.to_dict(),
            "overall_quality_score": 0.0,
            "issues": [],
            "recommendations": [],
        }

        try:
            # Выполняем все проверки
            missing_check = self.check_missing_values(data)
            outlier_check = self.check_outliers(data)
            distribution_check = self.check_data_distribution(data)
            integrity_check = self.check_data_integrity(data)
            categorical_check = self.check_categorical_data(data)

            # Добавляем результаты в отчет
            report["missing_values"] = missing_check
            report["outliers"] = outlier_check
            report["distributions"] = distribution_check
            report["data_integrity"] = integrity_check
            report["categorical_data"] = categorical_check

            # Вычисляем общий скор качества
            quality_score = 100.0

            # Штрафы за проблемы
            if missing_check.get("critical_columns"):
                quality_score -= len(missing_check["critical_columns"]) * 5
                report["issues"].append(
                    f"Критические колонки с пропусками: {len(missing_check['critical_columns'])}"
                )

            if outlier_check.get("critical_columns"):
                quality_score -= len(outlier_check["critical_columns"]) * 3
                report["issues"].append(
                    f"Колонки с выбросами: {len(outlier_check['critical_columns'])}"
                )

            if integrity_check.get("duplicate_percentage", 0) > 5:
                quality_score -= 10
                report["issues"].append(
                    f"Высокий процент дубликатов: {integrity_check['duplicate_percentage']:.2f}%"
                )

            if integrity_check.get("constant_columns"):
                quality_score -= len(integrity_check["constant_columns"]) * 2
                report["issues"].append(
                    f"Константные колонки: {len(integrity_check['constant_columns'])}"
                )

            report["overall_quality_score"] = max(0, quality_score)

            # Генерируем рекомендации
            if missing_check.get("critical_columns"):
                report["recommendations"].append(
                    "Рассмотрите удаление или заполнение колонок с высоким процентом пропусков"
                )

            if outlier_check.get("critical_columns"):
                report["recommendations"].append(
                    "Проверьте и обработайте выбросы в данных"
                )

            if integrity_check.get("duplicate_rows", 0) > 0:
                report["recommendations"].append("Удалите дубликаты строк")

            if integrity_check.get("constant_columns"):
                report["recommendations"].append("Удалите константные колонки")

            if integrity_check.get("highly_correlated_columns"):
                report["recommendations"].append(
                    "Рассмотрите удаление высоко коррелированных колонок"
                )

            logger.info(
                f"Отчет о качестве данных сгенерирован. Скор: {report['overall_quality_score']:.1f}"
            )

        except Exception as e:
            logger.error(f"Ошибка при генерации отчета: {e}")
            report["error"] = str(e)
            report["overall_quality_score"] = 0

        return report

    def save_quality_report(self, report: Dict[str, Any], output_path: str) -> None:
        """
        Сохраняет отчет о качестве данных.

        Args:
            report: Отчет о качестве данных
            output_path: Путь для сохранения
        """
        try:
            output_file = (
                Path(output_path)
                / f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Отчет о качестве данных сохранен: {output_file}")

        except Exception as e:
            logger.error(f"Ошибка при сохранении отчета: {e}")


def main():
    """Основная функция для запуска мониторинга качества данных."""
    # Параметры
    data_path = "data/processed/X_train.csv"
    output_path = "monitoring/reports"

    # Проверяем наличие файла данных
    if not Path(data_path).exists():
        logger.error(f"Файл данных не найден: {data_path}")
        return 1

    try:
        # Загружаем данные
        logger.info(f"Загрузка данных из {data_path}")
        data = pd.read_csv(data_path)
        logger.info(f"Загружено {data.shape[0]} строк и {data.shape[1]} колонок")

        # Создаем монитор
        monitor = DataQualityMonitor()

        # Генерируем отчет
        report = monitor.generate_quality_report(data)

        # Сохраняем отчет
        monitor.save_quality_report(report, output_path)

        # Выводим краткую сводку
        print(f"\n ОТЧЕТ О КАЧЕСТВЕ ДАННЫХ")
        print(f"Время: {report['timestamp']}")
        print(f"Размер данных: {report['data_shape']}")
        print(f"Скор качества: {report['overall_quality_score']:.1f}/100")
        print(f"Проблем: {len(report.get('issues', []))}")

        if report.get("issues"):
            print("\n  ПРОБЛЕМЫ:")
            for issue in report["issues"]:
                print(f"  • {issue}")

        if report.get("recommendations"):
            print("\n РЕКОМЕНДАЦИИ:")
            for rec in report["recommendations"]:
                print(f"  • {rec}")

        return 0 if report["overall_quality_score"] > 70 else 1

    except Exception as e:
        logger.error(f"Ошибка в main: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
