#!/usr/bin/env python3
"""
Скрипт подготовки данных для системы кредитного скоринга.
Этот скрипт обрабатывает сырые данные и подготавливает их для обучения моделей.
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Обработчик данных для кредитного скоринга."""

    def __init__(self, data_path: str, output_path: str):
        """Инициализация обработчика данных."""
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.scalers = {}
        self.encoders = {}

        # Создание выходных директорий
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "processed").mkdir(exist_ok=True)
        (self.output_path / "artifacts").mkdir(exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Загрузка сырых данных."""
        logger.info(f"Загрузка данных из {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Файл данных не найден: {self.data_path}")

        try:
            if self.data_path.suffix == ".csv":
                df = pd.read_csv(self.data_path, low_memory=False)
            elif self.data_path.suffix == ".parquet":
                df = pd.read_parquet(self.data_path)
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {self.data_path.suffix}")
        except Exception as e:
            raise Exception(f"Ошибка загрузки данных: {e}")

        logger.info(f"Загружено {len(df)} строк и {len(df.columns)} колонок")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка набора данных."""
        logger.info("Очистка данных...")

        # Удаление дубликатов
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        if removed_duplicates > 0:
            logger.info(f"Удалено {removed_duplicates} дублирующихся строк")

        # Обработка пропущенных значений
        missing_before = df.isnull().sum().sum()

        # Заполнение пропущенных значений в зависимости от типа колонки
        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                # Используем медиану для числовых колонок, только если есть непустые значения
                if not df[col].isnull().all():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                else:
                    df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("неизвестно")

        missing_after = df.isnull().sum().sum()
        if missing_before - missing_after > 0:
            logger.info(f"Заполнено {missing_before - missing_after} пропущенных значений")

        return df

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание бинарной целевой переменной."""
        logger.info("Создание целевой переменной...")

        # Для датасета UCI Credit Card используем существующую целевую колонку
        if "default.payment.next.month" in df.columns:
            df["target"] = df["default.payment.next.month"]
            logger.info("Используется существующая целевая колонка 'default.payment.next.month'")
        elif "loan_status" in df.columns:
            # Маппинг статуса кредита в бинарную целевую переменную
            def map_loan_status(status):
                if pd.isna(status):
                    return 1  # По умолчанию плохой кредит для пропущенных значений

                status_str = str(status).strip().lower()

                # Хорошие кредиты
                good_indicators = ["fully paid", "current", "good loan"]
                if any(good in status_str for good in good_indicators):
                    return 0
                # Плохие кредиты
                else:
                    return 1

            df["target"] = df["loan_status"].apply(map_loan_status)
            logger.info("Создана целевая переменная из колонки 'loan_status'")
        else:
            raise ValueError("Не найдена подходящая целевая колонка в данных")

        # Логирование распределения целевой переменной
        target_dist = df["target"].value_counts()
        logger.info(f"Распределение целевой переменной: {target_dist.to_dict()}")

        # Проверка наличия обоих классов
        if len(target_dist) < 2:
            logger.warning(f"Присутствует только один класс в целевой переменной: {target_dist.index[0]}")

        return df

    def engineer_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание кастомных признаков для кредитного скоринга (6 признаков из заявки)."""
        logger.info("Создание кастомных признаков...")

        # Используем только 6 признаков из заявки
        custom_features = df[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0']].copy()

        # Переименование колонок для единообразия
        custom_features = custom_features.rename(columns={
            'LIMIT_BAL': 'limit_bal',
            'SEX': 'sex',
            'EDUCATION': 'education_new',
            'MARRIAGE': 'marriage_new',
            'AGE': 'age',
            'PAY_0': 'pay_new'
        })

        # Обработка категориальных признаков
        # Пол: 1=мужской, 2=женский (оставляем как есть)
        # Семейное положение: перекодируем в 0=неизвестно, 1=женат/замужем, 2=не женат/не замужем, 3=другое
        custom_features['marriage_new'] = custom_features['marriage_new'].map({
            0: 0,  # неизвестно
            1: 1,  # женат/замужем
            2: 2,  # не женат/не замужем
            3: 3  # другое
        }).fillna(0)  # все остальные значения маппим в неизвестно

        # Образование: перекодируем в 1=аспирантура, 2=университет, 3=средняя школа, 4=другое
        custom_features['education_new'] = custom_features['education_new'].map({
            1: 1,  # аспирантура
            2: 2,  # университет
            3: 3,  # средняя школа
            4: 4,  # другое
            5: 0,  # неизвестно
            6: 0  # неизвестно
        }).fillna(0)  # все остальные значения маппим в неизвестно

        # Статус платежей: оставляем как есть (-2, -1, 0, 1, 2, ...) но обеспечиваем правильный формат
        custom_features['pay_new'] = custom_features['pay_new'].astype(int)

        logger.info(f"Создано {len(custom_features.columns)} кастомных признаков")
        return custom_features

    def select_custom_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Выбор только 6 кастомных признаков из заявки."""
        logger.info("Выбор кастомных признаков...")

        # Получаем кастомные признаки
        X_custom = self.engineer_custom_features(df)
        y = df["target"].copy()

        logger.info(f"Выбрано {len(X_custom.columns)} кастомных признаков: {list(X_custom.columns)}")
        logger.info(f"Размер целевой переменной: {y.shape}")

        return X_custom, y

    def preprocess_custom_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Предобработка кастомных признаков - БЕЗ МАСШТАБИРОВАНИЯ для tree-based моделей."""
        logger.info("Предобработка кастомных признаков (без масштабирования для tree-based моделей)...")

        X_processed = X.copy()

        # Для кастомных признаков не применяем масштабирование, потому что:
        # - Tree-based модели (Random Forest, CatBoost) не требуют масштабирования
        # - Мы хотим сохранить исходные распределения признаков
        # - Только Logistic Regression может выиграть, но мы обработаем это отдельно

        logger.info("Масштабирование признаков пропущено (оптимально для tree-based моделей)")

        return X_processed

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        """
        Разделение данных на обучающую и тестовую выборки с сохранением распределения целевой переменной.
        """
        logger.info("Разделение данных на обучающую и тестовую выборки...")

        # Проверка распределения целевой переменной
        target_distribution = y.value_counts().sort_index()
        logger.info(f"Распределение целевой переменной: {target_distribution.to_dict()}")

        # Используем стратификацию только если есть оба класса
        if len(target_distribution) >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            logger.info("Использовано стратифицированное разделение")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=None
            )
            logger.info("Использовано случайное разделение (только один класс в целевой переменной)")

        # Логирование результатов разделения
        train_good_pct = (y_train == 0).sum() / len(y_train) * 100 if len(y_train) > 0 else 0
        train_bad_pct = (y_train == 1).sum() / len(y_train) * 100 if len(y_train) > 0 else 0
        test_good_pct = (y_test == 0).sum() / len(y_test) * 100 if len(y_test) > 0 else 0
        test_bad_pct = (y_test == 1).sum() / len(y_test) * 100 if len(y_test) > 0 else 0

        logger.info(f"Обучающая выборка: {train_good_pct:.1f}% хороших, {train_bad_pct:.1f}% плохих")
        logger.info(f"Тестовая выборка: {test_good_pct:.1f}% хороших, {test_bad_pct:.1f}% плохих")
        logger.info(f"Размер обучающей выборки: {len(X_train):,}")
        logger.info(f"Размер тестовой выборки: {len(X_test):,}")

        return X_train, X_test, y_train, y_test

    def save_processed_data(
            self,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame,
            y_train: pd.Series,
            y_test: pd.Series,
    ):
        """Сохранение обработанных данных."""
        logger.info("Сохранение обработанных данных...")

        try:
            # Сохранение данных
            X_train.to_csv(self.output_path / "processed" / "X_train.csv", index=False)
            X_test.to_csv(self.output_path / "processed" / "X_test.csv", index=False)
            y_train.to_csv(self.output_path / "processed" / "y_train.csv", index=False)
            y_test.to_csv(self.output_path / "processed" / "y_test.csv", index=False)

            # Сохранение информации о признаках для справки
            feature_info = {
                'application_features': list(X_train.columns),
                'feature_descriptions': {
                    'limit_bal': 'Кредитный лимит в TWD',
                    'sex': 'Пол (1=мужской, 2=женский)',
                    'marriage_new': 'Семейное положение (0=неизвестно, 1=женат/замужем, 2=не женат/не замужем, 3=другое)',
                    'age': 'Возраст в годах',
                    'pay_new': 'Статус платежей (-2=не использовался, -1=оплачен вовремя, 0=револьверный кредит, 1=задержка 1 месяц, ...)',
                    'education_new': 'Уровень образования (1=аспирантура, 2=университет, 3=средняя школа, 4=другое)'
                }
            }
            joblib.dump(feature_info, self.output_path / "artifacts" / "feature_info.pkl")

            logger.info("Обработанные данные успешно сохранены")

        except Exception as e:
            logger.error(f"Ошибка сохранения обработанных данных: {e}")
            raise

    def process(self):
        """Основной пайплайн обработки данных."""
        logger.info("Запуск пайплайна обработки данных...")

        try:
            # Загрузка данных
            df = self.load_data()

            # Очистка данных
            df = self.clean_data(df)

            # Создание целевой переменной
            df = self.create_target_variable(df)

            # Выбор кастомных признаков (6 признаков из заявки)
            X, y = self.select_custom_features(df)

            # Предобработка признаков (БЕЗ МАСШТАБИРОВАНИЯ)
            X_processed = self.preprocess_custom_features(X, fit=True)

            # Разделение данных
            X_train, X_test, y_train, y_test = self.split_data(X_processed, y)

            # Сохранение обработанных данных
            self.save_processed_data(X_train, X_test, y_train, y_test)

            logger.info("Пайплайн обработки данных успешно завершен!")
            logger.info(f"Использовано {len(X_processed.columns)} кастомных признаков из заявки")
            logger.info("Масштабирование признаков не применялось (оптимально для tree-based моделей)")

        except Exception as e:
            logger.error(f"Пайплайн обработки данных завершился ошибкой: {e}")
            raise


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(
        description="Подготовка данных для модели кредитного скоринга"
    )
    parser.add_argument("--data-path", required=True, help="Путь к файлу с сырыми данными")
    parser.add_argument(
        "--output-path", required=True, help="Путь для сохранения обработанных данных"
    )

    args = parser.parse_args()

    try:
        # Создание обработчика данных
        processor = DataProcessor(args.data_path, args.output_path)

        # Обработка данных
        processor.process()

    except Exception as e:
        logger.error(f"Выполнение скрипта завершилось ошибкой: {e}")
        raise


if __name__ == "__main__":
    import sys

    main()