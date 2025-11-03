#!/usr/bin/env python3
"""
Скрипт подготовки данных для системы кредитного скоринга.
Обрабатывает сырые данные и подготавливает их для обучения моделей.
Использует только признаки доступные при подаче заявки.
"""

import argparse
import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

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
            df = pd.read_csv(self.data_path, low_memory=False)
        except Exception as e:
            raise Exception(f"Ошибка загрузки данных: {e}")

        logger.info(f"Загружено {len(df)} строк и {len(df.columns)} колонок")
        logger.info(f"Исходные названия колонок: {list(df.columns)}")

        return df

    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Приводит названия колонок к snake_case."""
        logger.info("Нормализация названий колонок...")

        # Создаем mapping для нормализации
        column_mapping = {}
        for col in df.columns:
            # Приводим к нижнему регистру и заменяем не-буквенно-цифровые символы на _
            normalized = re.sub(r"[^a-z0-9]+", "_", str(col).lower().strip())
            # Убираем лишние подчеркивания
            normalized = re.sub(r"_+", "_", normalized).strip("_")
            column_mapping[col] = normalized

        # Переименовываем колонки
        df = df.rename(columns=column_mapping)

        logger.info(f"Нормализованные названия колонок: {list(df.columns)}")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка набора данных."""
        logger.info("Очистка данных...")

        # Удаление дубликатов
        начальное_количество = len(df)
        df = df.drop_duplicates()
        удалено_дубликатов = начальное_количество - len(df)
        if удалено_дубликатов > 0:
            logger.info(f"Удалено {удалено_дубликатов} дублирующихся строк")

        return df

    def find_target_column(self, df: pd.DataFrame) -> str:
        """Автоматический поиск целевой переменной."""
        logger.info("Поиск целевой переменной...")

        # Нормализация названий колонок
        norm = {c: re.sub(r"[^a-z0-9]+", "", c.lower()) for c in df.columns}

        # Паттерны для целевой переменной
        предпочтительные = ["defaultpaymentnextmonth", "default", "target", "y"]

        # 1) Точное совпадение
        for паттерн in предпочтительные:
            for колонка, нормализованное_имя in norm.items():
                if нормализованное_имя == паттерн:
                    logger.info(f"Найдена целевая переменная: {колонка}")
                    return колонка

        # 2) Бинарные колонки
        for колонка in df.columns:
            if df[колонка].dropna().nunique() == 2:
                logger.info(f"Найдена бинарная колонка: {колонка}")
                return колонка

        raise ValueError("Не удалось найти целевую переменную")

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание бинарной целевой переменной."""
        logger.info("Создание целевой переменной...")

        # Поиск целевой колонки
        целевая_колонка = self.find_target_column(df)

        # Проверка, является ли целевая переменная уже бинарной (0/1)
        уникальные_значения = df[целевая_колонка].unique()
        if set(уникальные_значения).issubset({0, 1}):
            logger.info("Целевая переменная уже бинарная (0/1), используем как есть")
            df["target"] = df[целевая_колонка].astype(int)
        else:
            # Простой маппинг для числовых значений
            df["target"] = (df[целевая_колонка] > 0).astype(int)

        # Логирование распределения
        распределение = df["target"].value_counts()
        logger.info(f"Распределение целевой переменной: {распределение.to_dict()}")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков для кредитного скоринга."""
        logger.info("Создание признаков...")

        # Теперь все колонки в snake_case, можно использовать напрямую
        df["education_new"] = np.where(df["education"] >= 4, 4, df["education"])
        logger.info("Создан education_new")

        df["marriage_new"] = np.where(df["marriage"] >= 3, 3, df["marriage"])
        logger.info("Создан marriage_new")

        # Преобразование статусов платежей
        pay_columns = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]

        has_positive_list = []
        has_zero_list = []

        for col in pay_columns:
            if col in df.columns:
                has_positive_list.append(df[col] > 0)
                has_zero_list.append(df[col] == 0)

        if has_positive_list and has_zero_list:
            has_any_positive = pd.concat(has_positive_list, axis=1).any(axis=1)
            has_any_zero = pd.concat(has_zero_list, axis=1).any(axis=1)

            df["pay_new"] = np.where(has_any_positive, 1, np.where(has_any_zero, 0, -1))
            logger.info("Создан pay_new из колонок платежей")
        else:
            df["pay_new"] = 0
            logger.warning(
                "Не найдены колонки платежей, используем значение по умолчанию"
            )

        logger.info("Создание признаков завершено")
        return df

    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Выбор признаков для обучения - только признаки доступные при подаче заявки."""
        logger.info("Выбор признаков доступных при подаче заявки...")

        # Признаки, которые можно получить при подаче заявки (все в snake_case)
        application_features = [
            "limit_bal",
            "sex",
            "marriage_new",
            "age",
            "pay_new",
            "education_new",
        ]

        # Проверка доступных признаков
        available_features = [col for col in application_features if col in df.columns]

        if not available_features:
            raise ValueError("Нет доступных признаков заявки")

        logger.info(
            f"Выбрано {len(available_features)} признаков заявки: {available_features}"
        )

        X = df[available_features].copy()
        y = df["target"].copy()

        return X, y

    def preprocess_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Предобработка признаков для моделирования."""
        logger.info("Предобработка признаков...")

        X_обработанные = X.copy()

        # Масштабирование числовых признаков
        числовые_колонки = X_обработанные.select_dtypes(include=[np.number]).columns

        for колонка in числовые_колонки:
            if fit:
                скалер = StandardScaler()
                if X_обработанные[колонка].nunique() > 1:
                    X_обработанные[колонка] = скалер.fit_transform(
                        X_обработанные[[колонка]]
                    )
                self.scalers[колонка] = скалер

        return X_обработанные

    def split_data(self, X: pd.DataFrame, y: pd.Series):
        """Разделение данных на обучающую и тестовую выборки."""
        logger.info("Разделение данных на обучающую и тестовую выборки...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Обучающая выборка: {len(X_train)} примеров")
        logger.info(f"Тестовая выборка: {len(X_test)} примеров")

        return X_train, X_test, y_train, y_test

    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Сохранение обработанных данных и артефактов."""
        logger.info("Сохранение обработанных данных...")

        # Сохранение данных
        X_train.to_csv(self.output_path / "processed" / "X_train.csv", index=False)
        X_test.to_csv(self.output_path / "processed" / "X_test.csv", index=False)
        y_train.to_csv(self.output_path / "processed" / "y_train.csv", index=False)
        y_test.to_csv(self.output_path / "processed" / "y_test.csv", index=False)

        # Сохранение препроцессоров
        joblib.dump(self.scalers, self.output_path / "artifacts" / "scalers.pkl")

        # Сохранение информации о признаках
        информация_о_признаках = {
            "application_features": list(X_train.columns),
            "target_column": "target",
        }
        joblib.dump(
            информация_о_признаках, self.output_path / "artifacts" / "feature_info.pkl"
        )

        logger.info("Обработанные данные успешно сохранены")

    def process(self):
        """Основной пайплайн обработки данных."""
        logger.info("Запуск пайплайна обработки данных...")

        try:
            # Загрузка данных
            df = self.load_data()

            # Нормализация названий колонок
            df = self.normalize_column_names(df)

            # Очистка данных
            df = self.clean_data(df)

            # Создание целевой переменной
            df = self.create_target_variable(df)

            # Создание признаков
            df = self.engineer_features(df)

            # Выбор признаков
            X, y = self.select_features(df)

            # Предобработка признаков
            X_processed = self.preprocess_features(X, fit=True)

            # Разделение данных
            X_train, X_test, y_train, y_test = self.split_data(X_processed, y)

            # Сохранение обработанных данных
            self.save_processed_data(X_train, X_test, y_train, y_test)

            logger.info("Пайплайн обработки данных успешно завершен!")

        except Exception as e:
            logger.error(f"Ошибка в пайплайне обработки данных: {e}")
            raise


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(
        description="Подготовка данных для модели кредитного скоринга"
    )
    parser.add_argument(
        "--data-path", required=True, help="Путь к файлу с сырыми данными"
    )
    parser.add_argument(
        "--output-path", required=True, help="Путь для сохранения обработанных данных"
    )

    args = parser.parse_args()

    try:
        processor = DataProcessor(args.data_path, args.output_path)
        processor.process()
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import sys

    main()
