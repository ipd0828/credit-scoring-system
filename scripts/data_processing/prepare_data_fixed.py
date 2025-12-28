#!/usr/bin/env python3
"""
scripts/data_processing/prepare_data_fixed.py
Исправленная подготовка данных для кредитного скоринга.
БЕЗ масштабирования для tree-based моделей.
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataProcessorFixed:
    """Исправленный обработчик данных."""

    def __init__(self, data_path: str, output_path: str):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)

        # Создание выходных директорий
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "artifacts").mkdir(exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Загрузка сырых данных."""
        logger.info(f"Загрузка данных из {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Файл данных не найден: {self.data_path}")

        df = pd.read_csv(self.data_path)
        logger.info(f"Загружено {len(df)} строк, {len(df.columns)} колонок")

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка данных."""
        logger.info("Очистка данных...")

        # Удаление дубликатов
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        if removed > 0:
            logger.info(f"Удалено {removed} дублирующихся строк")

        return df

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание целевой переменной."""
        logger.info("Создание целевой переменной...")

        # Для UCI Credit Card
        if 'default.payment.next.month' in df.columns:
            df['target'] = df['default.payment.next.month']
            logger.info("Используется default.payment.next.month как целевая переменная")
        else:
            raise ValueError("Не найдена целевая переменная")

        # Проверка распределения
        target_dist = df['target'].value_counts()
        logger.info(f"Распределение целевой переменной: {target_dist.to_dict()}")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков."""
        logger.info("Создание признаков...")

        # 6 признаков из заявки
        features = df[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0']].copy()

        # Переименование
        features.columns = ['limit_bal', 'sex', 'education', 'marriage', 'age', 'pay_0']

        # Обработка категориальных признаков
        # Образование
        features['education_new'] = features['education'].map({
            1: 1, 2: 2, 3: 3, 4: 4, 5: 0, 6: 0, 0: 0
        }).fillna(0).astype(int)

        # Семейное положение
        features['marriage_new'] = features['marriage'].map({
            1: 1, 2: 2, 3: 3, 0: 0
        }).fillna(0).astype(int)

        # Статус платежа
        features['pay_new'] = features['pay_0'].apply(
            lambda x: 1 if x > 0 else (0 if x == 0 else -1)
        )

        # Финальные признаки (БЕЗ МАСШТАБИРОВАНИЯ!)
        final_features = ['limit_bal', 'sex', 'age', 'education_new', 'marriage_new', 'pay_new']
        features_final = features[final_features].copy()

        logger.info(f"Создано {len(final_features)} признаков: {final_features}")

        return features_final

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Основной пайплайн подготовки."""
        # Загрузка
        df = self.load_data()

        # Очистка
        df = self.clean_data(df)

        # Целевая переменная
        df = self.create_target(df)

        # Признаки
        X = self.engineer_features(df)
        y = df['target'].copy()

        return X, y

    def split_and_save(self, X: pd.DataFrame, y: pd.Series):
        """Разделение и сохранение данных."""
        logger.info("Разделение данных на train/test...")

        # Стратифицированное разделение
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Дополнительно: validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.125, random_state=42, stratify=y_train  # 0.2 * 0.125 = 0.025
        )

        logger.info(f"Размеры:")
        logger.info(f"  Train: {X_train.shape}")
        logger.info(f"  Val:   {X_val.shape}")
        logger.info(f"  Test:  {X_test.shape}")

        # Сохранение
        X_train.to_csv(self.output_path / "X_train.csv", index=False)
        X_val.to_csv(self.output_path / "X_val.csv", index=False)
        X_test.to_csv(self.output_path / "X_test.csv", index=False)
        y_train.to_csv(self.output_path / "y_train.csv", index=False)
        y_val.to_csv(self.output_path / "y_val.csv", index=False)
        y_test.to_csv(self.output_path / "y_test.csv", index=False)

        # Сохранение информации о признаках
        feature_info = {
            'features': list(X.columns),
            'feature_descriptions': {
                'limit_bal': 'Кредитный лимит (оригинальный масштаб)',
                'sex': 'Пол (1=мужской, 2=женский)',
                'age': 'Возраст (оригинальный масштаб)',
                'education_new': 'Образование (0=неизвестно, 1=аспирантура, 2=университет, 3=средняя школа, 4=другое)',
                'marriage_new': 'Семейное положение (0=неизвестно, 1=женат/замужем, 2=не женат, 3=другое)',
                'pay_new': 'Статус платежа (-1=оплачен вовремя, 0=револьверный кредит, 1=задержка)'
            },
            'target': 'default.payment.next.month (1=дефолт, 0=не дефолт)',
            'note': 'Данные НЕ масштабированы (оптимально для tree-based моделей)'
        }

        joblib.dump(feature_info, self.output_path / "artifacts" / "feature_info.pkl")

        logger.info(f"Данные сохранены в {self.output_path}")

        # Выводим статистику для проверки
        logger.info(f"\nПроверка масштабирования:")
        for col in X_train.columns:
            logger.info(f"  {col}: mean={X_train[col].mean():.2f}, std={X_train[col].std():.2f}")

    def process(self):
        """Запуск полного пайплайна."""
        logger.info("Запуск исправленного пайплайна подготовки данных...")

        try:
            X, y = self.prepare_data()
            self.split_and_save(X, y)

            logger.info("Пайплайн успешно завершен!")
            logger.info("Данные готовы для обучения моделей (БЕЗ МАСШТАБИРОВАНИЯ)")

        except Exception as e:
            logger.error(f"Ошибка: {e}")
            raise


def main():
    """Основная функция."""
    parser = argparse.ArgumentParser(
        description="Исправленная подготовка данных (без масштабирования)"
    )
    parser.add_argument(
        "--data-path",
        default="data/raw/UCI_Credit_Card.csv",
        help="Путь к файлу с сырыми данными"
    )
    parser.add_argument(
        "--output-path",
        default="data/processed_fixed",
        help="Путь для сохранения обработанных данных"
    )

    args = parser.parse_args()

    try:
        processor = DataProcessorFixed(args.data_path, args.output_path)
        processor.process()
    except Exception as e:
        logger.error(f"Ошибка выполнения: {e}")
        raise


if __name__ == "__main__":
    main()