"""
Скрипт для предобработки данных кредитного скоринга.

Этот скрипт выполняет:
1. Очистку данных от пропусков и выбросов
2. Удаление ненужных столбцов
3. Обработку категориальных и числовых признаков
4. Разделение на обучающую и тестовую выборки
5. Сохранение обработанных данных
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


def load_processed_data(data_path: str) -> pd.DataFrame:
    """
    Загружает предобработанные данные из EDA.

    Args:
        data_path: Путь к файлу с данными

    Returns:
        pd.DataFrame: Загруженные данные
    """
    print("Загрузка предобработанных данных...")
    df = pd.read_csv(data_path)
    print(f"Загружено {df.shape[0]} строк и {df.shape[1]} столбцов")
    return df


def remove_high_missing_columns(
    df: pd.DataFrame, threshold: float = 0.6
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Удаляет столбцы с высоким процентом пропусков.

    Args:
        df: DataFrame для обработки
        threshold: Порог для удаления столбцов (по умолчанию 0.6)

    Returns:
        Tuple[pd.DataFrame, List[str]]: (очищенный DataFrame, список удаленных столбцов)
    """
    print(f"\nУдаление столбцов с >= {threshold:.1%} пропусков...")

    # Вычисляем долю пропусков в каждом столбце
    missing_ratio = df.isnull().mean()

    # Создаем маску: True — если пропусков МЕНЬШЕ threshold
    mask = missing_ratio < threshold

    # Определяем, какие столбцы будут удалены
    columns_to_drop = df.columns[~mask].tolist()

    print(f"Количество удаляемых столбцов: {len(columns_to_drop)}")
    if columns_to_drop:
        print("Список удалённых столбцов:")
        for i, col in enumerate(columns_to_drop, 1):
            print(f"  {i:2d}. {col} — {missing_ratio[col]:.1%} пропусков")
    else:
        print("Нет столбцов с высоким процентом пропусков.")

    # Применяем маску
    df_clean = df.loc[:, mask]

    print(f"Осталось столбцов: {df_clean.shape[1]} из {len(missing_ratio)}")

    return df_clean, columns_to_drop


def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет ненужные столбцы (ID, константные, дублирующиеся).

    Args:
        df: DataFrame для обработки

    Returns:
        pd.DataFrame: Очищенный DataFrame
    """
    print("\nУдаление ненужных столбцов...")

    # Столбцы для удаления
    columns_to_drop = [
        "Unnamed: 0",
        "id",
        "url",
        "policy_code",
        "loan_status",
        "member_id",
        "issue_d",
    ]

    # Проверяем, какие столбцы действительно существуют
    existing_columns = [col for col in columns_to_drop if col in df.columns]

    if existing_columns:
        print(f"Удаляем столбцы: {existing_columns}")
        df_clean = df.drop(columns=existing_columns)
    else:
        print("Нет ненужных столбцов для удаления.")
        df_clean = df.copy()

    print(f"Осталось столбцов: {df_clean.shape[1]}")

    return df_clean


def identify_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Определяет типы признаков (числовые и категориальные).

    Args:
        df: DataFrame для анализа

    Returns:
        Tuple[List[str], List[str]]: (числовые признаки, категориальные признаки)
    """
    print("\nОпределение типов признаков...")

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Исключаем целевую переменную из числовых признаков
    if "target" in numeric_columns:
        numeric_columns.remove("target")

    print(f"Числовых признаков: {len(numeric_columns)}")
    print(f"Категориальных признаков: {len(categorical_columns)}")

    return numeric_columns, categorical_columns


def handle_missing_values(
    df: pd.DataFrame, numeric_columns: List[str], categorical_columns: List[str]
) -> pd.DataFrame:
    """
    Обрабатывает пропущенные значения.

    Args:
        df: DataFrame для обработки
        numeric_columns: Список числовых столбцов
        categorical_columns: Список категориальных столбцов

    Returns:
        pd.DataFrame: DataFrame с обработанными пропусками
    """
    print("\nОбработка пропущенных значений...")

    df_clean = df.copy()

    # Заполняем пропуски в числовых признаках медианой
    for col in numeric_columns:
        if col in df_clean.columns and df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(
                f"  {col}: заполнено {df_clean[col].isnull().sum()} пропусков медианой ({median_val:.2f})"
            )

    # Заполняем пропуски в категориальных признаках модой
    for col in categorical_columns:
        if col in df_clean.columns and df_clean[col].isnull().any():
            mode_val = (
                df_clean[col].mode()[0] if not df_clean[col].mode().empty else "missing"
            )
            df_clean[col].fillna(mode_val, inplace=True)
            print(
                f"  {col}: заполнено {df[col].isnull().sum()} пропусков модой ('{mode_val}')"
            )

    # Проверяем, остались ли пропуски
    remaining_missing = df_clean.isnull().sum().sum()
    if remaining_missing == 0:
        print("Все пропуски успешно обработаны.")
    else:
        print(f"Остались пропуски: {remaining_missing}")

    return df_clean


def create_preprocessor(
    numeric_columns: List[str], categorical_columns: List[str]
) -> ColumnTransformer:
    """
    Создает препроцессор для обработки признаков.

    Args:
        numeric_columns: Список числовых столбцов
        categorical_columns: Список категориальных столбцов

    Returns:
        ColumnTransformer: Настроенный препроцессор
    """
    print("\nСоздание препроцессора...")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )

    print(
        f"Препроцессор создан для {len(numeric_columns)} числовых и {len(categorical_columns)} категориальных признаков"
    )

    return preprocessor


def split_data(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разделяет данные на обучающую и тестовую выборки.

    Args:
        df: DataFrame с данными
        target_col: Название столбца с целевой переменной
        test_size: Размер тестовой выборки
        random_state: Seed для воспроизводимости

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"\nРазделение данных на train/test ({1-test_size:.1%}/{test_size:.1%})...")

    if target_col not in df.columns:
        raise ValueError(f"Целевая переменная '{target_col}' не найдена в данных")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Обучающая выборка: {X_train.shape[0]} строк, {X_train.shape[1]} признаков")
    print(f"Тестовая выборка: {X_test.shape[0]} строк, {X_test.shape[1]} признаков")

    # Проверяем распределение целевой переменной
    train_dist = y_train.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()

    print(f"\nРаспределение целевой переменной:")
    print(
        f"  Обучающая выборка: {train_dist[0]:.1%} хороших, {train_dist[1]:.1%} плохих"
    )
    print(f"  Тестовая выборка: {test_dist[0]:.1%} хороших, {test_dist[1]:.1%} плохих")

    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    output_dir: str = "data/processed",
) -> None:
    """
    Сохраняет обработанные данные и препроцессор.

    Args:
        X_train: Обучающие признаки
        X_test: Тестовые признаки
        y_train: Обучающая целевая переменная
        y_test: Тестовая целевая переменная
        preprocessor: Обученный препроцессор
        output_dir: Папка для сохранения
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nСохранение обработанных данных в {output_path}...")

    # Сохраняем данные
    X_train.to_csv(output_path / "X_train.csv", index=False)
    X_test.to_csv(output_path / "X_test.csv", index=False)
    y_train.to_csv(output_path / "y_train.csv", index=False)
    y_test.to_csv(output_path / "y_test.csv", index=False)

    # Сохраняем препроцессор
    joblib.dump(preprocessor, output_path / "preprocessor.pkl")

    print("Данные успешно сохранены:")
    print(f"  - X_train.csv: {X_train.shape}")
    print(f"  - X_test.csv: {X_test.shape}")
    print(f"  - y_train.csv: {y_train.shape}")
    print(f"  - y_test.csv: {y_test.shape}")
    print(f"  - preprocessor.pkl")


def create_sample_data(
    df: pd.DataFrame, sample_frac: float = 0.1, random_state: int = 42
) -> pd.DataFrame:
    """
    Создает выборку для быстрого тестирования.

    Args:
        df: Исходный DataFrame
        sample_frac: Доля данных для выборки
        random_state: Seed для воспроизводимости

    Returns:
        pd.DataFrame: Выборка данных
    """
    print(f"\nСоздание выборки для тестирования ({sample_frac:.1%})...")

    sample_df = df.sample(frac=sample_frac, random_state=random_state)

    print(f"Выборка: {sample_df.shape[0]} строк из {df.shape[0]}")

    return sample_df


def main():
    """Основная функция для запуска предобработки."""
    # Путь к данным
    data_path = "data/processed/eda_processed_data.csv"

    if not Path(data_path).exists():
        print(f"Файл данных не найден: {data_path}")
        print("Сначала запустите скрипт EDA (eda.py)")
        return

    # Загружаем данные
    df = load_processed_data(data_path)

    # Создаем выборку для тестирования (опционально)
    if df.shape[0] > 100000:  # Если данных много, создаем выборку
        df = create_sample_data(df, sample_frac=0.1)

    # Удаляем столбцы с высоким процентом пропусков
    df_clean, dropped_columns = remove_high_missing_columns(df, threshold=0.6)

    # Удаляем ненужные столбцы
    df_clean = remove_unnecessary_columns(df_clean)

    # Определяем типы признаков
    numeric_columns, categorical_columns = identify_feature_types(df_clean)

    # Обрабатываем пропущенные значения
    df_clean = handle_missing_values(df_clean, numeric_columns, categorical_columns)

    # Создаем препроцессор
    preprocessor = create_preprocessor(numeric_columns, categorical_columns)

    # Разделяем данные
    X_train, X_test, y_train, y_test = split_data(df_clean, target_col="target")

    # Обучаем препроцессор на обучающих данных
    print("\nОбучение препроцессора...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"После препроцессинга:")
    print(f"  Обучающая выборка: {X_train_processed.shape}")
    print(f"  Тестовая выборка: {X_test_processed.shape}")

    # Сохраняем результаты
    save_processed_data(X_train, X_test, y_train, y_test, preprocessor)

    print("\n" + "=" * 60)
    print("ПРЕДОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО")
    print("=" * 60)


if __name__ == "__main__":
    main()
