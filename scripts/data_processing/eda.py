"""
Скрипт для проведения Exploratory Data Analysis (EDA) кредитного скоринга.

Этот скрипт выполняет:
1. Загрузку и анализ репрезентативности выборки
2. Анализ целевой переменной
3. Проверку на утечку данных
4. Детальный EDA с использованием кастомного EDAProcessor
5. Сохранение результатов анализа
"""

import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Добавляем корневую папку проекта в путь для импорта
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Импортируем кастомный EDAProcessor (предполагается, что он будет создан)
try:
    from eda_script import EDAProcessor
except ImportError:
    print("Предупреждение: EDAProcessor не найден. Создайте файл eda_script.py")
    EDAProcessor = None

warnings.filterwarnings("ignore")


def load_and_sample_data(
        data_path: str, sample_frac: float = 0.9, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает данные и создает репрезентативную выборку."""
    print("Загрузка данных...")
    df_sample = pd.read_csv(data_path, low_memory=False)

    # Создаем репрезентативную выборку
    df = df_sample.sample(frac=sample_frac, random_state=random_state)

    print(f"Размер полного датасета: {df_sample.shape}")
    print(f"Размер выборки: {df.shape}")
    print(f"Доля выборки: {sample_frac:.1%}")

    return df, df_sample


def analyze_representativeness(
        df: pd.DataFrame, df_sample: pd.DataFrame
) -> Dict[str, Any]:
    """Анализирует репрезентативность выборки по сравнению с полным датасетом."""
    print("\n" + "=" * 60)
    print("АНАЛИЗ РЕПРЕЗЕНТАТИВНОСТИ ВЫБОРКИ")
    print("=" * 60)

    # Проверяем на пустые DataFrame
    if df.empty or df_sample.empty:
        return {
            "Оценка": "Не применимо",
            "Причина": "Пустой DataFrame",
            "Числовые признаки": [],
            "Категориальные признаки": [],
        }

    # Определяем типы столбцов
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Исключаем служебные столбцы
    exclude_cols = ["id", "member_id", "issue_d"]
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

    # Анализ числовых признаков
    report_numeric = []
    for col in numeric_cols:
        full_mean = df[col].mean()
        sample_mean = df_sample[col].mean()

        if pd.isna(full_mean) or full_mean == 0:
            diff_percent = float("nan")
            status = "Не применимо"
        else:
            diff_percent = abs(full_mean - sample_mean) / abs(full_mean) * 100
            if diff_percent < 2:
                status = "Хорошо"
            elif diff_percent < 5:
                status = "Удовлетворительно"
            else:
                status = "Тревожно"

        report_numeric.append(
            {
                "Признак": col,
                "Среднее (полная)": full_mean,
                "Среднее (подвыборка)": sample_mean,
                "Абсолютное отклонение": abs(full_mean - sample_mean),
                "Относительное отклонение (%)": diff_percent,
                "Оценка": status,
            }
        )

    report_numeric = pd.DataFrame(report_numeric).round(4)

    # Анализ категориальных признаков
    report_categorical_summary = []
    for col in categorical_cols:
        full_dist = df[col].value_counts(normalize=True).sort_index()
        sample_dist = df_sample[col].value_counts(normalize=True).sort_index()

        # Совмещаем индексы
        all_categories = full_dist.index.union(sample_dist.index)
        full_aligned = full_dist.reindex(all_categories, fill_value=0)
        sample_aligned = sample_dist.reindex(all_categories, fill_value=0)

        # Разница по каждой категории
        diff_per_cat = (sample_aligned - full_aligned).abs()
        mean_abs_diff = diff_per_cat.mean() * 100

        if mean_abs_diff < 1:
            status = "Хорошо"
        elif mean_abs_diff < 3:
            status = "Удовлетворительно"
        else:
            status = "Тревожно"

        report_categorical_summary.append(
            {
                "Признак": col,
                "Среднее отклонение (%)": mean_abs_diff,
                "Кол-во категорий": len(all_categories),
                "Оценка": status,
            }
        )

    report_categorical_summary = pd.DataFrame(report_categorical_summary).round(4)

    # Итоговая оценка
    good_numeric = 0
    warning_numeric = 0
    bad_numeric = 0

    if not report_numeric.empty and "Оценка" in report_numeric.columns:
        good_numeric = len(report_numeric[report_numeric["Оценка"] == "Хорошо"])
        warning_numeric = len(report_numeric[report_numeric["Оценка"] == "Удовлетворительно"])
        bad_numeric = len(report_numeric[report_numeric["Оценка"] == "Тревожно"])

    good_cat = 0
    warning_cat = 0
    bad_cat = 0

    if not report_categorical_summary.empty and "Оценка" in report_categorical_summary.columns:
        good_cat = len(report_categorical_summary[report_categorical_summary["Оценка"] == "Хорошо"])
        warning_cat = len(report_categorical_summary[report_categorical_summary["Оценка"] == "Удовлетворительно"])
        bad_cat = len(report_categorical_summary[report_categorical_summary["Оценка"] == "Тревожно"])

    total_good = good_numeric + good_cat
    total_warning = warning_numeric + warning_cat
    total_bad = bad_numeric + bad_cat
    total_all = total_good + total_warning + total_bad

    pct_good = total_good / total_all * 100 if total_all > 0 else 0

    print(f"Числовые признаки: {len(numeric_cols)} всего")
    print(f"  Хорошо: {good_numeric}")
    print(f"  Удовлетворительно: {warning_numeric}")
    print(f"  Тревожно: {bad_numeric}")

    print(f"\nКатегориальные признаки: {len(categorical_cols)} всего")
    print(f"  Хорошо: {good_cat}")
    print(f"  Удовлетворительно: {warning_cat}")
    print(f"  Тревожно: {bad_cat}")

    print(f"\nОбщая статистика:")
    print(f"  Хорошо: {total_good} ({pct_good:.1f}%)")
    print(f"  Удовлетворительно: {total_warning}")
    print(f"  Тревожно: {total_bad}")

    if total_good > total_warning and total_good > total_bad:
        verdict = "Полная репрезентативность: большинство признаков имеют малые отклонения."
    elif total_warning > total_good and total_warning > total_bad:
        verdict = "Частичная репрезентативность: преобладают умеренные отклонения."
    elif total_bad > total_good and total_bad > total_warning:
        verdict = "Низкая репрезентативность: значительная часть признаков сильно искажена."
    else:
        verdict = "Неоднозначная репрезентативность: нет чёткого преобладания."

    print(f"\n{verdict}")

    return {
        "numeric_report": report_numeric,
        "categorical_report": report_categorical_summary,
        "verdict": verdict,
        "stats": {
            "total_good": total_good,
            "total_warning": total_warning,
            "total_bad": total_bad,
            "pct_good": pct_good,
        },
    }


def find_target_column(df: pd.DataFrame) -> str:
    """Автоматически находит столбец с целевой переменной."""
    # Нормализуем названия всех столбцов
    norm = {c: re.sub(r"[^a-z0-9]+", "", c.lower()) for c in df.columns}

    # Список ожидаемых имён целевой переменной
    preferred = [
        "loancondition", "loan_status", "loanstatus", "isdefault", "default",
        "badloan", "chargedoff", "defaulter", "target", "y",
    ]

    # 1) Проверяем точное совпадение
    for want in preferred:
        for c, cn in norm.items():
            if cn == want:
                return c

    # 2) Ищем по ключевым словам
    for c, cn in norm.items():
        if any(k in cn for k in ["default", "charg", "loanstatus", "badloan", "loancondition"]):
            return c

    # 3) Ищем первый бинарный столбец
    for c in df.columns:
        if df[c].dropna().nunique() == 2:
            return c

    return None


def analyze_target_variable(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Анализирует целевую переменную."""
    print(f"\nАнализ целевой переменной: {target_col}")
    print("-" * 40)

    # Сначала покажем все уникальные значения статусов для отладки
    unique_statuses = df[target_col].value_counts()
    print("Уникальные значения целевой переменной:")
    for status, count in unique_statuses.items():
        print(f"  '{status}' ({type(status).__name__}): {count} записей")

    # Удаляем пропуски в целевой переменной
    df_clean = df.dropna(subset=[target_col]).copy()

    # Проверяем тип данных целевой переменной
    dtype = df_clean[target_col].dtype
    print(f"Тип данных целевой переменной: {dtype}")

    # Если целевая переменная уже числовая и бинарная, используем как есть
    if dtype in ['int64', 'float64', 'int32', 'float32']:
        unique_vals = df_clean[target_col].unique()
        print(f"Уникальные числовые значения: {unique_vals}")

        if set(unique_vals).issubset({0, 1}):
            print("Целевая переменная уже бинарная (0 и 1), используем как есть")
            df_clean["target"] = df_clean[target_col].astype(int)
        else:
            print("Целевая переменная числовая, но не бинарная. Создаем бинарную версию.")
            df_clean["target"] = (df_clean[target_col] > 0).astype(int)
    else:
        # Оригинальная логика для текстовых статусов
        print("Целевая переменная текстовая, применяем маппинг статусов...")

        def map_loan_status(status):
            if pd.isna(status):
                return 1

            status_str = str(status).strip().lower()

            # Хорошие займы -> 0
            good_indicators = [
                "fully paid", "current", "good loan", "no default",
                "paid", "completed"
            ]

            # Плохие займы -> 1
            bad_indicators = [
                "charged off", "default", "late", "bad loan",
                "delinquent", "chargedoff", "defaulted"
            ]

            # Проверяем хорошие займы
            for indicator in good_indicators:
                if indicator in status_str:
                    return 0

            # Проверяем плохие займы
            for indicator in bad_indicators:
                if indicator in status_str:
                    return 1

            # Для числовых значений в текстовом формате
            if status_str in ['0', '0.0']:
                return 0
            elif status_str in ['1', '1.0']:
                return 1
            else:
                print(f"  Неизвестный статус: '{status}' -> помечен как плохой займ")
                return 1

        df_clean["target"] = df_clean[target_col].apply(map_loan_status).astype(int)

    # Анализ распределения
    value_counts = df_clean["target"].value_counts().sort_index()
    total = len(df_clean)

    print(f"\nОбщее количество записей: {total:,}")
    print(f"Хорошие займы (0): {value_counts.get(0, 0):,} ({value_counts.get(0, 0) / total * 100:.1f}%)")
    print(f"Плохие займы (1): {value_counts.get(1, 0):,} ({value_counts.get(1, 0) / total * 100:.1f}%)")

    # Проверяем, что есть записи в обоих классах
    if 0 not in value_counts.index or 1 not in value_counts.index:
        print("\nПРЕДУПРЕЖДЕНИЕ: Отсутствуют записи одного из классов целевой переменной!")
        if 0 not in value_counts.index:
            print("  - Нет хороших займов (класс 0)")
        if 1 not in value_counts.index:
            print("  - Нет плохих займов (класс 1)")

    return {
        "df_clean": df_clean,
        "target_distribution": value_counts,
        "total_records": total,
        "good_loans_pct": value_counts.get(0, 0) / total * 100,
        "bad_loans_pct": value_counts.get(1, 0) / total * 100,
        "target_col": target_col,
        "original_statuses": unique_statuses
    }


def check_for_target_leakage(df: pd.DataFrame, target_col: str = "target"):
    """
    Проверяет наличие утечки целевой переменной в признаках.
    """
    print("\n" + "=" * 60)
    print("ПРОВЕРКА НА УТЕЧКУ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
    print("=" * 60)

    results = {
        "duplicate_columns": [],
        "high_correlation_features": [],
        "problematic_columns": []
    }

    # 1. Проверяем точные дубликаты столбцов
    print("1. Проверка точных дубликатов столбцов:")
    duplicate_pairs = []
    columns = [col for col in df.columns if col != target_col]

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i < j and df[col1].equals(df[col2]):
                duplicate_pairs.append((col1, col2))
                print(f"   ВНИМАНИЕ: ДУБЛИКАТ: '{col1}' и '{col2}'")

    if duplicate_pairs:
        results["duplicate_columns"] = duplicate_pairs
    else:
        print("   OK: Дублирующихся столбцов не найдено")

    # 2. Проверяем корреляцию с целевой переменной
    print("\n2. Проверка корреляции с целевой переменной:")
    if target_col in df.columns:
        # Вычисляем корреляции только для числовых признаков
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]

        if len(numeric_cols) > 0:
            correlations = {}
            for col in numeric_cols:
                corr = df[col].corr(df[target_col])
                if not pd.isna(corr):
                    correlations[col] = abs(corr)

            # Сортируем по убыванию корреляции
            sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

            print("   Топ-10 признаков по корреляции с целевой переменной:")
            for i, (col, corr) in enumerate(sorted_correlations[:10], 1):
                status = "ВЫСОКАЯ" if corr > 0.9 else "нормальная"
                print(f"   {i:2d}. {col}: {corr:.4f} ({status})")

                if corr > 0.9:
                    results["high_correlation_features"].append((col, corr))

            if not results["high_correlation_features"]:
                print("   OK: Нет признаков с подозрительно высокой корреляцией")
        else:
            print("   INFO: Нет числовых признаков для анализа корреляции")
    else:
        print("   INFO: Целевая переменная не найдена для анализа корреляции")

    # 3. Проверяем конкретные проблемные столбцы
    print("\n3. Проверка конкретных проблемных столбцов:")
    problematic_columns = []
    suspicious_names = ['default.payment.next.month', 'y', 'default', 'is_bad', 'is_default']

    for col in df.columns:
        if col != target_col and col in suspicious_names:
            if df[col].nunique() == 2 and set(df[col].unique()).issubset({0, 1}):
                problematic_columns.append(col)
                print(f"   ВНИМАНИЕ: ПОДОЗРИТЕЛЬНЫЙ: '{col}' - вероятно дублирует целевую переменную")

    if problematic_columns:
        results["problematic_columns"] = problematic_columns
    else:
        print("   OK: Подозрительных столбцов не найдено")

    # 4. Рекомендации
    print("\n4. РЕКОМЕНДАЦИИ:")
    if results["duplicate_columns"] or results["high_correlation_features"] or results["problematic_columns"]:
        print("   ОБНАРУЖЕНЫ ПОТЕНЦИАЛЬНЫЕ ПРОБЛЕМЫ:")

        if results["duplicate_columns"]:
            print("   - Удалите дублирующиеся столбцы перед обучением")
            for col1, col2 in results["duplicate_columns"]:
                print(f"     * '{col1}' и '{col2}'")

        if results["high_correlation_features"]:
            print("   - Проверьте признаки с высокой корреляцией с целевой переменной:")
            for col, corr in results["high_correlation_features"]:
                print(f"     * '{col}': {corr:.4f}")

        if results["problematic_columns"]:
            print("   - Удалите подозрительные столбцы, которые могут дублировать целевую переменную:")
            for col in results["problematic_columns"]:
                print(f"     * '{col}'")
    else:
        print("   OK: Данные выглядят чистыми, утечек не обнаружено")

    return results


def run_detailed_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Запускает детальный EDA с использованием EDAProcessor."""
    print("\n" + "=" * 60)
    print("ДЕТАЛЬНЫЙ EDA")
    print("=" * 60)

    if EDAProcessor is None:
        print("EDAProcessor недоступен. Пропускаем детальный EDA.")
        return {}

    try:
        eda = EDAProcessor(df)
        eda_summary = eda.generate_eda_summary()

        print("Сводная таблица пропусков:")
        print(eda_summary["summary"])

        print(f"\nВсего дубликатов строк: {eda_summary['duplicate_count']}")

        if not eda_summary["duplicates"].empty:
            print("Дублированные строки:")
            print(eda_summary["duplicates"])
        else:
            print("Дубликатов строк не найдено.")

        if eda_summary["duplicate_columns"]:
            print("Дублированные столбцы:")
            for col1, col2 in eda_summary["duplicate_columns"]:
                print(f"- {col1} и {col2}")
        else:
            print("Дублированных столбцов не найдено.")

        return eda_summary

    except Exception as e:
        print(f"Ошибка при выполнении детального EDA: {e}")
        return {}


def save_eda_results(
        results: Dict[str, Any], output_dir: str = "data/processed"
) -> None:
    """Сохраняет результаты EDA."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Сохраняем отчеты
    if "representativeness" in results:
        rep = results["representativeness"]
        if "numeric_report" in rep:
            rep["numeric_report"].to_csv(
                output_path / "numeric_representativeness.csv", index=False
            )
        if "categorical_report" in rep:
            rep["categorical_report"].to_csv(
                output_path / "categorical_representativeness.csv", index=False
            )

    if "target_analysis" in results and "df_clean" in results["target_analysis"]:
        # Сохраняем ОЧИЩЕННЫЕ данные
        results["target_analysis"]["df_clean"].to_csv(
            output_path / "eda_processed_data.csv", index=False
        )

        # Сохраняем отчет о проверке утечек
        if "leakage_check" in results["target_analysis"]:
            leakage_df = pd.DataFrame({
                'problem_type': ['duplicate_columns', 'high_correlation', 'problematic_columns'],
                'count': [
                    len(results["target_analysis"]["leakage_check"]["duplicate_columns"]),
                    len(results["target_analysis"]["leakage_check"]["high_correlation_features"]),
                    len(results["target_analysis"]["leakage_check"]["problematic_columns"])
                ]
            })
            leakage_df.to_csv(output_path / "leakage_check_report.csv", index=False)

    print(f"\nРезультаты EDA сохранены в папку: {output_path}")


def main():
    """Основная функция для запуска EDA."""
    # Получаем правильный путь к данным
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_path = project_root / "data" / "raw" / "accepted_2007_to_2018Q4.csv"

    print(f"Ищем файл по пути: {data_path}")

    if not data_path.exists():
        print(f"Файл данных не найден: {data_path}")

        # Покажем какие файлы есть в папке data/raw
        raw_dir = project_root / "data" / "raw"
        if raw_dir.exists():
            print(f"\nФайлы в {raw_dir}:")
            for file in raw_dir.glob("*.csv"):
                print(f"  - {file.name}")
        else:
            print(f"\nДиректория {raw_dir} не существует")
        return

    # Загружаем данные
    df, df_sample = load_and_sample_data(str(data_path))

    # Анализируем репрезентативность
    representativeness_results = analyze_representativeness(df, df_sample)

    # Находим целевую переменную
    target_col = find_target_column(df)
    if target_col is None:
        print("Ошибка: не удалось найти целевую переменную")
        print("Доступные столбцы:", df.columns.tolist())
        return

    print(f"Найдена целевая переменная: {target_col}")

    # Дополнительная отладочная информация
    print(f"Тип данных целевой переменной: {df[target_col].dtype}")
    print(f"Уникальные значения: {df[target_col].unique()}")
    print(f"\nПервые 10 уникальных значений '{target_col}':")
    target_value_counts = df[target_col].value_counts().head(10)
    for value, count in target_value_counts.items():
        print(f"  '{value}': {count} записей")
    print(f"Всего уникальных значений: {df[target_col].nunique()}")

    # Анализируем целевую переменную
    target_results = analyze_target_variable(df, target_col)

    # Проверяем, что анализ целевой переменной прошел успешно
    if target_results is None or "df_clean" not in target_results:
        print("Ошибка: анализ целевой переменной не вернул очищенные данные")
        return

    # ПРОВЕРКА НА УТЕЧКУ ДАННЫХ И ОЧИСТКА
    leakage_results = check_for_target_leakage(target_results["df_clean"], "target")

    # УДАЛЯЕМ ПРОБЛЕМНЫЕ СТОЛБЦЫ СРАЗУ В EDA
    df_clean = target_results["df_clean"].copy()

    # Удаляем дублирующиеся столбцы
    for col1, col2 in leakage_results["duplicate_columns"]:
        if col2 in df_clean.columns:
            print(f"Удаляем дублирующий столбец: {col2}")
            df_clean = df_clean.drop(columns=[col2])

    # Удаляем подозрительные столбцы
    for col in leakage_results["problematic_columns"]:
        if col in df_clean.columns:
            print(f"Удаляем подозрительный столбец: {col}")
            df_clean = df_clean.drop(columns=[col])

    # Обновляем df_clean в результатах
    target_results["df_clean"] = df_clean
    target_results["leakage_check"] = leakage_results

    # Детальный EDA на очищенных данных
    eda_results = run_detailed_eda(target_results["df_clean"])

    # Сохраняем результаты
    all_results = {
        "representativeness": representativeness_results,
        "target_analysis": target_results,
        "detailed_eda": eda_results,
    }

    save_eda_results(all_results)

    print("\n" + "=" * 60)
    print("EDA ЗАВЕРШЕН УСПЕШНО")
    print("=" * 60)


if __name__ == "__main__":
    main()