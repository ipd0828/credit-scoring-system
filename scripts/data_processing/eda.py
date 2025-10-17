"""
Скрипт для проведения Exploratory Data Analysis (EDA) кредитного скоринга.

Этот скрипт выполняет:
1. Загрузку и анализ репрезентативности выборки
2. Анализ целевой переменной
3. Детальный EDA с использованием кастомного EDAProcessor
4. Сохранение результатов анализа
"""

import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path
import sys
from typing import Tuple, Dict, Any

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


def load_and_sample_data(data_path: str, sample_frac: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает данные и создает репрезентативную выборку.
    
    Args:
        data_path: Путь к файлу с данными
        sample_frac: Доля данных для выборки (по умолчанию 0.2)
        random_state: Seed для воспроизводимости
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (выборка, полный датасет)
    """
    print("Загрузка данных...")
    df_sample = pd.read_csv(data_path, low_memory=False)
    
    # Создаем репрезентативную выборку
    df = df_sample.sample(frac=sample_frac, random_state=random_state)
    
    print(f"Размер полного датасета: {df_sample.shape}")
    print(f"Размер выборки: {df.shape}")
    print(f"Доля выборки: {sample_frac:.1%}")
    
    return df, df_sample


def analyze_representativeness(df: pd.DataFrame, df_sample: pd.DataFrame) -> Dict[str, Any]:
    """
    Анализирует репрезентативность выборки по сравнению с полным датасетом.
    
    Args:
        df: Выборка данных
        df_sample: Полный датасет
    
    Returns:
        Dict с результатами анализа
    """
    print("\n" + "="*60)
    print("АНАЛИЗ РЕПРЕЗЕНТАТИВНОСТИ ВЫБОРКИ")
    print("="*60)
    
    # Проверяем на пустые DataFrame
    if df.empty or df_sample.empty:
        return {
            'Оценка': 'Не применимо',
            'Причина': 'Пустой DataFrame',
            'Числовые признаки': [],
            'Категориальные признаки': []
        }
    
    # Определяем типы столбцов
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Исключаем служебные столбцы
    exclude_cols = ['id', 'member_id', 'issue_d']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    # Анализ числовых признаков
    report_numeric = []
    for col in numeric_cols:
        full_mean = df[col].mean()
        sample_mean = df_sample[col].mean()
        
        if pd.isna(full_mean) or full_mean == 0:
            diff_percent = float('nan')
            status = "Не применимо"
        else:
            diff_percent = abs(full_mean - sample_mean) / abs(full_mean) * 100
            if diff_percent < 2:
                status = "Хорошо"
            elif diff_percent < 5:
                status = "Удовлетворительно"
            else:
                status = "Тревожно"

        report_numeric.append({
            'Признак': col,
            'Среднее (полная)': full_mean,
            'Среднее (подвыборка)': sample_mean,
            'Абсолютное отклонение': abs(full_mean - sample_mean),
            'Относительное отклонение (%)': diff_percent,
            'Оценка': status
        })
    
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

        report_categorical_summary.append({
            'Признак': col,
            'Среднее отклонение (%)': mean_abs_diff,
            'Кол-во категорий': len(all_categories),
            'Оценка': status
        })
    
    report_categorical_summary = pd.DataFrame(report_categorical_summary).round(4)
    
    # Итоговая оценка
    good_numeric = len(report_numeric[report_numeric['Оценка'] == "Хорошо"])
    warning_numeric = len(report_numeric[report_numeric['Оценка'] == "Удовлетворительно"])
    bad_numeric = len(report_numeric[report_numeric['Оценка'] == "Тревожно"])
    
    good_cat = len(report_categorical_summary[report_categorical_summary['Оценка'] == "Хорошо"])
    warning_cat = len(report_categorical_summary[report_categorical_summary['Оценка'] == "Удовлетворительно"])
    bad_cat = len(report_categorical_summary[report_categorical_summary['Оценка'] == "Тревожно"])
    
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
        'numeric_report': report_numeric,
        'categorical_report': report_categorical_summary,
        'verdict': verdict,
        'stats': {
            'total_good': total_good,
            'total_warning': total_warning,
            'total_bad': total_bad,
            'pct_good': pct_good
        }
    }


def find_target_column(df: pd.DataFrame) -> str:
    """
    Автоматически находит столбец с целевой переменной.
    
    Args:
        df: DataFrame для поиска
    
    Returns:
        str: Название столбца с целевой переменной
    """
    # Нормализуем названия всех столбцов
    norm = {c: re.sub(r'[^a-z0-9]+', '', c.lower()) for c in df.columns}
    
    # Список ожидаемых имён целевой переменной
    preferred = [
        'loancondition', 'loan_status', 'loanstatus', 'isdefault', 'default',
        'badloan', 'chargedoff', 'defaulter', 'target', 'y'
    ]
    
    # 1) Проверяем точное совпадение
    for want in preferred:
        for c, cn in norm.items():
            if cn == want:
                return c
    
    # 2) Ищем по ключевым словам
    for c, cn in norm.items():
        if any(k in cn for k in ['default', 'charg', 'loanstatus', 'badloan', 'loancondition']):
            return c
    
    # 3) Ищем первый бинарный столбец
    for c in df.columns:
        if df[c].dropna().nunique() == 2:
            return c
    
    return None


def analyze_target_variable(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Анализирует целевую переменную.
    
    Args:
        df: DataFrame с данными
        target_col: Название столбца с целевой переменной
    
    Returns:
        Dict с результатами анализа
    """
    print(f"\nАнализ целевой переменной: {target_col}")
    print("-" * 40)
    
    # Удаляем пропуски в целевой переменной
    df_clean = df.dropna(subset=[target_col]).copy()
    
    # Создаем бинарную целевую переменную
    def map_loan_status(status):
        if pd.isna(status):
            return 1
        
        status_str = str(status).strip().lower()
        
        # Хорошие займы -> 0
        if (
            'fully paid' in status_str or 'current' in status_str or 
            'good loan' in status_str or 'no default' in status_str
        ) and 'does not meet' not in status_str:
            return 0
        elif 'does not meet the credit policy. status:fully paid' in status_str:
            return 0
        else:
            return 1
    
    df_clean['target'] = df_clean[target_col].apply(map_loan_status).astype(int)
    
    # Анализ распределения
    value_counts = df_clean['target'].value_counts().sort_index()
    total = len(df_clean)
    
    print(f"Общее количество записей: {total:,}")
    print(f"Хорошие займы (0): {value_counts.get(0, 0):,} ({value_counts.get(0, 0)/total*100:.1f}%)")
    print(f"Плохие займы (1): {value_counts.get(1, 0):,} ({value_counts.get(1, 0)/total*100:.1f}%)")
    
    return {
        'df_clean': df_clean,
        'target_distribution': value_counts,
        'total_records': total,
        'good_loans_pct': value_counts.get(0, 0)/total*100,
        'bad_loans_pct': value_counts.get(1, 0)/total*100
    }


def run_detailed_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Запускает детальный EDA с использованием EDAProcessor.
    
    Args:
        df: DataFrame для анализа
    
    Returns:
        Dict с результатами EDA
    """
    print("\n" + "="*60)
    print("ДЕТАЛЬНЫЙ EDA")
    print("="*60)
    
    if EDAProcessor is None:
        print("EDAProcessor недоступен. Пропускаем детальный EDA.")
        return {}
    
    try:
        eda = EDAProcessor(df)
        eda_summary = eda.generate_eda_summary()
        
        print("Сводная таблица пропусков:")
        print(eda_summary['summary'])
        
        print(f"\nВсего дубликатов строк: {eda_summary['duplicate_count']}")
        
        if not eda_summary['duplicates'].empty:
            print("Дублированные строки:")
            print(eda_summary['duplicates'])
        else:
            print("Дубликатов строк не найдено.")
        
        if eda_summary['duplicate_columns']:
            print("Дублированные столбцы:")
            for col1, col2 in eda_summary['duplicate_columns']:
                print(f"- {col1} и {col2}")
        else:
            print("Дублированных столбцов не найдено.")
        
        return eda_summary
        
    except Exception as e:
        print(f"Ошибка при выполнении детального EDA: {e}")
        return {}


def save_eda_results(results: Dict[str, Any], output_dir: str = "data/processed") -> None:
    """
    Сохраняет результаты EDA.
    
    Args:
        results: Результаты анализа
        output_dir: Папка для сохранения
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем отчеты
    if 'representativeness' in results:
        rep = results['representativeness']
        if 'numeric_report' in rep:
            rep['numeric_report'].to_csv(output_path / 'numeric_representativeness.csv', index=False)
        if 'categorical_report' in rep:
            rep['categorical_report'].to_csv(output_path / 'categorical_representativeness.csv', index=False)
    
    if 'target_analysis' in results and 'df_clean' in results['target_analysis']:
        results['target_analysis']['df_clean'].to_csv(output_path / 'eda_processed_data.csv', index=False)
    
    print(f"\nРезультаты EDA сохранены в папку: {output_path}")


def main():
    """Основная функция для запуска EDA."""
    # Путь к данным
    data_path = "data/raw/accepted_2007_to_2018Q4.csv"
    
    if not Path(data_path).exists():
        print(f"Файл данных не найден: {data_path}")
        return
    
    # Загружаем данные
    df, df_sample = load_and_sample_data(data_path)
    
    # Анализируем репрезентативность
    representativeness_results = analyze_representativeness(df, df_sample)
    
    # Находим целевую переменную
    target_col = find_target_column(df)
    if target_col is None:
        print("Ошибка: не удалось найти целевую переменную")
        return
    
    print(f"Найдена целевая переменная: {target_col}")
    
    # Анализируем целевую переменную
    target_results = analyze_target_variable(df, target_col)
    
    # Детальный EDA
    eda_results = run_detailed_eda(target_results['df_clean'])
    
    # Сохраняем результаты
    all_results = {
        'representativeness': representativeness_results,
        'target_analysis': target_results,
        'detailed_eda': eda_results
    }
    
    save_eda_results(all_results)
    
    print("\n" + "="*60)
    print("EDA ЗАВЕРШЕН УСПЕШНО")
    print("="*60)


if __name__ == "__main__":
    main()
