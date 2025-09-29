"""
Unit тесты для модуля EDA (scripts/data_processing/eda.py).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.data_processing.eda import (
    load_and_sample_data,
    analyze_representativeness,
    find_target_column,
    analyze_target_variable,
    run_detailed_eda
)


class TestEDA:
    """Тесты для функций EDA."""
    
    @pytest.fixture
    def sample_data(self):
        """Создает тестовые данные."""
        np.random.seed(42)
        data = {
            'id': range(1000),
            'loan_amnt': np.random.normal(10000, 3000, 1000),
            'int_rate': np.random.normal(12, 3, 1000),
            'grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 1000),
            'loan_status': np.random.choice(['Fully Paid', 'Charged Off'], 1000, p=[0.8, 0.2]),
            'emp_length': np.random.choice(['< 1 year', '1 year', '2 years', '3 years'], 1000),
            'annual_inc': np.random.lognormal(10, 0.5, 1000)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Создает временный CSV файл."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    def test_load_and_sample_data(self, temp_csv_file):
        """Тест загрузки и выборки данных."""
        df, df_sample = load_and_sample_data(temp_csv_file, sample_frac=0.2, random_state=42)
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df_sample, pd.DataFrame)
        assert len(df) == 200  # 20% от 1000
        assert len(df_sample) == 1000
        assert df.shape[1] == df_sample.shape[1]
    
    def test_load_and_sample_data_invalid_path(self):
        """Тест загрузки с неверным путем."""
        with pytest.raises(FileNotFoundError):
            load_and_sample_data("nonexistent_file.csv")
    
    def test_analyze_representativeness(self, sample_data):
        """Тест анализа репрезентативности."""
        # Создаем подвыборку
        df_sample = sample_data.sample(frac=0.2, random_state=42)
        
        result = analyze_representativeness(df_sample, sample_data)
        
        assert isinstance(result, dict)
        assert 'numeric_report' in result
        assert 'categorical_report' in result
        assert 'verdict' in result
        assert 'stats' in result
        
        # Проверяем структуру отчета
        assert isinstance(result['numeric_report'], pd.DataFrame)
        assert isinstance(result['categorical_report'], pd.DataFrame)
        assert isinstance(result['verdict'], str)
        assert isinstance(result['stats'], dict)
    
    def test_find_target_column(self, sample_data):
        """Тест поиска целевой переменной."""
        # Тест с loan_status
        target_col = find_target_column(sample_data)
        assert target_col == 'loan_status'
        
        # Тест с переименованной колонкой
        sample_data_renamed = sample_data.rename(columns={'loan_status': 'target'})
        target_col = find_target_column(sample_data_renamed)
        assert target_col == 'target'
        
        # Тест без целевой переменной
        sample_data_no_target = sample_data.drop(columns=['loan_status'])
        target_col = find_target_column(sample_data_no_target)
        assert target_col is None
    
    def test_analyze_target_variable(self, sample_data):
        """Тест анализа целевой переменной."""
        result = analyze_target_variable(sample_data, 'loan_status')
        
        assert isinstance(result, dict)
        assert 'df_clean' in result
        assert 'target_distribution' in result
        assert 'total_records' in result
        assert 'good_loans_pct' in result
        assert 'bad_loans_pct' in result
        
        # Проверяем, что данные очищены
        assert isinstance(result['df_clean'], pd.DataFrame)
        assert 'target' in result['df_clean'].columns
        
        # Проверяем распределение
        assert isinstance(result['target_distribution'], pd.Series)
        assert result['total_records'] > 0
        assert 0 <= result['good_loans_pct'] <= 100
        assert 0 <= result['bad_loans_pct'] <= 100
    
    def test_analyze_target_variable_with_nulls(self):
        """Тест анализа целевой переменной с пропусками."""
        data_with_nulls = pd.DataFrame({
            'loan_status': ['Fully Paid', 'Charged Off', None, 'Fully Paid', 'Charged Off'],
            'other_col': [1, 2, 3, 4, 5]
        })
        
        result = analyze_target_variable(data_with_nulls, 'loan_status')
        
        assert isinstance(result, dict)
        assert 'df_clean' in result
        # Проверяем, что пропуски удалены
        assert result['df_clean']['loan_status'].notna().all()
    
    def test_run_detailed_eda_without_processor(self, sample_data):
        """Тест детального EDA без EDAProcessor."""
        result = run_detailed_eda(sample_data)
        
        # Должен вернуть пустой словарь, если EDAProcessor недоступен
        assert isinstance(result, dict)
    
    def test_eda_functions_with_empty_dataframe(self):
        """Тест функций EDA с пустым DataFrame."""
        empty_df = pd.DataFrame()
        
        # find_target_column должен вернуть None для пустого DataFrame
        assert find_target_column(empty_df) is None
        
        # analyze_representativeness должен обработать пустой DataFrame
        result = analyze_representativeness(empty_df, empty_df)
        assert isinstance(result, dict)
    
    def test_eda_functions_with_single_column(self):
        """Тест функций EDA с DataFrame с одной колонкой."""
        single_col_df = pd.DataFrame({'loan_status': ['Fully Paid', 'Charged Off']})
        
        # find_target_column должен найти целевую переменную
        assert find_target_column(single_col_df) == 'loan_status'
        
        # analyze_target_variable должен работать
        result = analyze_target_variable(single_col_df, 'loan_status')
        assert isinstance(result, dict)
        assert 'df_clean' in result


class TestEDAPerformance:
    """Тесты производительности для EDA."""
    
    def test_large_dataset_performance(self):
        """Тест производительности на большом датасете."""
        # Создаем большой датасет
        np.random.seed(42)
        large_data = pd.DataFrame({
            'id': range(10000),
            'loan_amnt': np.random.normal(10000, 3000, 10000),
            'int_rate': np.random.normal(12, 3, 10000),
            'grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 10000),
            'loan_status': np.random.choice(['Fully Paid', 'Charged Off'], 10000, p=[0.8, 0.2])
        })
        
        # Тест должен завершиться за разумное время
        import time
        start_time = time.time()
        
        result = analyze_representativeness(large_data, large_data)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Проверяем, что выполнение заняло менее 10 секунд
        assert execution_time < 10
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__])
