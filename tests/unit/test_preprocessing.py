"""
Unit тесты для модуля предобработки (scripts/data_processing/preprocessing.py).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.data_processing.preprocessing import (
    load_processed_data,
    remove_high_missing_columns,
    remove_unnecessary_columns,
    identify_feature_types,
    handle_missing_values,
    create_preprocessor,
    split_data,
    create_sample_data
)


class TestPreprocessing:
    """Тесты для функций предобработки."""
    
    @pytest.fixture
    def sample_data(self):
        """Создает тестовые данные."""
        np.random.seed(42)
        data = {
            'id': range(100),
            'loan_amnt': np.random.normal(10000, 3000, 100),
            'int_rate': np.random.normal(12, 3, 100),
            'grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 100),
            'emp_length': np.random.choice(['< 1 year', '1 year', '2 years', '3 years'], 100),
            'annual_inc': np.random.lognormal(10, 0.5, 100),
            'target': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
            'high_missing_col': [np.nan] * 70 + list(range(30)),  # 70% пропусков
            'constant_col': ['constant'] * 100,
            'url': ['http://example.com'] * 100,
            'policy_code': [1] * 100
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_data_files(self, sample_data):
        """Создает временные файлы данных."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создаем файлы данных
            X_train = sample_data.drop(columns=['target'])
            y_train = sample_data['target']
            X_test = sample_data.drop(columns=['target'])
            y_test = sample_data['target']
            
            X_train.to_csv(f"{temp_dir}/X_train.csv", index=False)
            X_test.to_csv(f"{temp_dir}/X_test.csv", index=False)
            y_train.to_csv(f"{temp_dir}/y_train.csv", index=False)
            y_test.to_csv(f"{temp_dir}/y_test.csv", index=False)
            
            # Создаем mock препроцессор
            import joblib
            from sklearn.preprocessing import StandardScaler
            preprocessor = StandardScaler()
            joblib.dump(preprocessor, f"{temp_dir}/preprocessor.pkl")
            
            yield temp_dir
    
    def test_load_processed_data(self, temp_data_files):
        """Тест загрузки обработанных данных."""
        X_train, X_test, y_train, y_test, preprocessor = load_processed_data(temp_data_files)
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert preprocessor is not None
        
        # Проверяем размеры
        assert X_train.shape[0] == 100
        assert X_test.shape[0] == 100
        assert len(y_train) == 100
        assert len(y_test) == 100
    
    def test_load_processed_data_invalid_path(self):
        """Тест загрузки с неверным путем."""
        with pytest.raises(FileNotFoundError):
            load_processed_data("nonexistent_directory")
    
    def test_remove_high_missing_columns(self, sample_data):
        """Тест удаления колонок с высоким процентом пропусков."""
        df_clean, dropped_columns = remove_high_missing_columns(sample_data, threshold=0.6)
        
        assert isinstance(df_clean, pd.DataFrame)
        assert isinstance(dropped_columns, list)
        
        # Проверяем, что колонка с 70% пропусков удалена
        assert 'high_missing_col' in dropped_columns
        assert 'high_missing_col' not in df_clean.columns
        
        # Проверяем, что остальные колонки сохранены
        assert 'loan_amnt' in df_clean.columns
        assert 'int_rate' in df_clean.columns
    
    def test_remove_high_missing_columns_no_high_missing(self):
        """Тест удаления колонок без высокого процента пропусков."""
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [6, 7, 8, 9, 10],
            'col3': [11, 12, 13, 14, 15]
        })
        
        df_clean, dropped_columns = remove_high_missing_columns(data, threshold=0.6)
        
        assert len(dropped_columns) == 0
        assert df_clean.shape == data.shape
    
    def test_remove_unnecessary_columns(self, sample_data):
        """Тест удаления ненужных колонок."""
        df_clean = remove_unnecessary_columns(sample_data)
        
        # Проверяем, что ненужные колонки удалены
        assert 'id' not in df_clean.columns
        assert 'url' not in df_clean.columns
        assert 'policy_code' not in df_clean.columns
        
        # Проверяем, что нужные колонки сохранены
        assert 'loan_amnt' in df_clean.columns
        assert 'target' in df_clean.columns
    
    def test_identify_feature_types(self, sample_data):
        """Тест определения типов признаков."""
        numeric_features, categorical_features = identify_feature_types(sample_data)
        
        assert isinstance(numeric_features, list)
        assert isinstance(categorical_features, list)
        
        # Проверяем, что target исключен из числовых признаков
        assert 'target' not in numeric_features
        
        # Проверяем типы признаков
        expected_numeric = ['loan_amnt', 'int_rate', 'annual_inc', 'high_missing_col']
        expected_categorical = ['grade', 'emp_length', 'constant_col', 'url', 'policy_code']
        
        for col in expected_numeric:
            if col in sample_data.columns:
                assert col in numeric_features
        
        for col in expected_categorical:
            if col in sample_data.columns:
                assert col in categorical_features
    
    def test_handle_missing_values(self, sample_data):
        """Тест обработки пропущенных значений."""
        # Добавляем пропуски
        sample_data.loc[0:10, 'loan_amnt'] = np.nan
        sample_data.loc[0:5, 'grade'] = np.nan
        
        numeric_columns = ['loan_amnt', 'int_rate', 'annual_inc']
        categorical_columns = ['grade', 'emp_length']
        
        df_clean = handle_missing_values(sample_data, numeric_columns, categorical_columns)
        
        # Проверяем, что пропуски обработаны
        assert df_clean['loan_amnt'].notna().all()
        assert df_clean['grade'].notna().all()
        
        # Проверяем, что данные не потеряны
        assert len(df_clean) == len(sample_data)
    
    def test_create_preprocessor(self, sample_data):
        """Тест создания препроцессора."""
        numeric_columns = ['loan_amnt', 'int_rate', 'annual_inc']
        categorical_columns = ['grade', 'emp_length']
        
        preprocessor = create_preprocessor(numeric_columns, categorical_columns)
        
        assert preprocessor is not None
        assert hasattr(preprocessor, 'transform')
        assert hasattr(preprocessor, 'fit')
    
    def test_split_data(self, sample_data):
        """Тест разделения данных."""
        X_train, X_test, y_train, y_test = split_data(
            sample_data, 
            target_col='target', 
            test_size=0.2, 
            random_state=42
        )
        
        # Проверяем размеры
        assert len(X_train) == 80  # 80% от 100
        assert len(X_test) == 20   # 20% от 100
        assert len(y_train) == 80
        assert len(y_test) == 20
        
        # Проверяем, что target исключен из X
        assert 'target' not in X_train.columns
        assert 'target' not in X_test.columns
        
        # Проверяем стратификацию
        train_positive_rate = y_train.mean()
        test_positive_rate = y_test.mean()
        
        # Разница в долях должна быть небольшой
        assert abs(train_positive_rate - test_positive_rate) < 0.1
    
    def test_split_data_invalid_target(self, sample_data):
        """Тест разделения данных с неверной целевой переменной."""
        with pytest.raises(ValueError):
            split_data(sample_data, target_col='nonexistent_column')
    
    def test_create_sample_data(self, sample_data):
        """Тест создания выборки данных."""
        sample_df = create_sample_data(sample_data, sample_frac=0.5, random_state=42)
        
        assert isinstance(sample_df, pd.DataFrame)
        assert len(sample_df) == 50  # 50% от 100
        assert sample_df.shape[1] == sample_data.shape[1]
    
    def test_create_sample_data_full_size(self, sample_data):
        """Тест создания выборки полного размера."""
        sample_df = create_sample_data(sample_data, sample_frac=1.0, random_state=42)
        
        assert len(sample_df) == len(sample_data)
        assert sample_df.shape == sample_data.shape


class TestPreprocessingEdgeCases:
    """Тесты для граничных случаев предобработки."""
    
    def test_empty_dataframe(self):
        """Тест с пустым DataFrame."""
        empty_df = pd.DataFrame()
        
        # remove_high_missing_columns должен обработать пустой DataFrame
        df_clean, dropped_columns = remove_high_missing_columns(empty_df)
        assert df_clean.empty
        assert len(dropped_columns) == 0
        
        # identify_feature_types должен вернуть пустые списки
        numeric_features, categorical_features = identify_feature_types(empty_df)
        assert len(numeric_features) == 0
        assert len(categorical_features) == 0
    
    def test_single_column_dataframe(self):
        """Тест с DataFrame с одной колонкой."""
        single_col_df = pd.DataFrame({'target': [0, 1, 0, 1, 0]})
        
        # identify_feature_types должен исключить target
        numeric_features, categorical_features = identify_feature_types(single_col_df)
        assert 'target' not in numeric_features
        assert 'target' not in categorical_features
    
    def test_all_missing_values(self):
        """Тест с DataFrame где все значения пропущены."""
        all_missing_df = pd.DataFrame({
            'col1': [np.nan] * 100,
            'col2': [np.nan] * 100,
            'target': [0, 1] * 50
        })
        
        # remove_high_missing_columns должен удалить все колонки кроме target
        df_clean, dropped_columns = remove_high_missing_columns(all_missing_df, threshold=0.5)
        assert len(dropped_columns) == 2
        assert df_clean.empty or 'target' in df_clean.columns
    
    def test_very_large_dataset(self):
        """Тест производительности на большом датасете."""
        # Создаем большой датасет
        np.random.seed(42)
        large_data = pd.DataFrame({
            'id': range(10000),
            'feature1': np.random.normal(0, 1, 10000),
            'feature2': np.random.normal(0, 1, 10000),
            'target': np.random.choice([0, 1], 10000)
        })
        
        import time
        start_time = time.time()
        
        # Тест должен завершиться за разумное время
        df_clean, dropped_columns = remove_high_missing_columns(large_data)
        numeric_features, categorical_features = identify_feature_types(df_clean)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Проверяем, что выполнение заняло менее 5 секунд
        assert execution_time < 5
        assert isinstance(df_clean, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__])
