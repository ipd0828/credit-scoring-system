"""
Unit тесты для модулей мониторинга.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock
import joblib

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.monitoring.model_monitoring import ModelMonitor
from scripts.monitoring.data_quality_monitor import DataQualityMonitor


class TestModelMonitoring:
    """Тесты для мониторинга моделей."""
    
    @pytest.fixture
    def sample_data(self):
        """Создает тестовые данные."""
        np.random.seed(42)
        data = {
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
            'grade': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'emp_length': np.random.choice(['< 1 year', '1 year', '2 years'], 100),
            'target': np.random.choice([0, 1], 100, p=[0.7, 0.3])
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mock_model(self):
        """Создает mock модель."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ])
        return model
    
    @pytest.fixture
    def temp_model_file(self, mock_model, sample_data):
        """Создает временный файл модели."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            # Обучаем модель на тестовых данных
            X = sample_data.drop(columns=['target'])
            y = sample_data['target']
            mock_model.fit(X, y)
            
            joblib.dump(mock_model, f.name)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def temp_reference_data_file(self, sample_data):
        """Создает временный файл референсных данных."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    def test_model_monitor_initialization(self, temp_model_file, temp_reference_data_file):
        """Тест инициализации монитора модели."""
        monitor = ModelMonitor(temp_model_file, temp_reference_data_file)
        
        assert monitor.model_path == temp_model_file
        assert monitor.reference_data_path == temp_reference_data_file
        assert monitor.model is not None
        assert isinstance(monitor.reference_data, pd.DataFrame)
    
    def test_model_monitor_initialization_invalid_files(self):
        """Тест инициализации с неверными файлами."""
        with pytest.raises(FileNotFoundError):
            ModelMonitor("nonexistent_model.pkl", "nonexistent_data.csv")
    
    def test_detect_data_drift(self, temp_model_file, temp_reference_data_file, sample_data):
        """Тест детекции дрифта данных."""
        monitor = ModelMonitor(temp_model_file, temp_reference_data_file)
        
        # Создаем данные с дрифтом
        drifted_data = sample_data.copy()
        drifted_data['feature1'] = drifted_data['feature1'] + 2  # Добавляем дрифт
        
        feature_columns = ['feature1', 'feature2']
        result = monitor.detect_data_drift(drifted_data, feature_columns)
        
        assert isinstance(result, dict)
        assert 'overall_drift_detected' in result
        assert 'feature_drifts' in result
        assert 'drift_score' in result
        assert 'timestamp' in result
        
        # Проверяем, что дрифт обнаружен
        assert result['overall_drift_detected'] is True
        assert result['drift_score'] > 0
    
    def test_detect_data_drift_no_drift(self, temp_model_file, temp_reference_data_file, sample_data):
        """Тест детекции дрифта без дрифта."""
        monitor = ModelMonitor(temp_model_file, temp_reference_data_file)
        
        # Используем те же данные (без дрифта)
        feature_columns = ['feature1', 'feature2']
        result = monitor.detect_data_drift(sample_data, feature_columns)
        
        assert isinstance(result, dict)
        assert 'overall_drift_detected' in result
        assert 'drift_score' in result
        
        # Дрифт должен быть минимальным
        assert result['drift_score'] < 0.5
    
    def test_monitor_model_performance(self, temp_model_file, temp_reference_data_file, sample_data):
        """Тест мониторинга производительности модели."""
        monitor = ModelMonitor(temp_model_file, temp_reference_data_file)
        
        X_test = sample_data.drop(columns=['target'])
        y_test = sample_data['target']
        
        result = monitor.monitor_model_performance(X_test, y_test)
        
        assert isinstance(result, dict)
        assert 'current_metrics' in result
        assert 'performance_degraded' in result
        assert 'timestamp' in result
        
        # Проверяем метрики
        metrics = result['current_metrics']
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Проверяем, что метрики в разумных пределах
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_detect_prediction_bias(self, temp_model_file, temp_reference_data_file, sample_data):
        """Тест детекции смещения предсказаний."""
        monitor = ModelMonitor(temp_model_file, temp_reference_data_file)
        
        X_test = sample_data.drop(columns=['target'])
        y_test = sample_data['target']
        sensitive_attributes = ['grade', 'emp_length']
        
        result = monitor.detect_prediction_bias(X_test, y_test, sensitive_attributes)
        
        assert isinstance(result, dict)
        assert 'bias_detected' in result
        assert 'attribute_bias' in result
        assert 'timestamp' in result
        
        # Проверяем структуру результата
        assert isinstance(result['attribute_bias'], dict)
        for attr in sensitive_attributes:
            if attr in X_test.columns:
                assert attr in result['attribute_bias']
    
    def test_generate_monitoring_report(self, temp_model_file, temp_reference_data_file, sample_data):
        """Тест генерации отчета мониторинга."""
        monitor = ModelMonitor(temp_model_file, temp_reference_data_file)
        
        X_test = sample_data.drop(columns=['target'])
        y_test = sample_data['target']
        feature_columns = ['feature1', 'feature2']
        sensitive_attributes = ['grade']
        
        result = monitor.generate_monitoring_report(
            sample_data, X_test, y_test, feature_columns, sensitive_attributes
        )
        
        assert isinstance(result, dict)
        assert 'timestamp' in result
        assert 'model_path' in result
        assert 'reference_data_path' in result
        assert 'alerts' in result
        assert 'data_drift' in result
        assert 'performance' in result
        assert 'bias' in result
        assert 'overall_status' in result
        
        # Проверяем, что статус определен
        assert result['overall_status'] in ['healthy', 'issues_detected', 'error']


class TestDataQualityMonitoring:
    """Тесты для мониторинга качества данных."""
    
    @pytest.fixture
    def sample_data(self):
        """Создает тестовые данные."""
        np.random.seed(42)
        data = {
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
            'feature4': np.random.choice(['X', 'Y'], 100),
            'high_missing': [np.nan] * 30 + list(range(70)),  # 30% пропусков
            'constant_col': ['constant'] * 100,
            'duplicate_col1': [1, 2, 3, 4, 5] * 20,
            'duplicate_col2': [1, 2, 3, 4, 5] * 20  # Дубликат duplicate_col1
        }
        return pd.DataFrame(data)
    
    def test_data_quality_monitor_initialization(self):
        """Тест инициализации монитора качества данных."""
        monitor = DataQualityMonitor()
        
        assert monitor.config is not None
        assert 'missing_threshold' in monitor.config
        assert 'outlier_threshold' in monitor.config
        assert 'correlation_threshold' in monitor.config
    
    def test_check_missing_values(self, sample_data):
        """Тест проверки пропущенных значений."""
        monitor = DataQualityMonitor()
        
        result = monitor.check_missing_values(sample_data)
        
        assert isinstance(result, dict)
        assert 'total_missing' in result
        assert 'missing_percentage' in result
        assert 'columns_with_missing' in result
        assert 'critical_columns' in result
        assert 'timestamp' in result
        
        # Проверяем, что критическая колонка найдена
        assert len(result['critical_columns']) > 0
        assert 'high_missing' in [col['column'] for col in result['critical_columns']]
    
    def test_check_outliers(self, sample_data):
        """Тест проверки выбросов."""
        monitor = DataQualityMonitor()
        
        result = monitor.check_outliers(sample_data)
        
        assert isinstance(result, dict)
        assert 'columns_with_outliers' in result
        assert 'total_outliers' in result
        assert 'critical_columns' in result
        assert 'timestamp' in result
        
        # Проверяем структуру результата
        assert isinstance(result['columns_with_outliers'], dict)
        assert isinstance(result['total_outliers'], int)
        assert result['total_outliers'] >= 0
    
    def test_check_data_distribution(self, sample_data):
        """Тест проверки распределений данных."""
        monitor = DataQualityMonitor()
        
        result = monitor.check_data_distribution(sample_data)
        
        assert isinstance(result, dict)
        assert 'columns_distribution' in result
        assert 'skewed_columns' in result
        assert 'high_kurtosis_columns' in result
        assert 'low_variance_columns' in result
        assert 'timestamp' in result
        
        # Проверяем структуру результата
        assert isinstance(result['columns_distribution'], dict)
        assert isinstance(result['skewed_columns'], list)
        assert isinstance(result['high_kurtosis_columns'], list)
        assert isinstance(result['low_variance_columns'], list)
    
    def test_check_data_integrity(self, sample_data):
        """Тест проверки целостности данных."""
        monitor = DataQualityMonitor()
        
        result = monitor.check_data_integrity(sample_data)
        
        assert isinstance(result, dict)
        assert 'duplicate_rows' in result
        assert 'duplicate_percentage' in result
        assert 'duplicate_columns' in result
        assert 'constant_columns' in result
        assert 'highly_correlated_columns' in result
        assert 'timestamp' in result
        
        # Проверяем, что константная колонка найдена
        assert 'constant_col' in result['constant_columns']
        
        # Проверяем, что дубликаты колонок найдены
        assert len(result['duplicate_columns']) > 0
    
    def test_check_categorical_data(self, sample_data):
        """Тест проверки категориальных данных."""
        monitor = DataQualityMonitor()
        
        result = monitor.check_categorical_data(sample_data)
        
        assert isinstance(result, dict)
        assert 'categorical_columns' in result
        assert 'high_cardinality_columns' in result
        assert 'low_cardinality_columns' in result
        assert 'timestamp' in result
        
        # Проверяем структуру результата
        assert isinstance(result['categorical_columns'], dict)
        assert isinstance(result['high_cardinality_columns'], list)
        assert isinstance(result['low_cardinality_columns'], list)
    
    def test_generate_quality_report(self, sample_data):
        """Тест генерации отчета о качестве данных."""
        monitor = DataQualityMonitor()
        
        result = monitor.generate_quality_report(sample_data)
        
        assert isinstance(result, dict)
        assert 'timestamp' in result
        assert 'data_shape' in result
        assert 'data_types' in result
        assert 'overall_quality_score' in result
        assert 'issues' in result
        assert 'recommendations' in result
        
        # Проверяем, что скор качества определен
        assert 0 <= result['overall_quality_score'] <= 100
        
        # Проверяем, что проблемы найдены
        assert len(result['issues']) > 0
        assert len(result['recommendations']) > 0


class TestMonitoringEdgeCases:
    """Тесты для граничных случаев мониторинга."""
    
    def test_empty_dataframe_monitoring(self):
        """Тест мониторинга с пустым DataFrame."""
        empty_df = pd.DataFrame()
        monitor = DataQualityMonitor()
        
        result = monitor.generate_quality_report(empty_df)
        
        assert isinstance(result, dict)
        assert result['overall_quality_score'] == 0
        assert 'error' in result
    
    def test_single_column_dataframe_monitoring(self):
        """Тест мониторинга с DataFrame с одной колонкой."""
        single_col_df = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        monitor = DataQualityMonitor()
        
        result = monitor.generate_quality_report(single_col_df)
        
        assert isinstance(result, dict)
        assert result['overall_quality_score'] >= 0
    
    def test_all_missing_values_monitoring(self):
        """Тест мониторинга с DataFrame где все значения пропущены."""
        all_missing_df = pd.DataFrame({
            'col1': [np.nan] * 100,
            'col2': [np.nan] * 100
        })
        monitor = DataQualityMonitor()
        
        result = monitor.generate_quality_report(all_missing_df)
        
        assert isinstance(result, dict)
        assert result['overall_quality_score'] < 50  # Низкий скор из-за пропусков
    
    def test_model_monitoring_with_invalid_model(self):
        """Тест мониторинга с неверной моделью."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            # Создаем файл с невалидной моделью
            f.write(b"invalid model data")
            f.flush()
            
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as data_f:
                pd.DataFrame({'feature1': [1, 2, 3]}).to_csv(data_f.name, index=False)
                
                with pytest.raises(Exception):
                    ModelMonitor(f.name, data_f.name)
            
            os.unlink(f.name)
            os.unlink(data_f.name)


if __name__ == "__main__":
    pytest.main([__file__])
