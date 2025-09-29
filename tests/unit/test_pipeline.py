"""
Unit тесты для главного пайплайна (scripts/run_pipeline.py).
"""

import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.run_pipeline import (
    run_script,
    check_data_exists,
    create_directories,
    main
)


class TestPipeline:
    """Тесты для главного пайплайна."""
    
    def test_check_data_exists_valid_file(self):
        """Тест проверки существования данных с валидным файлом."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(b"test data")
            f.flush()
            
            # Создаем временную структуру папок
            with tempfile.TemporaryDirectory() as temp_dir:
                data_dir = Path(temp_dir) / "data" / "raw"
                data_dir.mkdir(parents=True)
                
                # Перемещаем файл в правильное место
                target_file = data_dir / "accepted_2007_to_2018Q4.csv"
                Path(f.name).rename(target_file)
                
                # Мокаем project_root
                with patch('scripts.run_pipeline.project_root', Path(temp_dir)):
                    result = check_data_exists()
                    assert result is True
                
                os.unlink(target_file)
    
    def test_check_data_exists_invalid_file(self):
        """Тест проверки существования данных с невалидным файлом."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('scripts.run_pipeline.project_root', Path(temp_dir)):
                result = check_data_exists()
                assert result is False
    
    def test_create_directories(self):
        """Тест создания директорий."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('scripts.run_pipeline.project_root', Path(temp_dir)):
                create_directories()
                
                # Проверяем, что директории созданы
                expected_dirs = [
                    "data/processed",
                    "models/trained",
                    "models/artifacts",
                    "logs"
                ]
                
                for dir_path in expected_dirs:
                    full_path = Path(temp_dir) / dir_path
                    assert full_path.exists()
                    assert full_path.is_dir()
    
    def test_run_script_success(self):
        """Тест успешного запуска скрипта."""
        # Создаем временный скрипт, который завершается успешно
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("print('Test script executed successfully')")
            f.flush()
            
            result = run_script(f.name, "Test Script")
            
            assert result is True
            os.unlink(f.name)
    
    def test_run_script_failure(self):
        """Тест неудачного запуска скрипта."""
        # Создаем временный скрипт, который завершается с ошибкой
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("import sys; sys.exit(1)")
            f.flush()
            
            result = run_script(f.name, "Failing Script")
            
            assert result is False
            os.unlink(f.name)
    
    def test_run_script_nonexistent(self):
        """Тест запуска несуществующего скрипта."""
        result = run_script("nonexistent_script.py", "Nonexistent Script")
        
        assert result is False
    
    @patch('scripts.run_pipeline.check_data_exists')
    @patch('scripts.run_pipeline.create_directories')
    @patch('scripts.run_pipeline.run_script')
    def test_main_success(self, mock_run_script, mock_create_dirs, mock_check_data):
        """Тест успешного выполнения main функции."""
        # Настраиваем моки
        mock_check_data.return_value = True
        mock_create_dirs.return_value = None
        mock_run_script.return_value = True
        
        # Мокаем sys.argv
        with patch('sys.argv', ['run_pipeline.py', '--steps', 'eda']):
            result = main()
            assert result == 0
    
    @patch('scripts.run_pipeline.check_data_exists')
    def test_main_no_data(self, mock_check_data):
        """Тест main функции без данных."""
        mock_check_data.return_value = False
        
        with patch('sys.argv', ['run_pipeline.py']):
            result = main()
            assert result == 1
    
    @patch('scripts.run_pipeline.check_data_exists')
    @patch('scripts.run_pipeline.create_directories')
    @patch('scripts.run_pipeline.run_script')
    def test_main_with_failures(self, mock_run_script, mock_create_dirs, mock_check_data):
        """Тест main функции с ошибками."""
        # Настраиваем моки
        mock_check_data.return_value = True
        mock_create_dirs.return_value = None
        mock_run_script.return_value = False  # Все скрипты падают
        
        with patch('sys.argv', ['run_pipeline.py', '--steps', 'eda']):
            result = main()
            assert result == 1
    
    def test_main_with_help(self):
        """Тест main функции с флагом помощи."""
        with patch('sys.argv', ['run_pipeline.py', '--help']):
            with patch('argparse.ArgumentParser.parse_args') as mock_parse:
                mock_parse.return_value = MagicMock(steps=['all'], skip_checks=False)
                result = main()
                # main должна завершиться без ошибок даже с --help


class TestPipelineIntegration:
    """Интеграционные тесты для пайплайна."""
    
    def test_pipeline_script_structure(self):
        """Тест структуры скриптов пайплайна."""
        project_root = Path(__file__).parent.parent.parent
        
        # Проверяем, что все необходимые скрипты существуют
        required_scripts = [
            "scripts/run_pipeline.py",
            "scripts/data_processing/eda.py",
            "scripts/data_processing/preprocessing.py",
            "scripts/model_training/train_models.py",
            "scripts/model_training/hyperparameter_tuning.py",
            "scripts/model_training/validation.py",
            "scripts/monitoring/model_monitoring.py",
            "scripts/monitoring/data_quality_monitor.py"
        ]
        
        for script_path in required_scripts:
            full_path = project_root / script_path
            assert full_path.exists(), f"Скрипт {script_path} не найден"
            assert full_path.is_file(), f"{script_path} не является файлом"
    
    def test_pipeline_directories_structure(self):
        """Тест структуры директорий пайплайна."""
        project_root = Path(__file__).parent.parent.parent
        
        # Проверяем, что все необходимые директории существуют
        required_dirs = [
            "scripts/data_processing",
            "scripts/model_training",
            "scripts/monitoring",
            "scripts/deployment",
            "tests/unit",
            "tests/integration",
            "tests/e2e"
        ]
        
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            assert full_path.exists(), f"Директория {dir_path} не найдена"
            assert full_path.is_dir(), f"{dir_path} не является директорией"
    
    def test_pipeline_configuration_files(self):
        """Тест конфигурационных файлов пайплайна."""
        project_root = Path(__file__).parent.parent.parent
        
        # Проверяем, что все необходимые конфигурационные файлы существуют
        required_configs = [
            "requirements.txt",
            ".pre-commit-config.yaml",
            "Dockerfile.ml",
            "docker-compose.yml",
            ".github/workflows/ci-cd.yml"
        ]
        
        for config_path in required_configs:
            full_path = project_root / config_path
            assert full_path.exists(), f"Конфигурационный файл {config_path} не найден"
            assert full_path.is_file(), f"{config_path} не является файлом"


class TestPipelineErrorHandling:
    """Тесты обработки ошибок в пайплайне."""
    
    def test_run_script_with_exception(self):
        """Тест запуска скрипта с исключением."""
        # Создаем скрипт, который вызывает исключение
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("raise Exception('Test exception')")
            f.flush()
            
            result = run_script(f.name, "Exception Script")
            
            assert result is False
            os.unlink(f.name)
    
    def test_run_script_with_timeout(self):
        """Тест запуска скрипта с таймаутом."""
        # Создаем скрипт, который выполняется долго
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("import time; time.sleep(10)")
            f.flush()
            
            # Мокаем subprocess.run с таймаутом
            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired("python", 1)
                
                result = run_script(f.name, "Timeout Script")
                
                assert result is False
            os.unlink(f.name)
    
    def test_create_directories_permission_error(self):
        """Тест создания директорий с ошибкой прав доступа."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создаем директорию, которую нельзя изменить
            restricted_dir = Path(temp_dir) / "restricted"
            restricted_dir.mkdir()
            restricted_dir.chmod(0o444)  # Только чтение
            
            with patch('scripts.run_pipeline.project_root', restricted_dir):
                # create_directories должна обработать ошибку
                try:
                    create_directories()
                except PermissionError:
                    # Ожидаем ошибку прав доступа
                    pass
                except Exception as e:
                    # Другие ошибки тоже допустимы
                    assert isinstance(e, (PermissionError, OSError))
                finally:
                    # Восстанавливаем права
                    restricted_dir.chmod(0o755)


class TestPipelinePerformance:
    """Тесты производительности пайплайна."""
    
    def test_pipeline_script_import_time(self):
        """Тест времени импорта скриптов пайплайна."""
        import time
        
        scripts_to_test = [
            "scripts.data_processing.eda",
            "scripts.data_processing.preprocessing",
            "scripts.model_training.train_models",
            "scripts.model_training.mlflow_tracking",
            "scripts.monitoring.model_monitoring",
            "scripts.monitoring.data_quality_monitor"
        ]
        
        for script_name in scripts_to_test:
            start_time = time.time()
            
            try:
                __import__(script_name)
                end_time = time.time()
                import_time = end_time - start_time
                
                # Проверяем, что импорт занял менее 5 секунд
                assert import_time < 5, f"Импорт {script_name} занял {import_time:.2f} секунд"
                
            except ImportError as e:
                # Некоторые модули могут не импортироваться из-за зависимостей
                print(f"Предупреждение: не удалось импортировать {script_name}: {e}")
    
    def test_pipeline_memory_usage(self):
        """Тест использования памяти пайплайном."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Импортируем основные модули
        try:
            from scripts.data_processing import eda
            from scripts.data_processing import preprocessing
            from scripts.model_training import train_models
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Проверяем, что использование памяти не превышает 100 MB
            assert memory_increase < 100, f"Использование памяти увеличилось на {memory_increase:.2f} MB"
            
        except ImportError as e:
            print(f"Предупреждение: не удалось импортировать модули: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
