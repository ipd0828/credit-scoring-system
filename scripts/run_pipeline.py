"""
Главный скрипт для запуска полного пайплайна кредитного скоринга.

Этот скрипт последовательно выполняет:
1. EDA (Exploratory Data Analysis)
2. Предобработку данных
3. Обучение моделей
4. Подбор гиперпараметров
5. Валидацию моделей
"""

import sys
from pathlib import Path
import argparse
import subprocess
import time
from typing import List, Optional

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def run_script(script_path: str, description: str) -> bool:
    """
    Запускает скрипт и возвращает результат выполнения.
    
    Args:
        script_path: Путь к скрипту
        description: Описание скрипта
    
    Returns:
        bool: True если скрипт выполнен успешно
    """
    print(f"\n{'='*60}")
    print(f"ЗАПУСК: {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"{description} завершен успешно за {duration:.1f} секунд")
        
        if result.stdout:
            print("\nВывод скрипта:")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"{description} завершен с ошибкой за {duration:.1f} секунд")
        print(f"Код ошибки: {e.returncode}")
        
        if e.stdout:
            print("\nВывод скрипта:")
            print(e.stdout)
        
        if e.stderr:
            print("\nОшибки:")
            print(e.stderr)
        
        return False
    
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"{description} завершен с исключением за {duration:.1f} секунд")
        print(f"Ошибка: {e}")
        
        return False


def check_data_exists() -> bool:
    """
    Проверяет наличие исходных данных.
    
    Returns:
        bool: True если данные найдены
    """
    data_path = project_root / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
    
    if not data_path.exists():
        print(f"Исходные данные не найдены: {data_path}")
        print("Пожалуйста, поместите файл accepted_2007_to_2018Q4.csv в папку data/raw/")
        return False
    
    print(f"Исходные данные найдены: {data_path}")
    return True


def create_directories() -> None:
    """Создает необходимые директории."""
    directories = [
        "data/processed",
        "models/trained",
        "models/artifacts",
        "logs"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Создана директория: {directory}")


def main():
    """Основная функция для запуска пайплайна."""
    parser = argparse.ArgumentParser(description="Запуск пайплайна кредитного скоринга")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["eda", "preprocessing", "training", "tuning", "validation", "all"],
        default=["all"],
        help="Шаги для выполнения (по умолчанию: all)"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Пропустить проверки"
    )
    
    args = parser.parse_args()
    
    print("🚀 ЗАПУСК ПАЙПЛАЙНА КРЕДИТНОГО СКОРИНГА")
    print("="*60)
    
    # Проверяем наличие данных
    if not args.skip_checks and not check_data_exists():
        return 1
    
    # Создаем необходимые директории
    create_directories()
    
    # Определяем шаги для выполнения
    if "all" in args.steps:
        steps = ["eda", "preprocessing", "training", "tuning", "validation"]
    else:
        steps = args.steps
    
    # Пути к скриптам
    scripts = {
        "eda": "scripts/data_processing/eda.py",
        "preprocessing": "scripts/data_processing/preprocessing.py",
        "training": "scripts/model_training/train_models.py",
        "tuning": "scripts/model_training/hyperparameter_tuning.py",
        "validation": "scripts/model_training/validation.py"
    }
    
    descriptions = {
        "eda": "Исследовательский анализ данных (EDA)",
        "preprocessing": "Предобработка данных",
        "training": "Обучение моделей",
        "tuning": "Подбор гиперпараметров",
        "validation": "Валидация моделей"
    }
    
    # Выполняем шаги
    successful_steps = []
    failed_steps = []
    
    for step in steps:
        if step not in scripts:
            print(f"⚠️  Неизвестный шаг: {step}")
            continue
        
        script_path = scripts[step]
        description = descriptions[step]
        
        if run_script(script_path, description):
            successful_steps.append(step)
        else:
            failed_steps.append(step)
            
            # Спрашиваем, продолжать ли выполнение
            if step != "validation":  # Не спрашиваем после последнего шага
                response = input(f"\n Продолжить выполнение после ошибки в шаге '{step}'? (y/n): ")
                if response.lower() not in ['y', 'yes', 'да', 'д']:
                    break
    
    # Выводим итоговый отчет
    print(f"\n{'='*60}")
    print("ИТОГОВЫЙ ОТЧЕТ")
    print(f"{'='*60}")
    
    if successful_steps:
        print(f"Успешно выполнено: {', '.join(successful_steps)}")
    
    if failed_steps:
        print(f"Завершено с ошибками: {', '.join(failed_steps)}")
    
    if not failed_steps:
        print("\nПайплайн выполнен успешно!")
        print("\nРезультаты сохранены в:")
        print("  - data/processed/ - обработанные данные")
        print("  - models/trained/ - обученные модели")
        print("  - models/artifacts/ - графики и отчеты")
        
        return 0
    else:
        print(f"\n Пайплайн завершен с ошибками в {len(failed_steps)} шагах")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
