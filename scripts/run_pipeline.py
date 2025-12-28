#!/usr/bin/env python3
"""
Главный скрипт для запуска полного пайплайна кредитного скоринга.

Этот скрипт последовательно выполняет:
1. EDA (Exploratory Data Analysis)
2. Подготовку данных (стандартную или с кастомными признаками)
3. Обучение 4 моделей (LogReg, RF, LightGBM, Neural Network)
4. Подбор гиперпараметров для всех моделей
5. Валидацию моделей
6. Конвертацию в ONNX
7. Оптимизацию моделей
8. Валидацию ONNX конвертации
9. Углубленный бенчмарк производительности
10. Нагрузочное тестирование
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def run_script(script_path: str, description: str, timeout: int = None, step: str = None) -> bool:
    """
    Запускает скрипт и возвращает результат выполнения.

    Args:
        script_path: Путь к скрипту
        description: Описание скрипта
        timeout: Таймаут выполнения в секундах (None для автоматического определения)
        step: Имя шага для определения таймаута

    Returns:
        bool: True если скрипт выполнен успешно
    """
    print(f"\n{'=' * 60}")
    print(f"ЗАПУСК: {description}")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        # Автоматическое определение таймаутов по типу шага
        if timeout is None and step is not None:
            if step == "tuning":
                timeout = 1200  # 20 минут для подбора параметров
                print(f"Установлен увеличенный таймаут: {timeout} секунд (20 минут) для подбора гиперпараметров")
            elif step == "load_test":
                timeout = 1200  # 20 минут для нагрузочного тестирования
                print(f"Установлен увеличенный таймаут: {timeout} секунд (20 минут) для нагрузочного тестирования")
            elif step == "training":
                timeout = 900  # 15 минут для обучения моделей
                print(f"Установлен увеличенный таймаут: {timeout} секунд (15 минут) для обучения моделей")
            elif step == "optimize":
                timeout = 1800  # 30 минут для оптимизации
                print(f"Установлен увеличенный таймаут: {timeout} секунд (30 минут) для оптимизации моделей")
            elif step in ["eda", "data_prep"]:
                timeout = 600  # 10 минут для подготовки данных
                print(f"Установлен таймаут: {timeout} секунд (10 минут)")
            else:
                timeout = 300  # 5 минут по умолчанию

        result = subprocess.run(
            [sys.executable, script_path],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"{description} завершен успешно за {duration:.1f} секунд")

        if result.stdout:
            print("\nВывод скрипта:")
            # Показываем последние 1000 символов чтобы не перегружать консоль
            print(result.stdout[-1000:])

        return True

    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time

        print(f"{description} ТАЙМАУТ за {duration:.1f} секунд")
        print(f"Использованный таймаут: {timeout} секунд")

        if step == "tuning":
            print("\nВНИМАНИЕ: Подбор гиперпараметров занял слишком много времени.")
            print("Рекомендации:")
            print("1. Уменьшите размер сетки параметров")
            print("2. Используйте RandomizedSearchCV вместо GridSearchCV")
            print("3. Уменьшите количество итераций или cv слотов")
            print("4. Увеличьте таймаут при следующем запуске (--timeout-tune 1800)")

        elif step == "load_test":
            print("\nВНИМАНИЕ: Нагрузочное тестирование заняло слишком много времени.")
            print("Рекомендации:")
            print("1. Уменьшите количество запросов в load_testing.py")
            print("2. Уменьшите размер тестового набора данных")
            print("3. Увеличьте таймаут при следующем запуске (--timeout-load 1800)")

        elif step == "optimize":
            print("\nВНИМАНИЕ: Оптимизация моделей заняла слишком много времени.")
            print("Рекомендации:")
            print("1. Уменьшите количество экспериментов оптимизации")
            print("2. Используйте только один тип оптимизации за раз")
            print("3. Увеличьте таймаут при следующем запуске (--timeout-optimize 2400)")

        return False

    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"{description} завершен с ошибкой за {duration:.1f} секунд")
        print(f"Код ошибки: {e.returncode}")

        if e.stdout:
            print("\nВывод скрипта:")
            print(e.stdout[-500:])

        if e.stderr:
            print("\nОшибки:")
            print(e.stderr[-500:])

        return False

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"{description} завершен с исключением за {duration:.1f} секунд")
        print(f"Ошибка: {e}")

        return False


def run_advanced_features(
        use_custom_features: bool,
        include_optimization: bool = False,
        optimization_types: List[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Запускает расширенные возможности: ONNX, оптимизацию, валидацию, бенчмарк и нагрузочное тестирование.

    Args:
        use_custom_features: Использовать кастомные признаки
        include_optimization: Включить оптимизацию моделей
        optimization_types: Типы оптимизации для применения

    Returns:
        Tuple[List[str], List[str]]: Успешные и неудачные шаги
    """
    print(f"\n{'=' * 60}")
    print("ЗАПУСК РАСШИРЕННЫХ ВОЗМОЖНОСТЕЙ")
    print(f"{'=' * 60}")

    successful = []
    failed = []

    # 1. Конвертация моделей в ONNX
    print(f"\n{'=' * 60}")
    print("КОНВЕРТАЦИЯ МОДЕЛЕЙ В ONNX ФОРМАТ")
    print("=" * 60)

    try:
        onnx_dir = project_root / "models" / "onnx_final"
        onnx_dir.mkdir(parents=True, exist_ok=True)

        # Запускаем конвертацию
        print("Запуск конвертации моделей в ONNX...")
        converter_script = project_root / "scripts" / "model_training" / "convert_models_to_onnx.py"

        if converter_script.exists():
            if run_script(str(converter_script), "Конвертация в ONNX", 600, "onnx"):
                successful.append("ONNX_CONVERSION")
                print("ONNX конвертация успешно завершена")
            else:
                failed.append("ONNX_CONVERSION")
                print("Ошибка конвертации в ONNX")
        else:
            print(f"Скрипт конвертации не найден: {converter_script}")
            failed.append("ONNX_CONVERSION")
    except Exception as e:
        print(f"Ошибка при подготовке конвертации: {e}")
        failed.append("ONNX_CONVERSION")

    # 2. Валидация ONNX конвертации (КРИТИЧЕСКИЙ ШАГ)
    print(f"\n{'=' * 60}")
    print("ВАЛИДАЦИЯ КОРРЕКТНОСТИ ONNX КОНВЕРТАЦИИ")
    print("=" * 60)

    try:
        validate_script = project_root / "scripts" / "validation" / "validate_onnx_conversion.py"
        if validate_script.exists():
            print("Запуск валидации ONNX конвертации...")
            print("Это критический шаг для проверки корректности конвертации нейронной сети")

            if run_script(str(validate_script), "Валидация ONNX конвертации", 300, "validation"):
                successful.append("ONNX_VALIDATION")
                print("Валидация ONNX конвертации успешно пройдена")
            else:
                failed.append("ONNX_VALIDATION")
                print("Валидация ONNX конвертации завершилась с ошибками")
        else:
            print(f"Скрипт валидации не найден: {validate_script}")
            failed.append("ONNX_VALIDATION")
    except Exception as e:
        print(f"Ошибка при запуске валидации: {e}")
        failed.append("ONNX_VALIDATION")

    # 3. Оптимизация моделей (если включена)
    if include_optimization:
        print(f"\n{'=' * 60}")
        print("ОПТИМИЗАЦИЯ МОДЕЛЕЙ ДЛЯ ПРОДАКШЕНА")
        print("=" * 60)

        try:
            optimize_script = project_root / "scripts" / "model_training" / "optimize_model_fixed.py"

            if optimize_script.exists():
                print(f"Запуск оптимизации моделей...")
                print(f"Типы оптимизации: {optimization_types or ['all']}")

                if run_script(str(optimize_script), "Оптимизация моделей", 1800, "optimize"):
                    successful.append("OPTIMIZATION")
                    print("Оптимизация моделей завершена успешно")
                else:
                    failed.append("OPTIMIZATION")
                    print("Оптимизация моделей завершилась с ошибками")
            else:
                print(f"Скрипт оптимизации не найден: {optimize_script}")
                failed.append("OPTIMIZATION")

        except Exception as e:
            print(f"Ошибка при запуске оптимизации: {e}")
            failed.append("OPTIMIZATION")

    # 4. Углубленный бенчмарк производительности
    print(f"\n{'=' * 60}")
    print("УГЛУБЛЕННЫЙ БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)

    try:
        deep_benchmark_script = project_root / "scripts" / "performance" / "deep_benchmark.py"
        if deep_benchmark_script.exists():
            print("Запуск углубленного бенчмарка производительности...")
            print("Сравнение PyTorch vs ONNX на CPU")

            if run_script(str(deep_benchmark_script), "Углубленный бенчмарк", 600, "benchmark"):
                successful.append("DEEP_BENCHMARK")
                print("Углубленный бенчмарк завершен успешно")
            else:
                failed.append("DEEP_BENCHMARK")
                print("Углубленный бенчмарк завершился с ошибками")
        else:
            # Если нового скрипта нет, используем старый
            print("Углубленный бенчмарк не найден, используем стандартный...")
            benchmark_script = project_root / "scripts" / "performance" / "benchmark.py"
            if benchmark_script.exists():
                if run_script(str(benchmark_script), "Бенчмарк моделей", 300, "benchmark"):
                    successful.append("BENCHMARK")
                else:
                    failed.append("BENCHMARK")
            else:
                print(f"Скрипт бенчмарка не найден: {benchmark_script}")
                failed.append("BENCHMARK")
    except Exception as e:
        print(f"Ошибка при запуске бенчмарка: {e}")
        failed.append("BENCHMARK")

    # 5. Нагрузочное тестирование
    print(f"\n{'=' * 60}")
    print("НАГРУЗОЧНОЕ ТЕСТИРОВАНИЕ")
    print("=" * 60)

    try:
        load_test_script = project_root / "scripts" / "performance" / "load_testing.py"
        if load_test_script.exists():
            print("Нагрузочное тестирование может занять несколько минут...")
            if run_script(str(load_test_script), "Нагрузочное тестирование", 1200, "load_test"):
                successful.append("LOAD_TESTING")
                print("Нагрузочное тестирование завершено успешно")
            else:
                failed.append("LOAD_TESTING")
                print("Нагрузочное тестирование завершилось с ошибками")
        else:
            print(f"Скрипт нагрузочного тестирования не найден: {load_test_script}")
            failed.append("LOAD_TESTING")
    except Exception as e:
        print(f"Ошибка при запуске нагрузочного тестирования: {e}")
        failed.append("LOAD_TESTING")

    return successful, failed


def create_best_onnx_model():
    """Создает best_model.onnx на основе метрик качества."""
    try:
        onnx_dir = project_root / "models" / "onnx_final"
        if not onnx_dir.exists():
            print("Директория ONNX не найдена")
            return

        # Ищем все ONNX файлы
        onnx_files = list(onnx_dir.glob("*.onnx"))
        if not onnx_files:
            print("ONNX файлы не найдены")
            return

        # Исключаем уже существующий best_model.onnx из поиска
        onnx_files = [f for f in onnx_files if f.name != "best_model.onnx"]

        if not onnx_files:
            print("Нет других ONNX файлов для выбора лучшей модели")
            return

        # Предпочтение: neural_network.onnx как обязательная модель
        neural_network_path = onnx_dir / "neural_network.onnx"
        if neural_network_path.exists():
            best_model_path = neural_network_path
            best_model_name = "Neural Network (обязательная модель)"
        else:
            # Если нет нейронной сети, выбираем по размеру/производительности
            onnx_files.sort(key=lambda x: x.stat().st_size)
            best_model_path = onnx_files[0]
            best_model_name = f"{best_model_path.name} (самая маленькая)"

        # Копируем как best_model.onnx
        best_path = onnx_dir / "best_model.onnx"
        shutil.copy2(best_model_path, best_path)

        print(f"Лучшая ONNX модель создана:")
        print(f"  Исходная: {best_model_path.name}")
        print(f"  Копия: {best_path.name}")
        print(f"  Размер: {best_path.stat().st_size / 1024:.1f} KB")
        print(f"  Причина выбора: {best_model_name}")

        # Создаем конфигурационный файл
        config = {
            "best_model": best_model_path.name,
            "selected_reason": best_model_name,
            "size_kb": best_path.stat().st_size / 1024,
            "available_models": [f.name for f in onnx_dir.glob("*.onnx")],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        config_path = onnx_dir / "best_model_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"Ошибка при создании best_model.onnx: {e}")
        import traceback
        traceback.print_exc()


def copy_best_optimized_model(optimization_types):
    """Копирует лучшую оптимизированную модель в production."""
    try:
        optimized_dir = project_root / "models" / "optimized"
        production_dir = project_root / "production"

        if not optimized_dir.exists():
            print("Директория с оптимизированными моделями не найдена")
            return

        production_dir.mkdir(parents=True, exist_ok=True)

        # Предпочтение нейронной сети
        nn_onnx_path = optimized_dir / "neural_network.onnx"
        if nn_onnx_path.exists():
            production_model_path = production_dir / "best_optimized_model.onnx"
            shutil.copy2(nn_onnx_path, production_model_path)
            print(f"Нейронная сеть скопирована в production: {production_model_path}")
        else:
            # Ищем любую ONNX модель
            onnx_files = list(optimized_dir.glob("*.onnx"))
            if onnx_files:
                production_model_path = production_dir / "best_optimized_model.onnx"
                shutil.copy2(onnx_files[0], production_model_path)
                print(f"Оптимизированная модель скопирована: {production_model_path}")
            else:
                print("ONNX модели не найдены в optimized/")

    except Exception as e:
        print(f"Ошибка при копировании оптимизированной модели: {e}")
        import traceback
        traceback.print_exc()


def run_custom_data_preparation() -> bool:
    """Запускает подготовку данных с кастомными признаками через subprocess."""
    print(f"\n{'=' * 60}")
    print("ЗАПУСК: Подготовка данных с кастомными признаками")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        # Используем prepare_data_fixed.py для кастомных признаков
        script_path = project_root / "scripts" / "data_processing" / "prepare_data_fixed.py"

        data_path = project_root / "data" / "raw" / "UCI_Credit_Card.csv"
        output_path = project_root / "data" / "processed_fixed"

        if not script_path.exists():
            print(f"Скрипт не найден: {script_path}")
            return False

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--data-path",
                str(data_path),
                "--output-path",
                str(output_path),
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"Подготовка данных с кастомными признаками завершена за {duration:.1f} секунд")

        if result.stdout:
            print("\nВывод скрипта:")
            print(result.stdout[-500:])

        return True

    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"Подготовка данных завершена с ошибкой за {duration:.1f} секунд")
        print(f"Код ошибки: {e.returncode}")

        if e.stdout:
            print("\nВывод скрипта:")
            print(e.stdout[-500:])

        if e.stderr:
            print("\nОшибки:")
            print(e.stderr[-500:])

        return False

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"Подготовка данных завершена с исключением за {duration:.1f} секунд")
        print(f"Ошибка: {e}")
        return False


def check_data_exists() -> bool:
    """
    Проверяет наличие исходных данных.

    Returns:
        bool: True если данные найдены
    """
    data_path = project_root / "data" / "raw" / "UCI_Credit_Card.csv"

    if not data_path.exists():
        print(f"Исходные данные не найдены: {data_path}")
        print("Пожалуйста, поместите файл UCI_Credit_Card.csv в папку data/raw/")
        return False

    print(f"Исходные данные найдены: {data_path}")
    return True


def create_directories() -> None:
    """Создает необходимые директории."""
    directories = [
        "data/processed",
        "data/processed_fixed",
        "models/trained",
        "models/trained_final",
        "models/onnx_final",
        "models/optimized",
        "models/artifacts",
        "models/artifacts/benchmark_charts",
        "models/artifacts/benchmark_reports",
        "models/artifacts/benchmark_results",
        "models/artifacts/load_testing",
        "models/artifacts/optimization_reports",
        "models/artifacts/onnx_validation",
        "models/artifacts/deep_benchmark",
        "production",
        "production_optimized",
        "logs",
    ]

    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Создана директория: {directory}")


def create_deployment_instructions(use_custom_features: bool, optimize: bool) -> None:
    """
    Создает краткие инструкции по запуску API.

    Args:
        use_custom_features: Использовать кастомные признаки
        optimize: Была ли выполнена оптимизация
    """
    try:
        print(f"\n{'=' * 60}")
        print("ИНСТРУКЦИИ ПО ЗАПУСКУ API СЕРВЕРА")
        print("=" * 60)

        print("\nВаш FastAPI сервер готов к использованию:")
        print("  • Файл: scripts/api/credit_scoring_api.py")
        print("  • Порт: 8000 (по умолчанию)")

        print("\nЗапуск сервера:")
        print("  python scripts/api/credit_scoring_api.py")
        print("\nИли:")
        print("  uvicorn scripts.api.credit_scoring_api:app --host 0.0.0.0 --port 8000")

        print("\nБыстрая проверка:")
        print("  # Health check")
        print("  curl http://localhost:8000/api/v1/health")
        print("\n  # Пример предсказания")
        print('  curl -X POST http://localhost:8000/api/v1/predict \\')
        print('    -H "Content-Type: application/json" \\')
        print(
            '    -d \'{"limit_bal": 20000, "sex": 1, "marriage_new": 1, "age": 30, "pay_new": 0, "education_new": 2}\'')

        print("\nДокументация API:")
        print("  http://localhost:8000/docs")

        if optimize:
            print("\nПримечание: Используются оптимизированные модели из production_optimized/")
        elif use_custom_features:
            print("\nПримечание: Используются модели на кастомных признаках (6 фич)")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"Ошибка при выводе инструкций: {e}")


def main():
    """Основная функция для запуска пайплайна."""
    parser = argparse.ArgumentParser(description="Запуск пайплайна кредитного скоринга")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[
            "eda",
            "data_prep",
            "training",
            "tuning",
            "validation",
            "onnx",
            "validate_onnx",
            "deep_benchmark",
            "optimize",
            "benchmark",
            "load_test",
            "all",
            "custom",
            "advanced",
        ],
        default=["all"],
        help="Шаги для выполнения (по умолчанию: all)",
    )
    parser.add_argument(
        "--skip-checks", action="store_true", help="Пропустить проверки"
    )
    parser.add_argument(
        "--use-custom-features",
        action="store_true",
        help="Использовать кастомные признаки (только признаки доступные при заявке)",
    )
    parser.add_argument(
        "--cleanup-old",
        action="store_true",
        help="Очистить старые модели перед запуском",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Включить оптимизацию моделей (pruning, quantization)",
    )
    parser.add_argument(
        "--optimization-types",
        nargs="+",
        choices=["pruning", "quantization", "onnx", "all"],
        default=["all"],
        help="Типы оптимизации для применения",
    )
    parser.add_argument(
        "--timeout-tune",
        type=int,
        default=1200,
        help="Таймаут для подбора гиперпараметров в секундах (по умолчанию: 1200)",
    )
    parser.add_argument(
        "--timeout-load",
        type=int,
        default=1200,
        help="Таймаут для нагрузочного тестирования в секундах (по умолчанию: 1200)",
    )
    parser.add_argument(
        "--timeout-train",
        type=int,
        default=900,
        help="Таймаут для обучения моделей в секундах (по умолчанию: 900)",
    )
    parser.add_argument(
        "--timeout-optimize",
        type=int,
        default=1800,
        help="Таймаут для оптимизации моделей в секундах (по умолчанию: 1800)",
    )
    parser.add_argument(
        "--no-timeout",
        action="store_true",
        help="Отключить таймаут полностью (может зависнуть)",
    )

    args = parser.parse_args()

    print("ЗАПУСК ПАЙПЛАЙНА КРЕДИТНОГО СКОРИНГА")
    print("=" * 60)
    print("ПОЛНАЯ ВЕРСИЯ: 4 модели, ONNX, Валидация, Бенчмарк, Нагрузочное тестирование")
    print("=" * 60)

    if args.use_custom_features:
        print("РЕЖИМ: Использование кастомных признаков")
        print("Будут использоваться только признаки доступные при подаче заявки:")
        print("  - limit_bal (кредитный лимит)")
        print("  - sex (пол)")
        print("  - marriage_new (семейное положение)")
        print("  - age (возраст)")
        print("  - pay_new (поведение платежей)")
        print("  - education_new (уровень образования)")
    else:
        print("РЕЖИМ: Стандартные признаки")
        print("Будут использоваться все доступные признаки из данных")

    # Проверяем наличие данных
    if not args.skip_checks and not check_data_exists():
        return 1

    # Создаем необходимые директории
    create_directories()

    # Определяем шаги для выполнения
    if "all" in args.steps:
        basic_steps = ["eda", "data_prep", "training", "tuning", "validation"]
        advanced_steps = ["onnx", "validate_onnx", "deep_benchmark", "benchmark", "load_test"]
        if args.optimize:
            advanced_steps.append("optimize")
        steps = basic_steps + advanced_steps
    elif "custom" in args.steps:
        steps = ["data_prep", "training", "tuning", "validation"]
        args.use_custom_features = True
    elif "advanced" in args.steps:
        steps = ["onnx", "validate_onnx", "deep_benchmark", "load_test"]
        if args.optimize:
            steps.append("optimize")
    else:
        steps = args.steps

    # Пути к скриптам
    if args.use_custom_features:
        scripts = {
            "eda": "scripts/data_processing/eda.py",
            "data_prep": None,
            "training": "scripts/model_training/train_all_models.py",
            "tuning": "scripts/model_training/hyperparameter_tuning_all_models.py",
            "validation": "scripts/model_training/validation_all_models.py",
            "onnx": "scripts/model_training/convert_models_to_onnx.py",
            "validate_onnx": "scripts/validation/validate_onnx_conversion.py",
            "deep_benchmark": "scripts/performance/deep_benchmark.py",
            "optimize": "scripts/model_training/optimize_model_fixed.py",
            "benchmark": "scripts/performance/benchmark.py",
            "load_test": "scripts/performance/load_testing.py",
        }
    else:
        scripts = {
            "eda": "scripts/data_processing/eda.py",
            "data_prep": "scripts/data_processing/preprocessing.py",
            "training": "scripts/model_training/train_models.py",
            "tuning": "scripts/model_training/hyperparameter_tuning.py",
            "validation": "scripts/model_training/validation.py",
            "onnx": "scripts/model_training/convert_models_to_onnx.py",
            "validate_onnx": "scripts/validation/validate_onnx_conversion.py",
            "deep_benchmark": "scripts/performance/deep_benchmark.py",
            "optimize": "scripts/model_training/optimize_model_fixed.py",
            "benchmark": "scripts/performance/benchmark.py",
            "load_test": "scripts/performance/load_testing.py",
        }

    descriptions = {
        "eda": "Исследовательский анализ данных (EDA)",
        "data_prep": (
            "Подготовка данных с кастомными признаками"
            if args.use_custom_features
            else "Предобработка данных"
        ),
        "training": (
            "Обучение 4 моделей на кастомных признаках (LogReg, RF, LightGBM, NN)"
            if args.use_custom_features
            else "Обучение моделей"
        ),
        "tuning": (
            "Подбор гиперпараметров для кастомных признаков"
            if args.use_custom_features
            else "Подбор гиперпараметров"
        ),
        "validation": (
            "Валидация моделей с кастомными признаками"
            if args.use_custom_features
            else "Валидация моделей"
        ),
        "onnx": "Конвертация моделей в ONNX формат",
        "validate_onnx": "Валидация корректности ONNX конвертации (обязательно для нейронной сети)",
        "deep_benchmark": "Углубленный бенчмарк производительности (PyTorch vs ONNX на CPU)",
        "optimize": "Оптимизация моделей (pruning, quantization)",
        "benchmark": "Бенчмарк производительности моделей",
        "load_test": "Нагрузочное тестирование лучшей модели",
    }

    # Выполняем шаги
    successful_steps = []
    failed_steps = []

    for step in steps:
        if step not in scripts and step not in ["onnx", "validate_onnx", "deep_benchmark", "benchmark", "load_test",
                                                "optimize"]:
            print(f"Неизвестный шаг: {step}")
            continue

        print(f"\n{'=' * 60}")
        print(f"ШАГ {len(successful_steps) + len(failed_steps) + 1}: {step.upper()}")
        print("=" * 60)

        # Особый случай для подготовки данных в кастомном режиме
        if step == "data_prep" and args.use_custom_features:
            if run_custom_data_preparation():
                successful_steps.append(step)
            else:
                failed_steps.append(step)

        # Особый случай для расширенных возможностей
        elif step in ["onnx", "validate_onnx", "deep_benchmark", "benchmark", "load_test", "optimize"]:
            # Эти шаги обрабатываются в run_advanced_features
            pass

        else:
            script_path = scripts[step]
            description = descriptions[step]

            if script_path is None:
                print(f"Скрипт для шага '{step}' не определен")
                failed_steps.append(step)
                continue

            # Определяем кастомный таймаут
            custom_timeout = None
            if not args.no_timeout:
                if step == "tuning":
                    custom_timeout = args.timeout_tune
                elif step == "load_test":
                    custom_timeout = args.timeout_load
                elif step == "training":
                    custom_timeout = args.timeout_train
                elif step == "optimize":
                    custom_timeout = args.timeout_optimize

            if run_script(script_path, description, custom_timeout, step):
                successful_steps.append(step)
            else:
                failed_steps.append(step)

        # Спрашиваем, продолжать ли выполнение после ошибки
        if step not in ["onnx", "validate_onnx", "deep_benchmark", "benchmark", "load_test",
                        "optimize"] and failed_steps:
            if sys.stdin.isatty():
                response = input(
                    f"\nПродолжить выполнение после ошибки в шаге '{step}'? (y/n): "
                )
                if response.lower() not in ["y", "yes", "да", "д"]:
                    break
            else:
                print(
                    f"\nАвтоматически продолжаем выполнение после ошибки в шаге '{step}'..."
                )

    # Запускаем расширенные возможности
    if any(step in steps for step in ["onnx", "validate_onnx", "deep_benchmark", "benchmark", "load_test", "optimize"]):
        print(f"\n{'=' * 60}")
        print("ЗАПУСК РАСШИРЕННЫХ ВОЗМОЖНОСТЕЙ")
        print("=" * 60)

        include_optimization = args.optimize or "optimize" in steps
        advanced_successful, advanced_failed = run_advanced_features(
            use_custom_features=args.use_custom_features,
            include_optimization=include_optimization,
            optimization_types=args.optimization_types if args.optimization_types != ["all"] else None
        )

        successful_steps.extend(advanced_successful)
        failed_steps.extend(advanced_failed)

    # Выводим итоговый отчет
    print(f"\n{'=' * 60}")
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 60)

    if successful_steps:
        print(f"Успешно выполнено ({len(successful_steps)}):")
        for step in successful_steps:
            print(f"  • {step}")

    if failed_steps:
        print(f"Завершено с ошибками ({len(failed_steps)}):")
        for step in failed_steps:
            print(f"  • {step}")

    if not failed_steps:
        print("\nПАЙПЛАЙН ВЫПОЛНЕН УСПЕШНО!")
        print("\nРЕЗУЛЬТАТЫ СОХРАНЕНЫ В:")

        if args.use_custom_features:
            print("  • data/processed_fixed/     - обработанные данные (6 признаков)")
            print("  • models/trained_final/     - 4 обученные модели (LogReg, RF, LightGBM, NN)")
            print("  • models/onnx_final/        - ONNX модели для продакшена")
            if args.optimize:
                print("  • models/optimized/        - оптимизированные модели")
                print("  • production_optimized/    - готовые для продакшена оптимизированные модели")
            print("  • models/artifacts/         - графики, отчеты и метрики")
        else:
            print("  • data/processed/           - обработанные данные")
            print("  • models/trained/           - обученные модели")
            print("  • models/artifacts/         - графики и отчеты")

        # Показываем информацию о продвинутых возможностях
        advanced_capabilities = []
        if "ONNX_CONVERSION" in successful_steps:
            advanced_capabilities.append("ONNX модели готовы для продакшена")
        if "ONNX_VALIDATION" in successful_steps:
            advanced_capabilities.append("Валидация ONNX конвертации пройдена")
        if "DEEP_BENCHMARK" in successful_steps or "BENCHMARK" in successful_steps:
            advanced_capabilities.append("Бенчмарк производительности выполнен")
        if "OPTIMIZATION" in successful_steps:
            advanced_capabilities.append("Оптимизированные модели созданы")
        if "LOAD_TESTING" in successful_steps:
            advanced_capabilities.append("Нагрузочное тестирование завершено")

        if advanced_capabilities:
            print("\nРАСШИРЕННЫЕ ВОЗМОЖНОСТИ:")
            for capability in advanced_capabilities:
                print(f"  • {capability}")

        print("\nРЕКОМЕНДАЦИИ ДЛЯ ПРОДАКШЕНА:")
        print("  1. Используйте neural_network.onnx - обязательная модель для проекта")
        print("  2. Для REST API используйте best_model.onnx")
        print("  3. Проверьте результаты валидации в models/artifacts/onnx_validation/")
        print("  4. Оптимизируйте порог классификации под вашу бизнес-логику")

        # Показываем краткие инструкции по запуску API
        create_deployment_instructions(args.use_custom_features, args.optimize)

        return 0
    else:
        print(f"\nПАЙПЛАЙН ЗАВЕРШЕН С ОШИБКАМИ В {len(failed_steps)} ШАГАХ")

        # Даже с ошибками, может быть полезная информация
        if successful_steps:
            print("\nЧАСТИЧНЫЕ РЕЗУЛЬТАТЫ:")
            for step in successful_steps:
                print(f"  • {step} - выполнено успешно")

        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)