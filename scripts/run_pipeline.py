"""
Главный скрипт для запуска полного пайплайна кредитного скоринга.

Этот скрипт последовательно выполняет:
1. EDA (Exploratory Data Analysis)
2. Подготовку данных (стандартную или с кастомными признаками)
3. Обучение моделей
4. Подбор гиперпараметров
5. Валидацию моделей
"""

import argparse
import subprocess
import sys
import time
import shutil
from pathlib import Path
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
    print(f"\n{'=' * 60}")
    print(f"ЗАПУСК: {description}")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
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


def run_custom_data_preparation() -> bool:
    """
    Запускает подготовку данных с кастомными признаками через subprocess.
    """
    print(f"\n{'=' * 60}")
    print("ЗАПУСК: Подготовка данных с кастомными признаками")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        # Запускаем prepare_data.py как отдельный скрипт
        script_path = project_root / "scripts" / "prepare_data.py"
        data_path = project_root / "data" / "raw" / "UCI_Credit_Card.csv"
        output_path = project_root / "data" / "processed_custom"

        if not script_path.exists():
            print(f"✗ Скрипт не найден: {script_path}")
            return False

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--data-path", str(data_path),
                "--output-path", str(output_path)
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"Подготовка данных с кастомными признаками завершена успешно за {duration:.1f} секунд")

        if result.stdout:
            print("\nВывод скрипта:")
            print(result.stdout)

        # Исправляем структуру папок для совместимости
        fix_custom_data_structure(output_path)

        # Проверяем созданные признаки
        feature_info_path = output_path / "artifacts" / "feature_info.pkl"
        if feature_info_path.exists():
            import joblib
            feature_info = joblib.load(feature_info_path)
            print(f"\nИспользованные признаки ({len(feature_info['application_features'])}):")
            for feature in feature_info['application_features']:
                print(f"  - {feature}")

        return True

    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"Подготовка данных с кастомными признаками завершена с ошибкой за {duration:.1f} секунд")
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

        print(f"Подготовка данных с кастомными признаками завершена с исключением за {duration:.1f} секунд")
        print(f"Ошибка: {e}")
        return False


def fix_custom_data_structure(output_path: Path):
    """
    Исправляет структуру папок кастомных данных для совместимости со скриптами обучения.
    Перемещает файлы из processed_custom/processed/ в processed_custom/
    """
    processed_subfolder = output_path / "processed"

    if processed_subfolder.exists():
        print("Исправление структуры папок для совместимости...")

        # Перемещаем все файлы из подпапки processed в основную папку
        for file_path in processed_subfolder.glob("*"):
            if file_path.is_file():
                target_path = output_path / file_path.name
                shutil.move(str(file_path), str(target_path))
                print(f"  Перемещен: {file_path.name}")

        # Удаляем пустую подпапку processed
        try:
            processed_subfolder.rmdir()
            print("  Удалена пустая подпапка processed")
        except OSError:
            print("  Не удалось удалить подпапку processed (возможно, не пустая)")

        print("Структура папок исправлена")


def check_data_exists() -> bool:
    """
    Проверяет наличие исходных данных.

    Returns:
        bool: True если данные найдены
    """
    data_path = project_root / "data" / "raw" / "UCI_Credit_Card.csv"

    if not data_path.exists():
        print(f"Исходные данные не найдены: {data_path}")
        print(
            "Пожалуйста, поместите файл UCI_Credit_Card.csv в папку data/raw/"
        )
        return False

    print(f"Исходные данные найдены: {data_path}")
    return True


def create_directories() -> None:
    """Создает необходимые директории."""
    directories = [
        "data/processed",
        "data/processed_custom",
        "models/trained",
        "models/trained_custom",
        "models/artifacts",
        "models/artifacts_custom",
        "logs"
    ]

    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Создана директория: {directory}")


def update_training_scripts_for_custom_features():
    """
    Создает временные версии скриптов обучения для работы с кастомными признаками.
    Заменяет XGBoost на CatBoost в создаваемых скриптах.
    """
    try:
        # Обновляем пути в скрипте обучения и заменяем XGBoost на CatBoost
        train_script_path = project_root / "scripts" / "model_training" / "train_models_custom.py"

        if not train_script_path.exists():
            # Создаем кастомную версию скрипта обучения
            original_script = project_root / "scripts" / "model_training" / "train_models.py"
            if original_script.exists():
                content = original_script.read_text(encoding='utf-8')
                # Заменяем пути на кастомные
                content = content.replace('data/processed', 'data/processed_custom')
                content = content.replace('models/trained', 'models/trained_custom')
                content = content.replace('models/artifacts', 'models/artifacts_custom')
                # Заменяем XGBoost на CatBoost
                content = content.replace('xgboost', 'catboost')
                content = content.replace('XGBoost', 'CatBoost')
                content = content.replace('XGBClassifier', 'CatBoostClassifier')
                train_script_path.write_text(content, encoding='utf-8')
                print("✓ Создан train_models_custom.py с CatBoost")

        # Аналогично для скрипта подбора гиперпараметров
        tuning_script_path = project_root / "scripts" / "model_training" / "hyperparameter_tuning_custom.py"
        original_tuning_script = project_root / "scripts" / "model_training" / "hyperparameter_tuning.py"
        if original_tuning_script.exists() and not tuning_script_path.exists():
            content = original_tuning_script.read_text(encoding='utf-8')
            content = content.replace('data/processed', 'data/processed_custom')
            content = content.replace('models/trained', 'models/trained_custom')
            content = content.replace('models/artifacts', 'models/artifacts_custom')
            # Заменяем XGBoost на CatBoost
            content = content.replace('xgboost', 'catboost')
            content = content.replace('XGBoost', 'CatBoost')
            content = content.replace('XGBClassifier', 'CatBoostClassifier')
            tuning_script_path.write_text(content, encoding='utf-8')
            print("✓ Создан hyperparameter_tuning_custom.py с CatBoost")

        # Для валидации просто обновляем пути
        validation_script_path = project_root / "scripts" / "model_training" / "validation_custom.py"
        original_validation_script = project_root / "scripts" / "model_training" / "validation.py"
        if original_validation_script.exists() and not validation_script_path.exists():
            content = original_validation_script.read_text(encoding='utf-8')
            content = content.replace('data/processed', 'data/processed_custom')
            content = content.replace('models/trained', 'models/trained_custom')
            content = content.replace('models/artifacts', 'models/artifacts_custom')
            validation_script_path.write_text(content, encoding='utf-8')
            print("✓ Создан validation_custom.py")

        return True

    except Exception as e:
        print(f"Ошибка при обновлении скриптов: {e}")
        return False


def cleanup_old_xgboost_models():
    """Удаляет старые файлы моделей XGBoost если они существуют."""
    try:
        models_dirs = [
            project_root / "models" / "trained_custom",
            project_root / "models" / "trained"
        ]

        xgboost_files = [
            "xgboost.pkl", "tuned_xgbclassifier.pkl",
            "best_model.pkl", "best_tuned_model.pkl"
        ]

        for models_dir in models_dirs:
            if models_dir.exists():
                for xgboost_file in xgboost_files:
                    file_path = models_dir / xgboost_file
                    if file_path.exists():
                        file_path.unlink()
                        print(f"🗑️ Удален старый файл: {file_path}")

        print("✓ Очистка старых моделей XGBoost завершена")
    except Exception as e:
        print(f"⚠️ Ошибка при очистке старых моделей: {e}")


def check_catboost_installation():
    """Проверяет установлен ли CatBoost."""
    try:
        import catboost
        print("✅ CatBoost установлен")
        return True
    except ImportError:
        print("❌ CatBoost не установлен. Установите: pip install catboost")
        return False


def main():
    """Основная функция для запуска пайплайна."""
    parser = argparse.ArgumentParser(description="Запуск пайплайна кредитного скоринга")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["eda", "data_prep", "training", "tuning", "validation", "all", "custom"],
        default=["all"],
        help="Шаги для выполнения (по умолчанию: all)",
    )
    parser.add_argument(
        "--skip-checks", action="store_true", help="Пропустить проверки"
    )
    parser.add_argument(
        "--use-custom-features", action="store_true",
        help="Использовать кастомные признаки (только признаки доступные при заявке)"
    )
    parser.add_argument(
        "--cleanup-old", action="store_true",
        help="Очистить старые модели XGBoost перед запуском"
    )

    args = parser.parse_args()

    print("ЗАПУСК ПАЙПЛАЙНА КРЕДИТНОГО СКОРИНГА")
    print("=" * 60)
    print("🔥 ОБНОВЛЕНИЕ: Используется CatBoost вместо XGBoost")
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

    # Проверяем установку CatBoost
    if not check_catboost_installation():
        return 1

    # Очищаем старые модели если нужно
    if args.cleanup_old:
        cleanup_old_xgboost_models()

    # Создаем необходимые директории
    create_directories()

    # Определяем шаги для выполнения
    if "all" in args.steps:
        steps = ["eda", "data_prep", "training", "tuning", "validation"]
    elif "custom" in args.steps:
        steps = ["data_prep", "training", "tuning", "validation"]
        args.use_custom_features = True
    else:
        steps = args.steps

    # Обновляем скрипты если используем кастомные признаки
    if args.use_custom_features:
        print("\nПодготовка скриптов для кастомных признаков...")
        if update_training_scripts_for_custom_features():
            print("✓ Скрипты обновлены для кастомных признаков")
            print("✓ XGBoost заменен на CatBoost в кастомных скриптах")
        else:
            print("✗ Ошибка обновления скриптов")

    # Пути к скриптам (меняем в зависимости от режима)
    if args.use_custom_features:
        scripts = {
            "eda": "scripts/data_processing/eda.py",  # EDA остается тем же
            "data_prep": None,  # Будет использовать функцию
            "training": "scripts/model_training/train_models_custom.py",
            "tuning": "scripts/model_training/hyperparameter_tuning_custom.py",
            "validation": "scripts/model_training/validation_custom.py",
        }
    else:
        scripts = {
            "eda": "scripts/data_processing/eda.py",
            "data_prep": "scripts/data_processing/preprocessing.py",
            "training": "scripts/model_training/train_models.py",
            "tuning": "scripts/model_training/hyperparameter_tuning.py",
            "validation": "scripts/model_training/validation.py",
        }

    descriptions = {
        "eda": "Исследовательский анализ данных (EDA)",
        "data_prep": "Подготовка данных с кастомными признаками" if args.use_custom_features else "Предобработка данных",
        "training": "Обучение моделей CatBoost на кастомных признаках" if args.use_custom_features else "Обучение моделей CatBoost",
        "tuning": "Подбор гиперпараметров CatBoost для кастомных признаков" if args.use_custom_features else "Подбор гиперпараметров CatBoost",
        "validation": "Валидация моделей CatBoost с кастомными признаками" if args.use_custom_features else "Валидация моделей CatBoost",
    }

    # Выполняем шаги
    successful_steps = []
    failed_steps = []

    for step in steps:
        if step not in scripts and step != "data_prep":
            print(f"Неизвестный шаг: {step}")
            continue

        # Особый случай для подготовки данных в кастомном режиме
        if step == "data_prep" and args.use_custom_features:
            if run_custom_data_preparation():
                successful_steps.append(step)
            else:
                failed_steps.append(step)
        else:
            script_path = scripts[step]
            description = descriptions[step]

            if run_script(script_path, description):
                successful_steps.append(step)
            else:
                failed_steps.append(step)

        # Спрашиваем, продолжать ли выполнение после ошибки
        if step != "validation" and failed_steps:
            if sys.stdin.isatty():
                response = input(
                    f"\nПродолжить выполнение после ошибки в шаге '{step}'? (y/n): "
                )
                if response.lower() not in ["y", "yes", "да", "д"]:
                    break
            else:
                print(f"\nАвтоматически продолжаем выполнение после ошибки в шаге '{step}'...")

    # Выводим итоговый отчет
    print(f"\n{'=' * 60}")
    print("ИТОГОВЫЙ ОТЧЕТ")
    print(f"{'=' * 60}")

    if successful_steps:
        print(f"✅ Успешно выполнено: {', '.join(successful_steps)}")

    if failed_steps:
        print(f"❌ Завершено с ошибками: {', '.join(failed_steps)}")

    if not failed_steps:
        print("\n🎉 Пайплайн выполнен успешно!")
        print("\n📁 Результаты сохранены в:")
        if args.use_custom_features:
            print("  - data/processed_custom/ - обработанные данные с кастомными признаками")
            print("  - models/trained_custom/ - обученные модели CatBoost")
            print("  - models/artifacts_custom/ - графики и отчеты")
            print(f"\n🔧 Использовано 6 признаков доступных при заявке")
            print("🤖 Основная модель: CatBoost с оптимизацией под AUC")
        else:
            print("  - data/processed/ - обработанные данные")
            print("  - models/trained/ - обученные модели CatBoost")
            print("  - models/artifacts/ - графики и отчеты")

        return 0
    else:
        print(f"\n💥 Пайплайн завершен с ошибками в {len(failed_steps)} шагах")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)