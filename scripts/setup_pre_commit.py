#!/usr/bin/env python3
"""
Скрипт для настройки pre-commit hooks в проекте кредитного скоринга.

Этот скрипт:
1. Устанавливает pre-commit
2. Настраивает hooks
3. Создает baseline для detect-secrets
4. Запускает проверки
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Выполняет команду и выводит результат."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"✅ {description} - успешно")
        if result.stdout:
            print(f"   Вывод: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - ошибка")
        print(f"   Код ошибки: {e.returncode}")
        if e.stdout:
            print(f"   Вывод: {e.stdout.strip()}")
        if e.stderr:
            print(f"   Ошибка: {e.stderr.strip()}")
        return False


def check_python_version():
    """Проверяет версию Python."""
    print("🐍 Проверка версии Python...")
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8 или выше")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def install_pre_commit():
    """Устанавливает pre-commit."""
    return run_command(
        "pip install pre-commit",
        "Установка pre-commit"
    )


def install_hooks():
    """Устанавливает pre-commit hooks."""
    return run_command(
        "pre-commit install",
        "Установка pre-commit hooks"
    )


def install_hooks_ci():
    """Устанавливает pre-commit hooks для CI."""
    return run_command(
        "pre-commit install --hook-type pre-push",
        "Установка pre-commit hooks для CI"
    )


def create_secrets_baseline():
    """Создает baseline для detect-secrets."""
    if not Path(".secrets.baseline").exists():
        return run_command(
            "detect-secrets scan --baseline .secrets.baseline",
            "Создание baseline для detect-secrets"
        )
    else:
        print("✅ Baseline для detect-secrets уже существует")
        return True


def run_all_hooks():
    """Запускает все hooks на всех файлах."""
    return run_command(
        "pre-commit run --all-files",
        "Запуск всех hooks на всех файлах"
    )


def update_hooks():
    """Обновляет hooks до последних версий."""
    return run_command(
        "pre-commit autoupdate",
        "Обновление hooks до последних версий"
    )


def show_help():
    """Показывает справку по использованию."""
    print("""
🔧 Настройка Pre-commit Hooks для проекта кредитного скоринга

Использование:
    python scripts/setup_pre_commit.py [ОПЦИИ]

Опции:
    --install-only     Только установка hooks без запуска проверок
    --update          Обновить hooks до последних версий
    --run-all         Запустить все hooks на всех файлах
    --help            Показать эту справку

Примеры:
    python scripts/setup_pre_commit.py
    python scripts/setup_pre_commit.py --install-only
    python scripts/setup_pre_commit.py --run-all

Что делают hooks:
    • Black - форматирование Python кода
    • isort - сортировка импортов
    • flake8 - линтинг Python кода
    • mypy - проверка типов
    • bandit - проверка безопасности
    • yamllint - проверка YAML файлов
    • markdownlint - проверка Markdown файлов
    • detect-secrets - поиск секретов в коде
    • hadolint - проверка Dockerfile
    """)


def main():
    """Основная функция."""
    args = sys.argv[1:]
    
    if "--help" in args or "-h" in args:
        show_help()
        return 0
    
    print("🚀 Настройка Pre-commit Hooks для проекта кредитного скоринга")
    print("=" * 60)
    
    # Проверяем версию Python
    if not check_python_version():
        return 1
    
    # Устанавливаем pre-commit
    if not install_pre_commit():
        print("❌ Не удалось установить pre-commit")
        return 1
    
    # Обновляем hooks если нужно
    if "--update" in args:
        if not update_hooks():
            print("❌ Не удалось обновить hooks")
            return 1
    
    # Устанавливаем hooks
    if not install_hooks():
        print("❌ Не удалось установить pre-commit hooks")
        return 1
    
    # Устанавливаем hooks для CI
    if not install_hooks_ci():
        print("❌ Не удалось установить pre-commit hooks для CI")
        return 1
    
    # Создаем baseline для detect-secrets
    if not create_secrets_baseline():
        print("❌ Не удалось создать baseline для detect-secrets")
        return 1
    
    # Если только установка, не запускаем проверки
    if "--install-only" in args:
        print("\n✅ Pre-commit hooks установлены успешно!")
        print("💡 Теперь hooks будут автоматически запускаться при коммитах")
        return 0
    
    # Запускаем все hooks
    if "--run-all" in args or not args:
        print("\n🔍 Запуск проверок на всех файлах...")
        if not run_all_hooks():
            print("\n❌ Некоторые проверки не прошли")
            print("💡 Исправьте ошибки и попробуйте снова")
            return 1
        else:
            print("\n✅ Все проверки прошли успешно!")
    
    print("\n🎉 Настройка pre-commit hooks завершена!")
    print("\n📋 Полезные команды:")
    print("   pre-commit run --all-files    # Запустить все hooks")
    print("   pre-commit run <hook-name>    # Запустить конкретный hook")
    print("   pre-commit autoupdate         # Обновить hooks")
    print("   pre-commit uninstall          # Удалить hooks")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
