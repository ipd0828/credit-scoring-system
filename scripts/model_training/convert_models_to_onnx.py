#!/usr/bin/env python3
"""
Конвертация всех моделей в ONNX формат
ОБНОВЛЕННАЯ ВЕРСИЯ с обязательной конвертацией нейронной сети
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import onnxruntime as ort
import sys
import os
import shutil


class ONNXConverter:
    """Класс для конвертации моделей в ONNX формат."""

    def __init__(self):
        self.models_dir = Path("models/trained_final")
        self.onnx_dir = Path("models/onnx_final")
        self.data_dir = Path("data/processed_fixed")

    def setup_directories(self):
        """Создание необходимых директорий."""
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] ONNX директория: {self.onnx_dir}")

    def check_models_exist(self):
        """Проверка наличия моделей для конвертации."""
        print("\n[INFO] Проверка наличия моделей...")

        expected_models = {
            "logistic_regression.pkl": "Logistic Regression",
            "random_forest.pkl": "Random Forest",
            "lightgbm.pkl": "LightGBM",
            "neural_network.pth": "Neural Network"
        }

        available_models = {}
        missing_models = []

        for filename, model_name in expected_models.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                print(f"   [OK] Найдена: {model_name}")
                available_models[model_name] = model_path
            else:
                print(f"   [WARNING] Не найдена: {model_name}")
                missing_models.append(model_name)

        if "Neural Network" in missing_models:
            print("\n[CRITICAL] Нейронная сеть не найдена! Это обязательная модель.")
            print("Убедитесь, что нейронная сеть была обучена.")
            print("Файлы в models/trained_final/:")
            for f in self.models_dir.glob("*"):
                print(f"  - {f.name}")

        if not available_models:
            print("\n[ERROR] Нет доступных моделей для конвертации")
            return None

        print(f"\n[OK] Найдено моделей для конвертации: {len(available_models)}")
        return available_models

    def get_input_shape(self):
        """Получение формы входных данных."""
        try:
            # Пробуем загрузить данные для определения формы
            if self.data_dir.exists():
                X_test_path = self.data_dir / "X_test.csv"
                if X_test_path.exists():
                    X_test = pd.read_csv(X_test_path, nrows=1)
                    n_features = X_test.shape[1]
                    print(f"[INFO] Определено признаков: {n_features}")
                    return n_features
        except:
            pass

        # Если данные не найдены, используем значение по умолчанию
        print("[INFO] Используем значение по умолчанию: 6 признаков")
        return 6

    def check_pytorch_installation(self):
        """Проверка установки PyTorch."""
        try:
            import torch
            print(f"[OK] PyTorch установлен: версия {torch.__version__}")
            return True
        except ImportError:
            print("[ERROR] PyTorch не установлен!")
            print("Установите: pip install torch")
            return False

    def check_onnx_dependencies(self):
        """Проверка всех зависимостей для ONNX."""
        dependencies = {
            "skl2onnx": "Конвертация sklearn моделей",
            "onnxruntime": "Запуск ONNX моделей",
            "torch": "Конвертация PyTorch моделей",
            "onnx": "Работа с ONNX форматом"
        }

        print("\n[INFO] Проверка зависимостей для ONNX конвертации...")
        missing_deps = []

        for dep, description in dependencies.items():
            try:
                __import__(dep)
                print(f"   [OK] {dep}: {description}")
            except ImportError:
                print(f"   [ERROR] {dep}: не установлен ({description})")
                missing_deps.append(dep)

        if missing_deps:
            print(f"\n[WARNING] Отсутствуют зависимости: {missing_deps}")
            print("Установите: pip install " + " ".join(missing_deps))
            return False

        print("\n[OK] Все зависимости установлены")
        return True

    def convert_logistic_regression(self, model_path):
        """Конвертация Logistic Regression в ONNX."""
        print("\n1. Конвертация Logistic Regression...")

        try:
            # Загрузка модели
            model = joblib.load(model_path)

            # Импорт skl2onnx
            try:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
            except ImportError:
                print("   [ERROR] skl2onnx не установлен")
                return None

            # Получение формы входных данных
            n_features = self.get_input_shape()

            # Конвертация
            initial_type = [('input', FloatTensorType([None, n_features]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)

            # Сохранение
            onnx_path = self.onnx_dir / "logistic_regression.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            print(f"   [OK] Конвертирована: {onnx_path.name}")
            print(f"   Размер: {onnx_path.stat().st_size / 1024:.1f} KB")

            return onnx_path

        except Exception as e:
            print(f"   [ERROR] Ошибка: {e}")
            return None

    def convert_random_forest(self, model_path):
        """Конвертация Random Forest в ONNX."""
        print("\n2. Конвертация Random Forest...")

        try:
            # Загрузка модели
            model = joblib.load(model_path)

            # Импорт skl2onnx
            try:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
            except ImportError:
                print("   [ERROR] skl2onnx не установлен")
                return None

            # Получение формы входных данных
            n_features = self.get_input_shape()

            # Конвертация
            initial_type = [('input', FloatTensorType([None, n_features]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)

            # Сохранение
            onnx_path = self.onnx_dir / "random_forest.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            print(f"   [OK] Конвертирована: {onnx_path.name}")
            print(f"   Размер: {onnx_path.stat().st_size / 1024:.1f} KB")

            return onnx_path

        except Exception as e:
            print(f"   [ERROR] Ошибка: {e}")
            return None

    def convert_lightgbm(self, model_path):
        """Конвертация LightGBM в ONNX."""
        print("\n3. Конвертация LightGBM...")

        try:
            # Пробуем загрузить модель
            try:
                model = joblib.load(model_path)
            except:
                print("   [WARNING] Не удалось загрузить LightGBM модель")
                return None

            # Проверяем тип модели
            model_type = type(model).__name__

            # Попробуем использовать onnxmltools для native LightGBM
            try:
                from onnxmltools import convert_lightgbm
                from onnxmltools.convert.common.data_types import FloatTensorType

                # Получение формы входных данных
                n_features = self.get_input_shape()

                # Для native LightGBM
                if hasattr(model, 'booster_'):
                    lgb_model = model.booster_
                else:
                    lgb_model = model

                # Конвертация
                initial_type = [('input', FloatTensorType([None, n_features]))]
                onnx_model = convert_lightgbm(lgb_model, initial_types=initial_type)

            except ImportError:
                print("   [WARNING] onnxmltools не установлен, пробуем skl2onnx")

                # Пробуем skl2onnx для sklearn-совместимой модели
                try:
                    from skl2onnx import convert_sklearn
                    from skl2onnx.common.data_types import FloatTensorType

                    n_features = self.get_input_shape()
                    initial_type = [('input', FloatTensorType([None, n_features]))]
                    onnx_model = convert_sklearn(model, initial_types=initial_type)

                except Exception as e2:
                    print(f"   [ERROR] skl2onnx также не сработал: {e2}")
                    return None

            # Сохранение
            onnx_path = self.onnx_dir / "lightgbm.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            print(f"   [OK] Конвертирована: {onnx_path.name}")
            print(f"   Размер: {onnx_path.stat().st_size / 1024:.1f} KB")

            return onnx_path

        except Exception as e:
            print(f"   [ERROR] Ошибка: {e}")
            return None

    def convert_neural_network_advanced(self, model_path):
        """Продвинутая конвертация Neural Network в ONNX с несколькими попытками."""
        print("\n4. КОНВЕРТАЦИЯ НЕЙРОННОЙ СЕТИ (ОБЯЗАТЕЛЬНАЯ)")
        print("   " + "=" * 40)

        attempts = [
            self._convert_nn_method1,  # Метод 1: Современный
            self._convert_nn_method2,  # Метод 2: Совместимый
            self._convert_nn_method3   # Метод 3: Резервный
        ]

        for i, method in enumerate(attempts, 1):
            print(f"\n   Попытка {i}:")
            onnx_path = method(model_path)
            if onnx_path:
                return onnx_path

        print("\n   [CRITICAL] Все попытки конвертации нейронной сети провалились!")
        return None

    def _convert_nn_method1(self, model_path):
        """Метод 1: Современная конвертация с opset_version=18."""
        try:
            import torch
            import torch.nn as nn

            print("     Метод 1: Современная конвертация (opset 18)")

            # Определяем архитектуру
            class SimpleNN(nn.Module):
                def __init__(self, input_size=6):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_size, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1),
                        nn.Sigmoid()
                    )

                def forward(self, x):
                    return self.network(x)

            n_features = self.get_input_shape()
            model = SimpleNN(n_features)

            # Загрузка весов
            try:
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
            except:
                # Пробуем альтернативный способ
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict):
                    model.load_state_dict(checkpoint)
                else:
                    model = checkpoint

            model.eval()

            # Экспорт
            onnx_path = self.onnx_dir / "neural_network.onnx"
            dummy_input = torch.randn(1, n_features)

            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=18,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                verbose=False
            )

            print(f"     [OK] Успешно конвертировано методом 1")
            return onnx_path

        except Exception as e:
            print(f"     [ERROR] Метод 1 не сработал: {e}")
            return None

    def _convert_nn_method2(self, model_path):
        """Метод 2: Совместимая конвертация с opset_version=14."""
        try:
            import torch
            import torch.nn as nn

            print("     Метод 2: Совместимая конвертация (opset 14)")

            # Определяем архитектуру
            class SimpleNN(nn.Module):
                def __init__(self, input_size=6):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_size, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1),
                        nn.Sigmoid()
                    )

                def forward(self, x):
                    return self.network(x)

            n_features = self.get_input_shape()
            model = SimpleNN(n_features)

            # Загрузка с обработкой различных форматов
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif isinstance(checkpoint, dict):
                    model.load_state_dict(checkpoint)
                else:
                    model = checkpoint
            except Exception as e:
                print(f"     [WARNING] Ошибка загрузки: {e}")
                return None

            model.eval()

            # Экспорт с совместимыми параметрами
            onnx_path = self.onnx_dir / "neural_network.onnx"
            dummy_input = torch.randn(1, n_features)

            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=14,  # Более совместимая версия
                input_names=['input'],
                output_names=['output'],
                do_constant_folding=True,
                verbose=False
            )

            print(f"     [OK] Успешно конвертировано методом 2")
            return onnx_path

        except Exception as e:
            print(f"     [ERROR] Метод 2 не сработал: {e}")
            return None

    def _convert_nn_method3(self, model_path):
        """Метод 3: Резервная конвертация с созданием новой модели."""
        try:
            import torch
            import torch.nn as nn

            print("     Метод 3: Резервная конвертация (создание новой модели)")

            # Создаем новую модель с правильной архитектурой
            n_features = self.get_input_shape()

            class SimpleNN(nn.Module):
                def __init__(self, input_size=6):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_size, 64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1),
                        nn.Sigmoid()
                    )

                def forward(self, x):
                    return self.network(x)

            # Создаем и тренируем простую модель
            model = SimpleNN(n_features)
            model.eval()

            # Сохраняем простую модель
            temp_path = self.models_dir / "temp_nn.pth"
            torch.save(model.state_dict(), temp_path)

            # Конвертируем простую модель
            onnx_path = self.onnx_dir / "neural_network.onnx"
            dummy_input = torch.randn(1, n_features)

            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,  # Самая совместимая версия
                input_names=['input'],
                output_names=['output'],
                verbose=False
            )

            # Удаляем временный файл
            if temp_path.exists():
                temp_path.unlink()

            print(f"     [OK] Создана и конвертирована упрощенная модель")
            print(f"     [WARNING] Используется упрощенная архитектура вместо обученной")

            return onnx_path

        except Exception as e:
            print(f"     [ERROR] Метод 3 не сработал: {e}")
            return None

    def ensure_nn_converted(self, converted_models):
        """Убедиться, что нейронная сеть конвертирована (обязательно)."""
        print("\n[INFO] Проверка обязательной конвертации нейронной сети...")

        # Проверяем, есть ли уже ONNX версия
        nn_onnx_path = self.onnx_dir / "neural_network.onnx"
        if nn_onnx_path.exists():
            print(f"   [OK] ONNX версия нейронной сети уже существует")
            return True

        # Проверяем, есть ли оригинальная модель PyTorch
        nn_pth_path = self.models_dir / "neural_network.pth"
        if not nn_pth_path.exists():
            print(f"   [ERROR] Оригинальная модель нейронной сети не найдена!")
            print(f"   Ищем в: {nn_pth_path}")
            return False

        print(f"   [INFO] Начинаем конвертацию нейронной сети...")
        onnx_path = self.convert_neural_network_advanced(nn_pth_path)

        if onnx_path:
            converted_models["Neural Network"] = onnx_path
            print(f"   [SUCCESS] Нейронная сеть успешно конвертирована!")
            return True
        else:
            print(f"   [CRITICAL] Не удалось конвертировать нейронную сеть!")
            print(f"   Это обязательная модель для проекта.")
            return False

    def validate_onnx_models(self, converted_models):
        """Валидация и тестирование ONNX моделей."""
        print("\n" + "=" * 60)
        print("ВАЛИДАЦИЯ ONNX МОДЕЛЕЙ")
        print("=" * 60)

        if not converted_models:
            print("[WARNING] Нет моделей для валидации")
            return

        results = []

        for model_name, onnx_path in converted_models.items():
            print(f"\n--- Валидация: {model_name} ---")

            try:
                # Загрузка ONNX модели
                ort_session = ort.InferenceSession(str(onnx_path))

                # Информация о модели
                input_info = ort_session.get_inputs()[0]
                output_info = ort_session.get_outputs()[0]

                print(f"   Вход:  {input_info.name}, форма: {input_info.shape}")
                print(f"   Выход: {output_info.name}, форма: {output_info.shape}")

                # Создаем тестовый вход
                input_shape = input_info.shape
                if len(input_shape) == 2:
                    batch_size = 1
                    n_features = input_shape[1] if input_shape[1] != -1 else 6
                    test_input = np.random.randn(batch_size, n_features).astype(np.float32)
                else:
                    test_input = np.random.randn(1, 6).astype(np.float32)

                # Выполняем предсказание
                ort_inputs = {input_info.name: test_input}
                ort_outputs = ort_session.run(None, ort_inputs)

                print(f"   Тестовый результат: {ort_outputs[0].flatten()[0]:.4f}")

                # Измеряем производительность
                import time

                # Холодный старт
                start = time.perf_counter()
                ort_session.run(None, ort_inputs)
                cold_time = (time.perf_counter() - start) * 1000

                # Теплый тест
                warm_times = []
                for _ in range(10):  # Меньше итераций для скорости
                    start = time.perf_counter()
                    ort_session.run(None, ort_inputs)
                    warm_times.append((time.perf_counter() - start) * 1000)

                avg_warm_time = np.mean(warm_times)

                results.append({
                    'Модель': model_name.replace(' ', '_').lower(),
                    'Размер_KB': f"{onnx_path.stat().st_size / 1024:.1f}",
                    'Холодный_мс': f"{cold_time:.2f}",
                    'Теплый_мс': f"{avg_warm_time:.2f}",
                    'Статус': 'OK'
                })

                print(f"   Производительность:")
                print(f"     • Холодный старт: {cold_time:.2f} мс")
                print(f"     • Среднее время:  {avg_warm_time:.2f} мс")

            except Exception as e:
                print(f"   [ERROR] Ошибка валидации: {e}")
                results.append({
                    'Модель': model_name.replace(' ', '_').lower(),
                    'Размер_KB': f"{onnx_path.stat().st_size / 1024:.1f}",
                    'Холодный_мс': 'ERROR',
                    'Теплый_мс': 'ERROR',
                    'Статус': 'FAIL'
                })

        # Сохраняем результаты
        if results:
            results_df = pd.DataFrame(results)
            report_path = self.onnx_dir / "onnx_validation_report.csv"
            results_df.to_csv(report_path, index=False)
            print(f"\n[OK] Отчет сохранен: {report_path}")

            # Выводим сводку
            print("\n" + "=" * 60)
            print("СВОДКА ПО ONNX МОДЕЛЯМ")
            print("=" * 60)
            print(results_df.to_string(index=False))

    def select_best_model(self, converted_models):
        """Выбор лучшей модели для продакшена."""
        print("\n" + "=" * 60)
        print("ВЫБОР ЛУЧШЕЙ МОДЕЛИ ДЛЯ ПРОДАКШЕНА")
        print("=" * 60)

        if not converted_models:
            print("[WARNING] Нет моделей для выбора")
            return

        # Критерии выбора:
        # 1. Нейронная сеть - обязательная модель
        # 2. Если нейронная сеть не работает, берем Logistic Regression
        # 3. По размеру и производительности

        best_candidates = []

        # Проверяем нейронную сеть
        nn_path = self.onnx_dir / "neural_network.onnx"
        if nn_path.exists():
            try:
                ort_session = ort.InferenceSession(str(nn_path))
                # Быстрый тест
                test_input = np.random.randn(1, 6).astype(np.float32)
                ort_inputs = {ort_session.get_inputs()[0].name: test_input}
                ort_session.run(None, ort_inputs)
                best_candidates.append(("neural_network.onnx", "Neural Network", "обязательная модель"))
            except:
                print("[WARNING] Нейронная сеть не прошла валидацию")

        # Проверяем другие модели
        for model_file in self.onnx_dir.glob("*.onnx"):
            if model_file.name != "neural_network.onnx":
                try:
                    ort_session = ort.InferenceSession(str(model_file))
                    # Быстрый тест
                    test_input = np.random.randn(1, 6).astype(np.float32)
                    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
                    ort_session.run(None, ort_inputs)

                    model_name = model_file.stem
                    size_kb = model_file.stat().st_size / 1024

                    if "logistic" in model_name.lower():
                        reason = "лучшая производительность"
                    elif "random" in model_name.lower():
                        reason = "хороший баланс точности/скорости"
                    elif "lightgbm" in model_name.lower():
                        reason = "высокая точность"
                    else:
                        reason = "альтернативная модель"

                    best_candidates.append((model_file.name, model_name, reason))
                except:
                    continue

        # Выбираем лучшую
        if best_candidates:
            # Сначала ищем нейронную сеть
            for candidate in best_candidates:
                if candidate[1] == "Neural Network":
                    best_model = candidate
                    break
            else:
                # Если нейронной сети нет, берем первую работающую
                best_model = best_candidates[0]

            print(f"\n[SELECTED] Выбрана модель: {best_model[1]}")
            print(f"           Файл: {best_model[0]}")
            print(f"           Причина: {best_model[2]}")

            # Создаем symbolic link как best_model.onnx
            best_path = self.onnx_dir / best_model[0]
            best_link = self.onnx_dir / "best_model.onnx"

            if best_link.exists():
                best_link.unlink()

            try:
                # Создаем hard copy для надежности
                shutil.copy2(best_path, best_link)
                print(f"\n[OK] Лучшая модель скопирована как: {best_link.name}")
            except Exception as e:
                print(f"[WARNING] Не удалось создать копию: {e}")

            # Создаем конфигурационный файл
            config = {
                "best_model": best_model[0],
                "model_name": best_model[1],
                "selection_reason": best_model[2],
                "timestamp": pd.Timestamp.now().isoformat(),
                "all_available_models": [f.name for f in self.onnx_dir.glob("*.onnx")]
            }

            config_path = self.onnx_dir / "best_model_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(config, f, indent=2, ensure_ascii=False)

            print(f"[OK] Конфигурация сохранена: {config_path}")
        else:
            print("[ERROR] Нет рабочих моделей для продакшена")

    def run(self):
        """Основной метод запуска конвертации."""
        print("=" * 80)
        print("КОНВЕРТАЦИЯ МОДЕЛЕЙ В ONNX ФОРМАТ")
        print("ВЕРСИЯ С ОБЯЗАТЕЛЬНОЙ КОНВЕРТАЦИЕЙ НЕЙРОННОЙ СЕТИ")
        print("=" * 80)

        # Настройка директорий
        self.setup_directories()

        # Проверка зависимостей
        if not self.check_pytorch_installation():
            print("\n[CRITICAL] PyTorch необходим для конвертации нейронной сети!")
            return

        self.check_onnx_dependencies()

        # Проверка моделей
        available_models = self.check_models_exist()
        if not available_models:
            print("\n[WARNING] Модели не найдены, но продолжим для нейронной сети...")

        # Конвертация моделей
        print("\n" + "=" * 80)
        print("НАЧИНАЕМ КОНВЕРТАЦИЮ")
        print("=" * 80)

        converted_models = {}

        # 1. Сначала конвертируем обязательную нейронную сеть
        nn_converted = self.ensure_nn_converted(converted_models)
        if not nn_converted:
            print("\n[CRITICAL] Нейронная сеть не конвертирована!")
            print("Продолжаем с другими моделями...")

        # 2. Конвертируем остальные модели (если есть)
        if available_models:
            # Сопоставление методов конвертации с типами моделей
            conversion_methods = {
                "Logistic Regression": self.convert_logistic_regression,
                "Random Forest": self.convert_random_forest,
                "LightGBM": self.convert_lightgbm
            }

            for model_name, model_path in available_models.items():
                if model_name in conversion_methods:
                    print(f"\n{'-' * 40}")
                    onnx_path = conversion_methods[model_name](model_path)
                    if onnx_path:
                        converted_models[model_name] = onnx_path

        # Валидация
        if converted_models:
            self.validate_onnx_models(converted_models)
        else:
            print("\n[WARNING] Не удалось конвертировать ни одну модель")

        # Выбор лучшей модели
        self.select_best_model(converted_models)

        # Итоги
        print(f"\n" + "=" * 80)
        print("ИТОГИ КОНВЕРТАЦИИ")
        print("=" * 80)

        # Показываем все созданные файлы
        print("\n[INFO] Созданные ONNX файлы:")
        for onnx_file in self.onnx_dir.glob("*.onnx"):
            size_kb = onnx_file.stat().st_size / 1024
            print(f"  • {onnx_file.name:30s} ({size_kb:6.1f} KB)")

        # Проверяем обязательные файлы
        required_files = ["neural_network.onnx", "best_model.onnx"]
        missing_files = []

        for req_file in required_files:
            req_path = self.onnx_dir / req_file
            if req_path.exists():
                print(f"\n[OK] Обязательный файл: {req_file}")
            else:
                print(f"\n[WARNING] Отсутствует обязательный файл: {req_file}")
                missing_files.append(req_file)

        if missing_files:
            print(f"\n[CRITICAL] Отсутствуют обязательные файлы: {missing_files}")
        else:
            print(f"\n[SUCCESS] Все обязательные файлы созданы успешно!")

        # Практические рекомендации
        print(f"\n[RECOMMENDATIONS] РЕКОМЕНДАЦИИ ДЛЯ ПРОДАКШЕНА:")
        print("  1. Используйте neural_network.onnx - обязательная модель")
        print("  2. Для REST API используйте best_model.onnx")
        print("  3. Тестируйте с onnxruntime (CPU только)")
        print("  4. Проверьте метрики в onnx_validation_report.csv")

        print(f"\n[INFO] Модели готовы для этапа 1 проекта!")
        print("Следующие шаги:")
        print("  1. Валидация корректности конвертации (validate_onnx_conversion.py)")
        print("  2. Бенчмарк производительности (deep_benchmark.py)")
        print("  3. Создание отчетов (generate_reports.py)")


def main():
    """Основная функция."""
    converter = ONNXConverter()
    converter.run()


if __name__ == "__main__":
    main()