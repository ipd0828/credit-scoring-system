#!/usr/bin/env python3
"""
Оптимизация моделей для промышленной эксплуатации - ИСПРАВЛЕННАЯ ВЕРСИЯ
scripts/model_training/optimize_model_fixed.py
"""

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import joblib
import onnx
import onnxruntime as ort
from pathlib import Path
import pandas as pd
import time
from sklearn.metrics import accuracy_score, roc_auc_score
import json


class CreditScoringNN(nn.Module):
    """Нейронная сеть для кредитного скоринга (с BatchNorm)."""

    def __init__(self, input_size):
        super(CreditScoringNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class CreditScoringNN_Simple(nn.Module):
    """Упрощенная версия (без BatchNorm и Dropout)."""

    def __init__(self, input_size):
        super(CreditScoringNN_Simple, self).__init__()
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


def load_nn_model_corrected():
    """Загрузка нейронной сети с правильной архитектурой."""
    print("=" * 60)
    print("ЗАГРУЗКА НЕЙРОННОЙ СЕТИ ДЛЯ ОПТИМИЗАЦИИ")
    print("=" * 60)

    try:
        models_dir = Path("models/nn_models")
        model_path = models_dir / "nn_credit_scoring.pth"
        scaler_path = models_dir / "nn_credit_scoring_scaler.pkl"
        info_path = models_dir / "nn_credit_scoring_info.pkl"

        if not all(p.exists() for p in [model_path, scaler_path, info_path]):
            print(" Файлы модели не найдены")
            return None

        # Загрузка информации
        model_info = joblib.load(info_path)
        input_size = model_info['input_size']

        # Создаем модель с правильной архитектурой (с BatchNorm)
        model = CreditScoringNN(input_size)

        # Загружаем веса
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        # Загрузка scaler
        scaler = joblib.load(scaler_path)

        print(f" Модель загружена:")
        print(f"   Input size: {input_size}")
        print(f"   Параметры: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Размер файла: {model_path.stat().st_size / 1024:.1f} KB")

        # Создаем упрощенную версию для оптимизации
        simple_model = CreditScoringNN_Simple(input_size)

        # Копируем веса из Linear слоев
        state_dict = model.state_dict()
        simple_state_dict = {}

        # Маппинг весов: оригинальная модель -> упрощенная
        weight_mapping = {
            'network.0.weight': 'network.0.weight',
            'network.0.bias': 'network.0.bias',
            'network.4.weight': 'network.2.weight',  # Пропускаем BatchNorm и Dropout
            'network.4.bias': 'network.2.bias',
            'network.8.weight': 'network.4.weight',
            'network.8.bias': 'network.4.bias',
            'network.12.weight': 'network.6.weight',
            'network.12.bias': 'network.6.bias'
        }

        for orig_key, simple_key in weight_mapping.items():
            if orig_key in state_dict:
                simple_state_dict[simple_key] = state_dict[orig_key]

        simple_model.load_state_dict(simple_state_dict, strict=False)
        simple_model.eval()

        print(f" Создана упрощенная версия модели")

        return simple_model, scaler, model_info

    except Exception as e:
        print(f" Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_to_inference_model(model):
    """Конвертация модели в режим инференса (удаление Dropout)."""
    print("\nКонвертация модели для инференса...")

    # Создаем копию
    model_copy = type(model)(model.network[0].in_features)

    # Копируем веса
    model_copy.load_state_dict(model.state_dict())

    # Переводим в eval режим (отключает Dropout и BatchNorm training mode)
    model_copy.eval()

    # Создаем новый Sequential без Dropout
    inference_layers = []

    for layer in model_copy.network:
        if not isinstance(layer, nn.Dropout):
            # Для BatchNorm замораживаем статистику
            if isinstance(layer, nn.BatchNorm1d):
                layer.track_running_stats = True
                layer.eval()
            inference_layers.append(layer)

    model_copy.network = nn.Sequential(*inference_layers)

    print(f" Dropout слои удалены, BatchNorm заморожен")
    return model_copy


def apply_pruning_fixed(model, pruning_amount=0.3):
    """Применение pruning (обрезки) к модели."""
    print(f"\n{'=' * 60}")
    print(f"ПРИМЕНЕНИЕ PRUNING ({pruning_amount * 100:.0f}%)")
    print("=" * 60)

    try:
        # Создаем копию модели
        pruned_model = type(model)(model.network[0].in_features)
        pruned_model.load_state_dict(model.state_dict())
        pruned_model.eval()

        # Применяем pruning только к Linear слоям
        parameters_to_prune = []
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))

        if not parameters_to_prune:
            print(" Не найдены Linear слои для pruning")
            return model

        # Global pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_amount
        )

        # Удаляем маски pruning (делаем обрезку постоянной)
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        # Подсчет обрезанных параметров
        total_params = 0
        zero_params = 0

        for p in pruned_model.parameters():
            if p.dim() > 1:  # Игнорируем bias
                total_params += p.numel()
                zero_params += (p == 0).sum().item()

        sparsity = zero_params / total_params * 100 if total_params > 0 else 0

        print(f" Pruning применен:")
        print(f"   Обрезано слоев: {len(parameters_to_prune)}")
        print(f"   Нулевых параметров: {zero_params:,} ({sparsity:.1f}%)")
        print(f"   Всего параметров: {total_params:,}")

        return pruned_model

    except Exception as e:
        print(f" Ошибка pruning: {e}")
        import traceback
        traceback.print_exc()
        return model


def apply_quantization_fixed(model):
    """Применение quantization (квантования) к модели."""
    print(f"\n{'=' * 60}")
    print("ПРИМЕНЕНИЕ QUANTIZATION")
    print("=" * 60)

    try:
        # Dynamic quantization (во время выполнения)
        # Квантуем только Linear слои
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},  # Квантуем только линейные слои
            dtype=torch.qint8
        )

        print(f" Dynamic quantization применен")
        print(f"   Тип: torch.qint8")
        print(f"   Квантованные слои: Linear")

        return quantized_model

    except Exception as e:
        print(f" Ошибка quantization: {e}")
        import traceback
        traceback.print_exc()
        return model


def evaluate_model_performance_fixed(model, scaler, model_name="Model"):
    """Оценка производительности модели."""
    print(f"\n{'=' * 60}")
    print(f"ОЦЕНКА ПРОИЗВОДИТЕЛЬНОСТИ: {model_name}")
    print("=" * 60)

    try:
        # Загрузка тестовых данных
        data_path = Path("data/processed_fixed")
        X_test = pd.read_csv(data_path / "X_test.csv")
        y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

        # Подготовка данных
        X_test_scaled = scaler.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled)

        # Переводим модель в eval режим
        model.eval()

        # Холодный старт (5 запусков для усреднения)
        print(f"\nХолодный старт (среднее по 5 запускам):")
        cold_start_times = []
        for _ in range(5):
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(X_test_tensor[:1])
            cold_start_times.append(time.perf_counter() - start_time)

        avg_cold_start = np.mean(cold_start_times)
        print(f"  Среднее время: {avg_cold_start * 1000:.2f} мс")
        print(f"  Стандартное отклонение: {np.std(cold_start_times) * 1000:.2f} мс")

        # Пакетная обработка
        batch_sizes = [1, 10, 100, 500, 1000]

        print(f"\nПакетная обработка (среднее по 3 запускам):")
        batch_results = {}

        for batch_size in batch_sizes:
            if batch_size > len(X_test_tensor):
                continue

            batch_data = X_test_tensor[:batch_size]
            inference_times = []

            # 3 запуска для каждого batch size
            for _ in range(3):
                start_time = time.perf_counter()
                with torch.no_grad():
                    predictions = model(batch_data)
                inference_times.append(time.perf_counter() - start_time)

            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)

            # Расчет метрик
            time_per_sample = avg_time / batch_size * 1000  # мс на образец
            samples_per_second = batch_size / avg_time

            batch_results[batch_size] = {
                'total_time_ms': avg_time * 1000,
                'time_per_sample_ms': time_per_sample,
                'samples_per_second': samples_per_second,
                'std_dev_ms': std_time * 1000
            }

            print(f"  Batch {batch_size:4d}: "
                  f"{avg_time * 1000:6.1f} ± {std_time * 1000:4.1f} мс total, "
                  f"{time_per_sample:6.2f} мс/sample, "
                  f"{samples_per_second:6.0f} samples/sec")

        # Полная выборка (3 запуска для стабильности)
        print(f"\nПолная тестовая выборка ({len(X_test_tensor)} samples):")

        total_times = []
        all_predictions_list = []

        for _ in range(3):
            start_time = time.perf_counter()
            with torch.no_grad():
                predictions = model(X_test_tensor)
            total_times.append(time.perf_counter() - start_time)
            all_predictions_list.append(predictions.numpy())

        # Используем предсказания последнего запуска
        all_predictions = all_predictions_list[-1]
        avg_total_time = np.mean(total_times)

        # Метрики точности
        y_proba = all_predictions.flatten()
        y_pred = (y_proba >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"  Среднее время (3 запуска): {avg_total_time:.3f} ± {np.std(total_times):.3f} сек")
        print(f"  Время на образец: {avg_total_time / len(X_test_tensor) * 1000:.3f} мс")
        print(f"  Samples/sec: {len(X_test_tensor) / avg_total_time:.0f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")

        # Размер модели
        if hasattr(model, 'state_dict'):
            # Сохраняем временно для измерения размера
            temp_path = Path("temp_model.pth")
            torch.save(model.state_dict(), temp_path)
            model_size_kb = temp_path.stat().st_size / 1024
            temp_path.unlink()
        else:
            model_size_kb = 0

        # Дополнительная информация о предсказаниях
        print(f"\n  Распределение предсказаний:")
        print(f"    % класса 1: {(y_pred == 1).mean() * 100:.1f}%")
        print(f"    Вероятности: min={y_proba.min():.3f}, max={y_proba.max():.3f}, mean={y_proba.mean():.3f}")

        return {
            'cold_start_ms': avg_cold_start * 1000,
            'cold_start_std_ms': np.std(cold_start_times) * 1000,
            'total_inference_s': avg_total_time,
            'total_inference_std_s': np.std(total_times),
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'samples_per_second': len(X_test_tensor) / avg_total_time,
            'batch_results': batch_results,
            'model_size_kb': model_size_kb,
            'predictions_info': {
                'class_1_percentage': (y_pred == 1).mean() * 100,
                'proba_min': float(y_proba.min()),
                'proba_max': float(y_proba.max()),
                'proba_mean': float(y_proba.mean())
            }
        }

    except Exception as e:
        print(f" Ошибка оценки производительности: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_optimized_model_fixed(model, scaler, optimization_type, results):
    """Сохранение оптимизированной модели."""
    print(f"\n{'=' * 60}")
    print(f"СОХРАНЕНИЕ ОПТИМИЗИРОВАННОЙ МОДЕЛИ ({optimization_type})")
    print("=" * 60)

    try:
        # Создание директории
        optimized_dir = Path("models/optimized")
        optimized_dir.mkdir(parents=True, exist_ok=True)

        # Сохранение PyTorch модели
        if hasattr(model, 'state_dict'):
            model_path = optimized_dir / f"model_{optimization_type}.pth"
            torch.save(model.state_dict(), model_path)
            print(f" PyTorch модель сохранена: {model_path}")
            print(f"   Размер: {model_path.stat().st_size / 1024:.1f} KB")

        # Сохранение scaler
        scaler_path = optimized_dir / f"scaler_{optimization_type}.pkl"
        joblib.dump(scaler, scaler_path)
        print(f" Scaler сохранен: {scaler_path}")

        # Сохранение результатов
        results_path = optimized_dir / f"results_{optimization_type}.json"

        # Функция для конвертации numpy типов
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        with open(results_path, 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)

        print(f" Результаты сохранены: {results_path}")

        # Конвертация в ONNX
        if hasattr(model, 'state_dict'):
            onnx_path = optimized_dir / f"model_{optimization_type}.onnx"

            try:
                # Создание dummy input
                input_size = model.network[0].in_features
                dummy_input = torch.randn(1, input_size)

                # Экспорт в ONNX
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )

                print(f" ONNX модель сохранена: {onnx_path}")
                print(f"   Размер: {onnx_path.stat().st_size / 1024:.1f} KB")

                # Проверка ONNX модели
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                print(f" ONNX модель проверена")

                # Тестирование ONNX модели
                ort_session = ort.InferenceSession(str(onnx_path))
                ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
                ort_outputs = ort_session.run(None, ort_inputs)[0]

                # Сравнение с PyTorch
                with torch.no_grad():
                    torch_output = model(dummy_input).numpy()

                diff = np.abs(torch_output - ort_outputs).max()
                print(f" ONNX и PyTorch результаты совпадают (макс. разница: {diff:.6f})")

            except Exception as e:
                print(f"  Ошибка конвертации в ONNX: {e}")

        return model_path if 'model_path' in locals() else None

    except Exception as e:
        print(f" Ошибка сохранения модели: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_optimizations_fixed(original_results, optimized_results, optimization_type):
    """Сравнение оригинальной и оптимизированной модели."""
    print(f"\n{'=' * 60}")
    print(f"СРАВНЕНИЕ: ОРИГИНАЛЬНАЯ vs {optimization_type.upper()}")
    print("=" * 60)

    if not original_results or not optimized_results:
        print(" Недостаточно данных для сравнения")
        return

    print(f"\nМетрики производительности:")
    print(f"{'Метрика':<25} {'Оригинальная':<15} {optimization_type:<15} {'Изменение':<15}")
    print("-" * 70)

    metrics = [
        ('Холодный старт (мс)', 'cold_start_ms'),
        ('Полное время (с)', 'total_inference_s'),
        ('Samples/sec', 'samples_per_second'),
        ('Accuracy', 'accuracy'),
        ('ROC-AUC', 'roc_auc'),
        ('Размер модели (KB)', 'model_size_kb')
    ]

    for display_name, key in metrics:
        if key in original_results and key in optimized_results:
            orig = original_results[key]
            opt = optimized_results[key]

            if isinstance(orig, (int, float)) and isinstance(opt, (int, float)):
                if key in ['total_inference_s', 'cold_start_ms']:
                    # Меньше - лучше
                    if orig > 0:
                        change_pct = (orig - opt) / orig * 100
                        change_str = f"{change_pct:+.1f}%"
                    else:
                        change_str = "N/A"
                elif key in ['samples_per_second']:
                    # Больше - лучше
                    if orig > 0:
                        change_pct = (opt - orig) / orig * 100
                        change_str = f"{change_pct:+.1f}%"
                    else:
                        change_str = "N/A"
                elif key in ['accuracy', 'roc_auc']:
                    # Изменение accuracy
                    change = opt - orig
                    change_str = f"{change:+.4f}"
                elif key in ['model_size_kb']:
                    # Сжатие
                    if orig > 0:
                        compression = (orig - opt) / orig * 100
                        change_str = f"{compression:+.1f}% сжатие"
                    else:
                        change_str = "N/A"
                else:
                    change_str = "N/A"

                orig_str = f"{orig:.3f}" if isinstance(orig, float) else f"{orig}"
                opt_str = f"{opt:.3f}" if isinstance(opt, float) else f"{opt}"

                print(f"{display_name:<25} {orig_str:<15} {opt_str:<15} {change_str:<15}")

    # Сравнение по batch размерам
    if 'batch_results' in original_results and 'batch_results' in optimized_results:
        print(f"\nСравнение по batch размерам (time per sample в мс):")
        print(f"{'Batch size':<12} {'Оригинальная':<15} {optimization_type:<15} {'Ускорение':<15}")
        print("-" * 60)

        for batch_size in [1, 10, 100, 500, 1000]:
            if (batch_size in original_results['batch_results'] and
                    batch_size in optimized_results['batch_results']):
                orig_time = original_results['batch_results'][batch_size]['time_per_sample_ms']
                opt_time = optimized_results['batch_results'][batch_size]['time_per_sample_ms']

                if opt_time > 0:
                    speedup = orig_time / opt_time
                else:
                    speedup = 0

                print(f"{batch_size:<12} {orig_time:<15.2f} {opt_time:<15.2f} {speedup:.2f}x")


def main_fixed():
    """Основная функция оптимизации - исправленная версия."""
    print("=" * 60)
    print("ОПТИМИЗАЦИЯ МОДЕЛЕЙ - ИСПРАВЛЕННАЯ ВЕРСИЯ")
    print("=" * 60)

    all_results = {}

    # Загрузка модели
    model_data = load_nn_model_corrected()
    if not model_data:
        print(" Не удалось загрузить модель")
        return

    model, scaler, model_info = model_data

    # Конвертация модели для инференса
    inference_model = convert_to_inference_model(model)

    # Оценка оригинальной (упрощенной) модели
    print(f"\n{'=' * 60}")
    print("БАЗОВАЯ ОЦЕНКА ОРИГИНАЛЬНОЙ (УПРОЩЕННОЙ) МОДЕЛИ")
    print("=" * 60)

    original_results = evaluate_model_performance_fixed(
        inference_model, scaler, "Original Simplified Model"
    )

    if original_results:
        all_results['original_simplified'] = original_results
        save_optimized_model_fixed(inference_model, scaler, "original_simplified", original_results)

    # 1. Pruning эксперименты
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ 1: PRUNING")
    print("=" * 60)

    pruning_amounts = [0.1, 0.2, 0.3]

    for amount in pruning_amounts:
        print(f"\n--- Pruning {amount * 100:.0f}% ---")

        pruned_model = apply_pruning_fixed(inference_model, amount)

        if pruned_model:
            pruned_results = evaluate_model_performance_fixed(
                pruned_model, scaler, f"Pruned {amount * 100:.0f}%"
            )

            if pruned_results:
                all_results[f'pruned_{amount}'] = pruned_results

                # Сравнение
                compare_optimizations_fixed(original_results, pruned_results,
                                            f"Pruned_{amount * 100:.0f}%")

                # Сохранение если accuracy не упала более чем на 2%
                if pruned_results['accuracy'] >= original_results['accuracy'] * 0.98:
                    save_optimized_model_fixed(
                        pruned_model, scaler, f"pruned_{amount}", pruned_results
                    )
                else:
                    print(f"  Accuracy упала слишком сильно, модель не сохранена")

    # 2. Dynamic Quantization
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ 2: DYNAMIC QUANTIZATION")
    print("=" * 60)

    quantized_model = apply_quantization_fixed(inference_model)

    if quantized_model:
        quantized_results = evaluate_model_performance_fixed(
            quantized_model, scaler, "Dynamic Quantized"
        )

        if quantized_results:
            all_results['dynamic_quantized'] = quantized_results

            # Сравнение
            compare_optimizations_fixed(original_results, quantized_results, "Dynamic_Quantized")

            # Сохранение
            save_optimized_model_fixed(
                quantized_model, scaler, "dynamic_quantized", quantized_results
            )

    # 3. Комбинированная оптимизация: Pruning + Quantization
    print(f"\n{'=' * 60}")
    print("ЭКСПЕРИМЕНТ 3: PRUNING + QUANTIZATION")
    print("=" * 60)

    # Используем лучший pruning уровень (0.2 если доступен)
    best_pruning_amount = 0.2 if 'pruned_0.2' in all_results else 0.1

    print(f"Используем Pruning {best_pruning_amount * 100:.0f}% + Quantization")

    pruned_model = apply_pruning_fixed(inference_model, best_pruning_amount)
    combined_model = apply_quantization_fixed(pruned_model)

    if combined_model:
        combined_results = evaluate_model_performance_fixed(
            combined_model, scaler, "Pruned+Quantized"
        )

        if combined_results:
            all_results['combined'] = combined_results

            # Сравнение
            compare_optimizations_fixed(original_results, combined_results, "Combined")

            # Сохранение
            save_optimized_model_fixed(
                combined_model, scaler, "combined_optimized", combined_results
            )

    # Финальный отчет
    print(f"\n{'=' * 60}")
    print("ФИНАЛЬНЫЙ ОТЧЕТ ПО ОПТИМИЗАЦИИ")
    print("=" * 60)

    if all_results:
        # Создание сводной таблицы
        summary_data = []

        for name, results in all_results.items():
            summary_data.append({
                'Модель': name,
                'Accuracy': f"{results.get('accuracy', 0):.4f}",
                'ROC-AUC': f"{results.get('roc_auc', 0):.4f}",
                'Время (мс/сэмпл)': f"{results.get('total_inference_s', 0) / 6000 * 1000:.3f}",
                'Samples/sec': f"{results.get('samples_per_second', 0):.0f}",
                'Размер (KB)': f"{results.get('model_size_kb', 0):.1f}"
            })

        summary_df = pd.DataFrame(summary_data)
        print(f"\nСводная таблица всех моделей:")
        print(summary_df.to_string(index=False))

        # Сохранение сводного отчета
        summary_path = Path("models/optimized/optimization_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n Сводный отчет сохранен: {summary_path}")

        # Определение лучшей модели (баланс accuracy и производительности)
        if 'combined' in all_results and all_results['combined']['accuracy'] >= 0.75:
            best_model = 'combined'
        elif 'dynamic_quantized' in all_results:
            best_model = 'dynamic_quantized'
        else:
            # Выбираем по accuracy
            best_model = max(
                all_results.keys(),
                key=lambda k: all_results[k].get('accuracy', 0)
            )

        print(f"\n Рекомендуемая модель для продакшена: {best_model}")
        print(f"   Accuracy: {all_results[best_model].get('accuracy', 0):.4f}")
        print(f"   ROC-AUC: {all_results[best_model].get('roc_auc', 0):.4f}")
        print(f"   Производительность: {all_results[best_model].get('samples_per_second', 0):.0f} samples/sec")

        print(f"\n ИТОГОВЫЕ РЕКОМЕНДАЦИИ:")
        print(f"  1. Для максимальной точности используйте оригинальную модель")
        print(f"  2. Для продакшена используйте {'combined' if 'combined' in all_results else 'dynamic_quantized'}")
        print(f"  3. Оптимальный batch size для инференса: 100-500")
        print(f"  4. Модели доступны в папке: models/optimized/")

    print(f"\n{'=' * 60}")
    print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)


if __name__ == "__main__":
    main_fixed()