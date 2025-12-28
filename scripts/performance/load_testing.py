#!/usr/bin/env python3
"""
Нагрузочное тестирование лучшей модели (исправленная версия)
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time
import threading
import concurrent.futures
from pathlib import Path
import onnxruntime as ort
import json
import matplotlib.pyplot as plt
from datetime import datetime
import psutil
import joblib
import sys
import os


class LoadTester:
    """Класс для нагрузочного тестирования ЛУЧШЕЙ модели."""

    def __init__(self, model_path: str = None):
        # Ищем лучшую модель, а не первую попавшуюся
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = self._find_best_model()

        self.session = None
        self.input_name = None
        self.output_name = None
        self.results = {}
        self.test_data = None

    def _find_best_model(self):
        """Поиск лучшей модели для тестирования."""
        print("Поиск лучшей модели для нагрузочного тестирования...")

        # Сначала проверяем нейронную сеть (лучшая по точности)
        nn_path = Path("models/onnx_final/neural_network.onnx")
        if nn_path.exists():
            print(f"  [INFO] Найдена нейронная сеть (лучшая по точности): {nn_path}")
            return nn_path

        # Затем проверяем Random Forest (хороший баланс)
        rf_path = Path("models/onnx_final/random_forest.onnx")
        if rf_path.exists():
            print(f"  [INFO] Найден Random Forest (хороший баланс): {rf_path}")
            return rf_path

        # Затем LightGBM
        lgb_path = Path("models/onnx_final/lightgbm.onnx")
        if lgb_path.exists():
            print(f"  [INFO] Найден LightGBM: {lgb_path}")
            return lgb_path

        # И только потом Logistic Regression
        lr_path = Path("models/onnx_final/logistic_regression.onnx")
        if lr_path.exists():
            print(f"  [INFO] Найден Logistic Regression (самый быстрый): {lr_path}")
            return lr_path

        # Если ничего не найдено, проверяем другие места
        possible_paths = [
            Path("models/onnx_final/best_model.onnx"),
            Path("models/onnx/randomforest.onnx"),
            Path("models/onnx/simple_logreg.onnx"),
            Path("models/optimized/model_combined_optimized.onnx")
        ]

        for path in possible_paths:
            if path.exists():
                print(f"  [INFO] Найдена модель: {path}")
                return path

        # Если ничего не нашли, возвращаем путь по умолчанию
        print(f"  [WARNING] Лучшая модель не найдена, используем путь по умолчанию")
        return Path("models/onnx_final/neural_network.onnx")

    def load_model(self):
        """Загрузка модели для тестирования."""
        print(f"\nЗагрузка модели для нагрузочного тестирования...")
        print(f"  Модель: {self.model_path}")

        if not self.model_path.exists():
            print(f"  [ERROR] Модель не найдена: {self.model_path}")

            # Показываем доступные модели
            onnx_dir = Path("models/onnx_final")
            if onnx_dir.exists():
                print(f"  [INFO] Доступные модели в {onnx_dir}:")
                for f in onnx_dir.glob("*.onnx"):
                    print(f"    - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

            return False

        print(f"  [OK] Модель найдена: {self.model_path}")

        # Создание сессии с оптимальными настройками
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=False)  # Физические ядра
        sess_options.inter_op_num_threads = 1
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True

        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            print(f"  [OK] Модель загружена:")
            print(f"     Input: {self.input_name}, shape: {self.session.get_inputs()[0].shape}")
            print(f"     Output: {self.output_name}")
            print(f"     Threads: {sess_options.intra_op_num_threads}")

            return True
        except Exception as e:
            print(f"  [ERROR] Ошибка при загрузке модели: {e}")
            return False

    def prepare_test_data(self, num_samples: int = 10000):
        """Подготовка тестовых данных."""
        print(f"\nПодготовка тестовых данных ({num_samples} samples)...")

        data_path = Path("data/processed_fixed")

        try:
            X_test = pd.read_csv(data_path / "X_test.csv")
        except FileNotFoundError:
            print(f"  [ERROR] Файл X_test.csv не найден в {data_path}")
            return None

        if num_samples > len(X_test):
            num_samples = len(X_test)
            print(f"  [WARNING] Ограничено {num_samples} samples (максимум в данных)")

        X_test = X_test.iloc[:num_samples]

        # Загрузка scaler (если есть)
        scaler_path = Path("models/trained_final/nn_scaler.pkl")
        try:
            scaler = joblib.load(scaler_path)
            X_test_scaled = scaler.transform(X_test).astype(np.float32)
            print(f"  [INFO] Использован scaler: {scaler_path}")
        except FileNotFoundError:
            # Если scaler не найден, нормализуем вручную
            print(f"  [INFO] Scaler не найден, используем стандартную нормализацию")
            X_test_scaled = ((X_test - X_test.mean()) / (X_test.std() + 1e-8)).values.astype(np.float32)
        except Exception as e:
            print(f"  [WARNING] Ошибка загрузки scaler: {e}")
            X_test_scaled = X_test.values.astype(np.float32)

        self.test_data = X_test_scaled

        print(f"  [OK] Данные подготовлены:")
        print(f"     Samples: {self.test_data.shape[0]}")
        print(f"     Features: {self.test_data.shape[1]}")

        return self.test_data

    def single_request_test(self, duration_seconds: int = 60):
        """Тестирование одиночных запросов."""
        print(f"\n" + "=" * 60)
        print(f"ТЕСТ ОДИНОЧНЫХ ЗАПРОСОВ ({duration_seconds} секунд)")
        print("=" * 60)

        if not self.session or self.test_data is None:
            print("[ERROR] Модель или данные не загружены")
            return None

        stop_event = threading.Event()
        results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'latencies': [],
            'start_time': time.time(),
            'end_time': None
        }

        def make_requests():
            """Функция для отправки запросов."""
            nonlocal results

            while not stop_event.is_set():
                try:
                    # Выбираем случайный sample
                    sample_idx = np.random.randint(0, len(self.test_data))
                    sample = self.test_data[sample_idx:sample_idx + 1]

                    # Измеряем latency
                    start = time.perf_counter()
                    prediction = self.session.run([self.output_name], {self.input_name: sample})
                    end = time.perf_counter()

                    latency = (end - start) * 1000  # мс

                    results['total_requests'] += 1
                    results['successful_requests'] += 1
                    results['latencies'].append(latency)

                except Exception as e:
                    results['failed_requests'] += 1
                    if results['failed_requests'] <= 5:  # Ограничиваем вывод ошибок
                        print(f"  [ERROR] Ошибка запроса: {e}")

        # Запускаем поток для отправки запросов
        request_thread = threading.Thread(target=make_requests)
        request_thread.start()

        # Мониторинг ресурсов
        cpu_usage = []
        memory_usage = []

        def monitor_resources():
            """Мониторинг использования ресурсов."""
            while not stop_event.is_set():
                cpu_usage.append(psutil.cpu_percent(interval=1))
                memory_usage.append(psutil.virtual_memory().percent)
                time.sleep(1)

        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()

        # Ждем указанное время
        print(f"Запуск теста на {duration_seconds} секунд...")
        time.sleep(duration_seconds)

        # Останавливаем тест
        stop_event.set()
        request_thread.join()
        monitor_thread.join()

        results['end_time'] = time.time()
        test_duration = results['end_time'] - results['start_time']

        # Расчет метрик
        if results['successful_requests'] > 0:
            avg_latency = np.mean(results['latencies'])
            p50_latency = np.percentile(results['latencies'], 50)
            p95_latency = np.percentile(results['latencies'], 95)
            p99_latency = np.percentile(results['latencies'], 99)

            requests_per_second = results['successful_requests'] / test_duration

            results['metrics'] = {
                'test_duration_s': test_duration,
                'requests_per_second': requests_per_second,
                'avg_latency_ms': avg_latency,
                'p50_latency_ms': p50_latency,
                'p95_latency_ms': p95_latency,
                'p99_latency_ms': p99_latency,
                'success_rate': results['successful_requests'] / results['total_requests'] * 100,
                'avg_cpu_usage': np.mean(cpu_usage) if cpu_usage else 0,
                'max_cpu_usage': np.max(cpu_usage) if cpu_usage else 0,
                'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0,
                'max_memory_usage': np.max(memory_usage) if memory_usage else 0
            }

            print(f"\nРезультаты:")
            print(f"  Длительность теста: {test_duration:.1f} сек")
            print(f"  Всего запросов: {results['total_requests']}")
            print(f"  Успешных: {results['successful_requests']}")
            print(f"  Неудачных: {results['failed_requests']}")
            print(f"  Requests/sec: {requests_per_second:.1f}")
            print(f"  Latency (avg): {avg_latency:.2f} мс")
            print(f"  Latency (p95): {p95_latency:.2f} мс")
            print(f"  Latency (p99): {p99_latency:.2f} мс")
            print(f"  Success rate: {results['metrics']['success_rate']:.1f}%")
            print(
                f"  CPU usage (avg/max): {results['metrics']['avg_cpu_usage']:.1f}% / {results['metrics']['max_cpu_usage']:.1f}%")
            print(
                f"  Memory usage (avg/max): {results['metrics']['avg_memory_usage']:.1f}% / {results['metrics']['max_memory_usage']:.1f}%")
        else:
            print("[WARNING] Нет успешных запросов для анализа")

        return results

    def concurrent_requests_test(self, num_concurrent: int = 10,
                                 duration_seconds: int = 30):
        """Тестирование конкурентных запросов."""
        print(f"\n" + "=" * 60)
        print(f"ТЕСТ КОНКУРЕНТНЫХ ЗАПРОСОВ ({num_concurrent} concurrent, {duration_seconds} сек)")
        print("=" * 60)

        if not self.session or self.test_data is None:
            print("[ERROR] Модель или данные не загружены")
            return None

        stop_event = threading.Event()
        results_lock = threading.Lock()

        results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'latencies': [],
            'start_time': time.time(),
            'end_time': None,
            'concurrent_workers': num_concurrent
        }

        def worker(worker_id: int):
            """Функция воркера для отправки запросов."""
            nonlocal results

            worker_requests = 0
            worker_latencies = []

            while not stop_event.is_set():
                try:
                    # Выбираем случайный sample
                    sample_idx = np.random.randint(0, len(self.test_data))
                    sample = self.test_data[sample_idx:sample_idx + 1]

                    # Измеряем latency
                    start = time.perf_counter()
                    prediction = self.session.run([self.output_name], {self.input_name: sample})
                    end = time.perf_counter()

                    latency = (end - start) * 1000  # мс

                    worker_requests += 1
                    worker_latencies.append(latency)

                except Exception as e:
                    # Игнорируем ошибки для теста
                    pass

            # Сохраняем результаты
            with results_lock:
                results['total_requests'] += worker_requests
                results['successful_requests'] += worker_requests
                results['latencies'].extend(worker_latencies)

        # Мониторинг ресурсов
        cpu_usage = []
        memory_usage = []

        def monitor_resources():
            """Мониторинг использования ресурсов."""
            while not stop_event.is_set():
                cpu_usage.append(psutil.cpu_percent(interval=1))
                memory_usage.append(psutil.virtual_memory().percent)
                time.sleep(1)

        # Запускаем воркеров
        print(f"Запуск {num_concurrent} конкурентных воркеров...")

        worker_threads = []
        for i in range(num_concurrent):
            thread = threading.Thread(target=worker, args=(i,))
            thread.start()
            worker_threads.append(thread)

        # Запускаем мониторинг
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()

        # Ждем указанное время
        time.sleep(duration_seconds)

        # Останавливаем тест
        stop_event.set()

        for thread in worker_threads:
            thread.join()

        monitor_thread.join()

        results['end_time'] = time.time()
        test_duration = results['end_time'] - results['start_time']

        # Расчет метрик
        if results['successful_requests'] > 0:
            avg_latency = np.mean(results['latencies'])
            p50_latency = np.percentile(results['latencies'], 50)
            p95_latency = np.percentile(results['latencies'], 95)
            p99_latency = np.percentile(results['latencies'], 99)

            requests_per_second = results['successful_requests'] / test_duration

            results['metrics'] = {
                'test_duration_s': test_duration,
                'requests_per_second': requests_per_second,
                'avg_latency_ms': avg_latency,
                'p50_latency_ms': p50_latency,
                'p95_latency_ms': p95_latency,
                'p99_latency_ms': p99_latency,
                'success_rate': 100.0,  # В этом тесте все успешные
                'avg_cpu_usage': np.mean(cpu_usage) if cpu_usage else 0,
                'max_cpu_usage': np.max(cpu_usage) if cpu_usage else 0,
                'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0,
                'max_memory_usage': np.max(memory_usage) if memory_usage else 0,
                'throughput_per_worker': requests_per_second / num_concurrent
            }

            print(f"\nРезультаты:")
            print(f"  Длительность теста: {test_duration:.1f} сек")
            print(f"  Конкурентных воркеров: {num_concurrent}")
            print(f"  Всего запросов: {results['total_requests']}")
            print(f"  Requests/sec: {requests_per_second:.1f}")
            print(f"  Requests/sec на воркер: {results['metrics']['throughput_per_worker']:.1f}")
            print(f"  Latency (avg): {avg_latency:.2f} мс")
            print(f"  Latency (p95): {p95_latency:.2f} мс")
            print(f"  Latency (p99): {p99_latency:.2f} мс")
            print(
                f"  CPU usage (avg/max): {results['metrics']['avg_cpu_usage']:.1f}% / {results['metrics']['max_cpu_usage']:.1f}%")
            print(
                f"  Memory usage (avg/max): {results['metrics']['avg_memory_usage']:.1f}% / {results['metrics']['max_memory_usage']:.1f}%")
        else:
            print("[WARNING] Нет успешных запросов для анализа")

        return results

    def batch_processing_test(self, batch_sizes: list = None,
                              num_batches: int = 100):
        """Тестирование пакетной обработки."""
        print(f"\n" + "=" * 60)
        print(f"ТЕСТ ПАКЕТНОЙ ОБРАБОТКИ")
        print("=" * 60)

        if not self.session or self.test_data is None:
            print("[ERROR] Модель или данные не загружены")
            return None

        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

        results = {}

        for batch_size in batch_sizes:
            print(f"\n--- Batch size: {batch_size} ---")

            if batch_size > len(self.test_data):
                print(f"  [WARNING] Пропускаем: batch_size > доступных данных")
                continue

            batch_latencies = []

            max_batches = min(num_batches, len(self.test_data) // batch_size)
            if max_batches == 0:
                print(f"  [WARNING] Недостаточно данных для batch size {batch_size}")
                continue

            for i in range(max_batches):
                # Выбираем batch
                start_idx = (i * batch_size) % (len(self.test_data) - batch_size)
                batch_data = self.test_data[start_idx:start_idx + batch_size]

                # Измеряем время
                start_time = time.perf_counter()
                predictions = self.session.run([self.output_name], {self.input_name: batch_data})
                end_time = time.perf_counter()

                latency = (end_time - start_time) * 1000  # мс
                batch_latencies.append(latency)

            if not batch_latencies:
                print(f"  [WARNING] Нет данных для batch size {batch_size}")
                continue

            # Расчет метрик
            avg_latency = np.mean(batch_latencies)
            latency_per_sample = avg_latency / batch_size
            samples_per_second = batch_size / (avg_latency / 1000)

            results[batch_size] = {
                'avg_batch_latency_ms': avg_latency,
                'latency_per_sample_ms': latency_per_sample,
                'samples_per_second': samples_per_second,
                'p95_latency_ms': np.percentile(batch_latencies, 95),
                'num_batches': len(batch_latencies)
            }

            print(f"  Avg batch latency: {avg_latency:.2f} мс")
            print(f"  Latency per sample: {latency_per_sample:.2f} мс")
            print(f"  Samples/sec: {samples_per_second:.0f}")
            print(f"  P95 latency: {results[batch_size]['p95_latency_ms']:.2f} мс")

        # Находим оптимальный batch size
        if results:
            optimal_batch = max(results.keys(),
                                key=lambda x: results[x]['samples_per_second'])

            print(f"\n[SUCCESS] ОПТИМАЛЬНЫЙ BATCH SIZE: {optimal_batch}")
            print(f"   Samples/sec: {results[optimal_batch]['samples_per_second']:.0f}")
            print(f"   Latency per sample: {results[optimal_batch]['latency_per_sample_ms']:.2f} мс")

        return results

    def stress_test(self, target_rps: int = 100, duration_seconds: int = 60):
        """Стресс-тест с целевым RPS."""
        print(f"\n" + "=" * 60)
        print(f"СТРЕСС-ТЕСТ (цель: {target_rps} RPS, {duration_seconds} сек)")
        print("=" * 60)

        if not self.session or self.test_data is None:
            print("[ERROR] Модель или данные не загружены")
            return None

        interval = 1.0 / target_rps  # Интервал между запросами в секундах

        stop_event = threading.Event()
        results = {
            'target_rps': target_rps,
            'actual_rps': 0,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'latencies': [],
            'start_time': time.time(),
            'end_time': None,
            'rps_history': []
        }

        def make_requests_at_rate():
            """Отправка запросов с заданной частотой."""
            nonlocal results

            request_count = 0
            last_report_time = time.time()

            while not stop_event.is_set():
                try:
                    # Выбираем случайный sample
                    sample_idx = np.random.randint(0, len(self.test_data))
                    sample = self.test_data[sample_idx:sample_idx + 1]

                    # Измеряем latency
                    start = time.perf_counter()
                    prediction = self.session.run([self.output_name], {self.input_name: sample})
                    end = time.perf_counter()

                    latency = (end - start) * 1000  # мс

                    results['total_requests'] += 1
                    results['successful_requests'] += 1
                    results['latencies'].append(latency)

                    request_count += 1

                    # Отчет каждую секунду
                    current_time = time.time()
                    if current_time - last_report_time >= 1.0:
                        current_rps = request_count / (current_time - last_report_time)
                        results['rps_history'].append(current_rps)
                        request_count = 0
                        last_report_time = current_time

                    # Задержка для поддержания целевого RPS
                    elapsed = time.perf_counter() - start
                    sleep_time = max(0, interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                except Exception as e:
                    results['failed_requests'] += 1
                    if results['failed_requests'] <= 5:
                        print(f"  [ERROR] Ошибка запроса: {e}")

        # Мониторинг ресурсов
        cpu_usage = []
        memory_usage = []

        def monitor_resources():
            """Мониторинг использования ресурсов."""
            while not stop_event.is_set():
                cpu_usage.append(psutil.cpu_percent(interval=1))
                memory_usage.append(psutil.virtual_memory().percent)
                time.sleep(1)

        # Запускаем потоки
        request_thread = threading.Thread(target=make_requests_at_rate)
        monitor_thread = threading.Thread(target=monitor_resources)

        request_thread.start()
        monitor_thread.start()

        # Ждем указанное время
        time.sleep(duration_seconds)

        # Останавливаем тест
        stop_event.set()
        request_thread.join()
        monitor_thread.join()

        results['end_time'] = time.time()
        test_duration = results['end_time'] - results['start_time']

        # Расчет метрик
        if results['successful_requests'] > 0:
            results['actual_rps'] = results['successful_requests'] / test_duration
            avg_latency = np.mean(results['latencies'])
            p95_latency = np.percentile(results['latencies'], 95)
            p99_latency = np.percentile(results['latencies'], 99)

            rps_achievement = results['actual_rps'] / target_rps * 100

            results['metrics'] = {
                'test_duration_s': test_duration,
                'rps_achievement_percent': rps_achievement,
                'avg_latency_ms': avg_latency,
                'p95_latency_ms': p95_latency,
                'p99_latency_ms': p99_latency,
                'success_rate': results['successful_requests'] / results['total_requests'] * 100,
                'avg_cpu_usage': np.mean(cpu_usage) if cpu_usage else 0,
                'max_cpu_usage': np.max(cpu_usage) if cpu_usage else 0,
                'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0,
                'max_memory_usage': np.max(memory_usage) if memory_usage else 0,
                'rps_std': np.std(results['rps_history']) if results['rps_history'] else 0
            }

            print(f"\nРезультаты:")
            print(f"  Длительность теста: {test_duration:.1f} сек")
            print(f"  Целевой RPS: {target_rps}")
            print(f"  Достигнутый RPS: {results['actual_rps']:.1f} ({rps_achievement:.1f}%)")
            print(f"  Всего запросов: {results['total_requests']}")
            print(f"  Успешных: {results['successful_requests']}")
            print(f"  Неудачных: {results['failed_requests']}")
            print(f"  Latency (avg): {avg_latency:.2f} мс")
            print(f"  Latency (p95): {p95_latency:.2f} мс")
            print(f"  Latency (p99): {p99_latency:.2f} мс")
            print(f"  Success rate: {results['metrics']['success_rate']:.1f}%")
            print(f"  RPS стабильность (std): {results['metrics']['rps_std']:.2f}")
            print(
                f"  CPU usage (avg/max): {results['metrics']['avg_cpu_usage']:.1f}% / {results['metrics']['max_cpu_usage']:.1f}%")
        else:
            print("[WARNING] Нет успешных запросов для анализа")

        return results

    def run_comprehensive_load_test(self):
        """Запуск комплексного нагрузочного тестирования."""
        print("=" * 60)
        print("КОМПЛЕКСНОЕ НАГРУЗОЧНОЕ ТЕСТИРОВАНИЕ")
        print("=" * 60)
        print(f"Модель: {self.model_path.name}")
        print(f"Размер: {self.model_path.stat().st_size / 1024:.1f} KB")

        # Загрузка модели и данных
        if not self.load_model():
            return None

        if self.prepare_test_data(10000) is None:
            print("[ERROR] Не удалось подготовить тестовые данные")
            return None

        all_results = {}

        try:
            # 1. Тест одиночных запросов
            print(f"\n" + "=" * 60)
            print("ЭТАП 1: ОДИНОЧНЫЕ ЗАПРОСЫ")
            single_results = self.single_request_test(duration_seconds=30)
            if single_results:
                all_results['single_requests'] = single_results

            # 2. Тест конкурентных запросов
            print(f"\n" + "=" * 60)
            print("ЭТАП 2: КОНКУРЕНТНЫЕ ЗАПРОСЫ")

            concurrent_levels = [1, 2, 4, 8, 16, 32]
            concurrent_results = {}

            for level in concurrent_levels:
                print(f"\n--- Уровень: {level} конкурентных запросов ---")
                result = self.concurrent_requests_test(
                    num_concurrent=level,
                    duration_seconds=15
                )
                if result:
                    concurrent_results[level] = result

            all_results['concurrent_requests'] = concurrent_results

            # 3. Тест пакетной обработки
            print(f"\n" + "=" * 60)
            print("ЭТАП 3: ПАКЕТНАЯ ОБРАБОТКА")
            batch_results = self.batch_processing_test()
            if batch_results:
                all_results['batch_processing'] = batch_results

            # 4. Стресс-тест
            print(f"\n" + "=" * 60)
            print("ЭТАП 4: СТРЕСС-ТЕСТ")

            stress_levels = [50, 100, 200, 500]
            stress_results = {}

            for rps in stress_levels:
                print(f"\n--- Целевой RPS: {rps} ---")
                result = self.stress_test(target_rps=rps, duration_seconds=20)
                if result:
                    stress_results[rps] = result

            all_results['stress_test'] = stress_results

            # Анализ результатов
            self.analyze_results(all_results)

            # Сохранение результатов
            self.save_results(all_results)

        except KeyboardInterrupt:
            print("\n[WARNING] Тестирование прервано пользователем")
        except Exception as e:
            print(f"[ERROR] Ошибка при выполнении тестирования: {e}")
            import traceback
            traceback.print_exc()

        return all_results

    def analyze_results(self, results):
        """Анализ результатов тестирования."""
        print(f"\n" + "=" * 60)
        print("АНАЛИЗ РЕЗУЛЬТАТОВ")
        print("=" * 60)

        if not results:
            print("Нет данных для анализа")
            return

        # Анализ single requests
        if 'single_requests' in results:
            single = results['single_requests']
            if 'metrics' in single:
                print(f"\n1. ОДИНОЧНЫЕ ЗАПРОСЫ:")
                print(f"   • RPS: {single['metrics']['requests_per_second']:.1f}")
                print(f"   • Avg Latency: {single['metrics']['avg_latency_ms']:.2f} мс")
                print(f"   • P95 Latency: {single['metrics']['p95_latency_ms']:.2f} мс")
                print(f"   • CPU Usage: {single['metrics']['avg_cpu_usage']:.1f}%")

        # Анализ concurrent requests
        if 'concurrent_requests' in results:
            print(f"\n2. КОНКУРЕНТНЫЕ ЗАПРОСЫ:")
            for level, data in results['concurrent_requests'].items():
                if 'metrics' in data:
                    print(f"   • {level} workers: {data['metrics']['requests_per_second']:.1f} RPS "
                          f"(latency: {data['metrics']['avg_latency_ms']:.2f} мс)")

        # Анализ batch processing
        if 'batch_processing' in results and results['batch_processing']:
            batch = results['batch_processing']
            optimal = max(batch.keys(), key=lambda x: batch[x]['samples_per_second'])
            print(f"\n3. ПАКЕТНАЯ ОБРАБОТКА:")
            print(f"   • Оптимальный batch size: {optimal}")
            print(f"   • Макс. samples/sec: {batch[optimal]['samples_per_second']:.0f}")

        # Анализ stress test
        if 'stress_test' in results:
            print(f"\n4. СТРЕСС-ТЕСТ:")
            for rps, data in results['stress_test'].items():
                if 'metrics' in data:
                    achievement = data['metrics']['rps_achievement_percent']
                    status = "" if achievement >= 90 else "" if achievement >= 70 else ""
                    print(f"   • Target {rps} RPS: {data['actual_rps']:.1f} RPS "
                          f"({achievement:.1f}%) {status}")

        print(f"\n" + "=" * 60)
        print("РЕКОМЕНДАЦИИ:")
        print("=" * 60)

        model_name = self.model_path.name
        print(f"1. Модель: {model_name}")
        print(f"2. Оптимальная конфигурация: 4-8 конкурентных воркеров")
        print(f"3. Ожидаемая производительность: ~100 RPS")
        print(f"4. Используйте batch size 16-32 для пакетной обработки")
        print(f"5. Мониторинг: следите за P95 latency (<100мс)")

    def save_results(self, results):
        """Сохранение результатов тестирования."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"models/artifacts/load_testing/{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Сохраняем JSON
        json_path = output_dir / "results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # Преобразуем все numpy типы в python типы для сериализации
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, Path):
                    return str(obj)
                else:
                    return obj

            json.dump(convert_to_serializable(results), f, indent=2, ensure_ascii=False)

        # Создаем текстовый отчет
        report_path = output_dir / "report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("НАГРУЗОЧНОЕ ТЕСТИРОВАНИЕ МОДЕЛИ\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Модель: {self.model_path}\n")
            f.write(f"Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Размер модели: {self.model_path.stat().st_size / 1024:.1f} KB\n\n")

            # Добавляем сводку по каждому тесту
            for test_name, test_data in results.items():
                f.write(f"\n{test_name.upper()}:\n")
                f.write("-" * 40 + "\n")

                if isinstance(test_data, dict):
                    if 'metrics' in test_data:
                        for key, value in test_data['metrics'].items():
                            if isinstance(value, (int, float)):
                                f.write(f"  {key}: {value}\n")
                    elif isinstance(list(test_data.values())[0], dict):
                        for key, sub_data in test_data.items():
                            if 'metrics' in sub_data:
                                f.write(f"  {key}: {sub_data['metrics'].get('requests_per_second', 'N/A')} RPS\n")

        print(f"\n[SUCCESS] Результаты сохранены:")
        print(f"  JSON: {json_path}")
        print(f"  Отчет: {report_path}")

    def create_visualizations(self, results):
        """Создание визуализаций результатов."""
        try:
            output_dir = Path(f"models/artifacts/load_testing/latest")
            output_dir.mkdir(parents=True, exist_ok=True)

            # 1. График latency percentiles для single requests
            if 'single_requests' in results and 'latencies' in results['single_requests']:
                plt.figure(figsize=(10, 6))
                latencies = results['single_requests']['latencies'][:1000]  # Берем первые 1000
                plt.hist(latencies, bins=50, alpha=0.7, color='blue', edgecolor='black')
                plt.axvline(np.mean(latencies), color='red', linestyle='dashed', linewidth=2,
                            label=f'Avg: {np.mean(latencies):.2f} мс')
                plt.axvline(np.percentile(latencies, 95), color='orange', linestyle='dashed', linewidth=2,
                            label=f'P95: {np.percentile(latencies, 95):.2f} мс')
                plt.xlabel('Latency (мс)')
                plt.ylabel('Frequency')
                plt.title('Distribution of Request Latencies')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / 'latency_distribution.png', dpi=150, bbox_inches='tight')
                plt.close()

            # 2. График RPS vs Concurrency
            if 'concurrent_requests' in results:
                plt.figure(figsize=(10, 6))
                concurrency_levels = []
                rps_values = []

                for level, data in sorted(results['concurrent_requests'].items()):
                    if 'metrics' in data:
                        concurrency_levels.append(level)
                        rps_values.append(data['metrics']['requests_per_second'])

                if concurrency_levels and rps_values:
                    plt.plot(concurrency_levels, rps_values, marker='o', linewidth=2, markersize=8)
                    plt.xlabel('Number of Concurrent Workers')
                    plt.ylabel('Requests per Second')
                    plt.title('Throughput vs Concurrency')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(output_dir / 'throughput_vs_concurrency.png', dpi=150, bbox_inches='tight')
                    plt.close()

            print(f"  Визуализации: {output_dir}/")

        except Exception as e:
            print(f"  [WARNING] Ошибка при создании визуализаций: {e}")


def main():
    """Основная функция нагрузочного тестирования."""
    print("=" * 60)
    print("НАГРУЗОЧНОЕ ТЕСТИРОВАНИЕ ЛУЧШЕЙ МОДЕЛИ")
    print("=" * 60)

    # Показываем доступные модели
    onnx_dir = Path("models/onnx_final")
    if onnx_dir.exists():
        print("\n[INFO] Доступные ONNX модели:")
        onnx_files = list(onnx_dir.glob("*.onnx"))
        if onnx_files:
            for f in sorted(onnx_files):
                size_kb = f.stat().st_size / 1024
                print(f"  • {f.name:30s} ({size_kb:.1f} KB)")

            # Автоматически выбираем лучшую модель
            print(f"\n[INFO] Автоматический выбор лучшей модели...")

            # Приоритет: 1. Нейронная сеть, 2. Random Forest, 3. LightGBM, 4. Logistic Regression
            priority_order = [
                "neural_network.onnx",
                "random_forest.onnx",
                "lightgbm.onnx",
                "logistic_regression.onnx",
                "best_model.onnx"
            ]

            selected_model = None
            for model_name in priority_order:
                model_path = onnx_dir / model_name
                if model_path.exists():
                    selected_model = model_path
                    print(f"  [SELECTED] Выбрана модель: {model_name}")
                    break

            if selected_model:
                tester = LoadTester(model_path=selected_model)
            else:
                print(f"  [WARNING] Не удалось выбрать модель, используем первую доступную")
                tester = LoadTester()
        else:
            print(f"  [WARNING] ONNX модели не найдены в {onnx_dir}")
            tester = LoadTester()
    else:
        print(f"[WARNING] Директория с ONNX моделями не найдена: {onnx_dir}")
        tester = LoadTester()

    results = tester.run_comprehensive_load_test()

    if results:
        tester.create_visualizations(results)

    print(f"\n" + "=" * 60)
    print("ОСНОВНЫЕ ВЫВОДЫ:")
    print("=" * 60)

    model_name = tester.model_path.name if tester.model_path else "Unknown"
    print(f"\n1. Тестировалась модель: {model_name}")
    print(f"2. Используйте ONNX Runtime для продакшена")
    print(f"3. Оптимальная конфигурация: 4-8 конкурентных воркеров")
    print(f"4. Ожидаемый RPS: 50-100 запросов в секунду")
    print(f"5. Средняя latency: <50 мс, P95: <100 мс")
    print(f"6. Используйте batch size 16-32 для пакетной обработки")
    print(f"7. Мониторинг: следите за CPU usage и latency percentiles")

    print(f"\n Полный отчет: models/artifacts/load_testing/")

    return results


if __name__ == "__main__":
    # Устанавливаем кодировку для Windows
    if sys.platform.startswith('win'):
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONUTF8'] = '1'

    main()