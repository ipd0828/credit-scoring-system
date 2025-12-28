#!/usr/bin/env python3
"""
Углубленный бенчмарк производительности PyTorch vs ONNX на CPU.
Сравнение нейронной сети и других моделей.

Вход: models/trained_final/ и models/onnx_final/
Выход: Детальный отчет о производительности
"""

import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
import time
import psutil
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")


class DeepPerformanceBenchmark:
    """Углубленный бенчмарк производительности."""

    def __init__(self):
        self.results = {}
        self.device_info = self._get_device_info()

    def _get_device_info(self):
        """Получение информации об устройстве."""
        info = {
            'cpu': {
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'
            },
            'memory': {
                'total_gb': psutil.virtual_memory().total / 1024 ** 3,
                'available_gb': psutil.virtual_memory().available / 1024 ** 3
            },
            'software': {
                'pytorch': torch.__version__,
                'onnxruntime': ort.__version__,
                'numpy': np.__version__
            }
        }
        return info

    def print_device_info(self):
        """Вывод информации об устройстве."""
        print("=" * 70)
        print("ИНФОРМАЦИЯ ОБ УСТРОЙСТВЕ ДЛЯ БЕНЧМАРКА")
        print("=" * 70)

        print(f"\n[CPU] Процессор:")
        print(f"  • Ядра: {self.device_info['cpu']['cores']} физических")
        print(f"  • Потоки: {self.device_info['cpu']['threads']} логических")
        print(f"  • Частота: {self.device_info['cpu']['frequency']} MHz")

        print(f"\n[MEMORY] Память:")
        print(f"  • Всего: {self.device_info['memory']['total_gb']:.1f} GB")
        print(f"  • Доступно: {self.device_info['memory']['available_gb']:.1f} GB")

        print(f"\n[SOFTWARE] Версии ПО:")
        for name, version in self.device_info['software'].items():
            print(f"  • {name}: {version}")

    def benchmark_neural_network_cpu(self, num_samples=10000):
        """Бенчмарк нейронной сети: PyTorch CPU vs ONNX CPU."""
        print(f"\n" + "=" * 70)
        print("БЕНЧМАРК: НЕЙРОННАЯ СЕТЬ (PyTorch CPU vs ONNX CPU)")
        print("=" * 70)

        try:
            # Пути к файлам
            torch_path = Path("models/trained_final/neural_network.pth")
            scaler_path = Path("models/trained_final/nn_scaler.pkl")
            onnx_path = Path("models/onnx_final/neural_network.onnx")

            if not torch_path.exists() or not onnx_path.exists():
                print("[ERROR] Модели не найдены!")
                return None

            print(f"[MODELS] PyTorch: {torch_path.name}, ONNX: {onnx_path.name}")

            # Подготовка данных
            print("[DATA] Подготовка тестовых данных...")

            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                X_test = np.random.randn(num_samples, 6).astype(np.float32)
                X_scaled = scaler.transform(X_test)
            else:
                X_scaled = np.random.randn(num_samples, 6).astype(np.float32)

            # PyTorch модель
            print("[TORCH] Загрузка PyTorch модели...")

            class SimpleNN(torch.nn.Module):
                def __init__(self, input_size=6):
                    super().__init__()
                    self.network = torch.nn.Sequential(
                        torch.nn.Linear(input_size, 64),
                        torch.nn.ReLU(),
                        torch.nn.Linear(64, 32),
                        torch.nn.ReLU(),
                        torch.nn.Linear(32, 16),
                        torch.nn.ReLU(),
                        torch.nn.Linear(16, 1),
                        torch.nn.Sigmoid()
                    )

                def forward(self, x):
                    return self.network(x)

            torch_model = SimpleNN(6)
            torch_model.load_state_dict(torch.load(torch_path, map_location='cpu'))
            torch_model.eval()

            # ONNX модель
            print("[ONNX] Загрузка ONNX модели...")
            ort_session = ort.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )
            input_name = ort_session.get_inputs()[0].name

            # Тестирование разных batch sizes
            batch_sizes = [1, 10, 50, 100, 500, 1000, 5000]
            results = {'torch': {}, 'onnx': {}}

            for batch_size in batch_sizes:
                if batch_size > num_samples:
                    continue

                print(f"\n[BATCH] Размер батча: {batch_size}")
                batch_data = X_scaled[:batch_size]
                batch_tensor = torch.FloatTensor(batch_data)

                # PyTorch
                torch_times = []
                torch_memory_before = psutil.Process().memory_info().rss / 1024 ** 2

                # Прогрев
                with torch.no_grad():
                    _ = torch_model(batch_tensor)

                # Измерение
                for i in range(10):
                    start = time.perf_counter()
                    with torch.no_grad():
                        _ = torch_model(batch_tensor)
                    end = time.perf_counter()
                    torch_times.append(end - start)

                torch_memory_after = psutil.Process().memory_info().rss / 1024 ** 2
                torch_memory_used = torch_memory_after - torch_memory_before

                torch_avg = np.mean(torch_times)

                # ONNX
                onnx_times = []
                onnx_memory_before = psutil.Process().memory_info().rss / 1024 ** 2

                # Прогрев
                _ = ort_session.run(None, {input_name: batch_data})

                # Измерение
                for i in range(10):
                    start = time.perf_counter()
                    _ = ort_session.run(None, {input_name: batch_data})
                    end = time.perf_counter()
                    onnx_times.append(end - start)

                onnx_memory_after = psutil.Process().memory_info().rss / 1024 ** 2
                onnx_memory_used = onnx_memory_after - onnx_memory_before

                onnx_avg = np.mean(onnx_times)

                # Сохранение результатов
                results['torch'][batch_size] = {
                    'avg_time_ms': torch_avg * 1000,
                    'time_per_sample_ms': torch_avg / batch_size * 1000,
                    'samples_per_second': batch_size / torch_avg,
                    'memory_mb': torch_memory_used
                }

                results['onnx'][batch_size] = {
                    'avg_time_ms': onnx_avg * 1000,
                    'time_per_sample_ms': onnx_avg / batch_size * 1000,
                    'samples_per_second': batch_size / onnx_avg,
                    'memory_mb': onnx_memory_used
                }

                # Сравнение
                speedup = torch_avg / onnx_avg if onnx_avg > 0 else 0

                print(
                    f"  PyTorch: {torch_avg / batch_size * 1000:6.3f} ms/sample, {batch_size / torch_avg:6.0f} samples/sec")
                print(
                    f"  ONNX:    {onnx_avg / batch_size * 1000:6.3f} ms/sample, {batch_size / onnx_avg:6.0f} samples/sec")
                print(f"  Ускорение: {speedup:.2f}x")
                print(f"  Память: {torch_memory_used:.1f} MB vs {onnx_memory_used:.1f} MB")

            self.results['neural_network'] = results
            self._plot_nn_comparison(results, batch_sizes)

            return results

        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return None

    def benchmark_all_onnx_models(self, num_samples=5000):
        """Бенчмарк всех ONNX моделей."""
        print(f"\n" + "=" * 70)
        print("БЕНЧМАРК: ВСЕ ONNX МОДЕЛИ")
        print("=" * 70)

        try:
            onnx_dir = Path("models/onnx_final")
            onnx_files = list(onnx_dir.glob("*.onnx"))

            if not onnx_files:
                print("[ERROR] ONNX модели не найдены!")
                return None

            # Подготовка данных
            X_test = np.random.randn(num_samples, 6).astype(np.float32)
            results = {}

            for onnx_path in onnx_files:
                model_name = onnx_path.stem
                print(f"\n[MODEL] {model_name}...")

                try:
                    ort_session = ort.InferenceSession(str(onnx_path))
                    input_name = ort_session.get_inputs()[0].name

                    # Тестирование для batch=100 (типичный для онлайн)
                    batch_size = 100
                    batch_data = X_test[:batch_size]

                    # Измерение времени
                    times = []

                    # Прогрев
                    _ = ort_session.run(None, {input_name: batch_data})

                    for i in range(20):
                        start = time.perf_counter()
                        _ = ort_session.run(None, {input_name: batch_data})
                        end = time.perf_counter()
                        times.append(end - start)

                    avg_time = np.mean(times)

                    results[model_name] = {
                        'file_size_kb': onnx_path.stat().st_size / 1024,
                        'latency_ms': avg_time * 1000,
                        'latency_per_sample_ms': avg_time / batch_size * 1000,
                        'throughput_rps': batch_size / avg_time,
                        'memory_mb': psutil.Process().memory_info().rss / 1024 ** 2
                    }

                    print(f"  Размер: {results[model_name]['file_size_kb']:.1f} KB")
                    print(f"  Latency: {results[model_name]['latency_per_sample_ms']:.3f} ms/sample")
                    print(f"  Throughput: {results[model_name]['throughput_rps']:.0f} requests/sec")

                except Exception as e:
                    print(f"  [ERROR] {e}")

            self.results['all_onnx'] = results
            self._plot_all_models_comparison(results)

            return results

        except Exception as e:
            print(f"[ERROR] {e}")
            return None

    def _plot_nn_comparison(self, results, batch_sizes):
        """График сравнения PyTorch vs ONNX."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Данные
            torch_latency = [results['torch'][bs]['time_per_sample_ms'] for bs in batch_sizes]
            onnx_latency = [results['onnx'][bs]['time_per_sample_ms'] for bs in batch_sizes]
            torch_throughput = [results['torch'][bs]['samples_per_second'] for bs in batch_sizes]
            onnx_throughput = [results['onnx'][bs]['samples_per_second'] for bs in batch_sizes]

            # График 1: Latency
            ax1.plot(batch_sizes, torch_latency, 'o-', label='PyTorch', linewidth=2, markersize=8)
            ax1.plot(batch_sizes, onnx_latency, 's-', label='ONNX', linewidth=2, markersize=8)
            ax1.set_xlabel('Batch Size', fontsize=12)
            ax1.set_ylabel('Latency per Sample (ms)', fontsize=12)
            ax1.set_title('Latency: PyTorch vs ONNX (Neural Network)', fontsize=14, pad=20)
            ax1.set_xscale('log')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(fontsize=11)

            # График 2: Throughput
            ax2.plot(batch_sizes, torch_throughput, 'o-', label='PyTorch', linewidth=2, markersize=8)
            ax2.plot(batch_sizes, onnx_throughput, 's-', label='ONNX', linewidth=2, markersize=8)
            ax2.set_xlabel('Batch Size', fontsize=12)
            ax2.set_ylabel('Throughput (samples/sec)', fontsize=12)
            ax2.set_title('Throughput: PyTorch vs ONNX (Neural Network)', fontsize=14, pad=20)
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(fontsize=11)

            plt.tight_layout()

            # Сохранение
            charts_dir = Path("models/artifacts/deep_benchmark")
            charts_dir.mkdir(parents=True, exist_ok=True)

            chart_path = charts_dir / 'pytorch_vs_onnx_comparison.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[OK] График сохранен: {chart_path}")

        except Exception as e:
            print(f"[WARNING] Не удалось создать график: {e}")

    def _plot_all_models_comparison(self, results):
        """График сравнения всех моделей."""
        try:
            if not results:
                return

            model_names = list(results.keys())
            latencies = [results[m]['latency_per_sample_ms'] for m in model_names]
            throughputs = [results[m]['throughput_rps'] for m in model_names]
            sizes = [results[m]['file_size_kb'] for m in model_names]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # График 1: Latency
            bars1 = ax1.bar(model_names, latencies, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Model', fontsize=12)
            ax1.set_ylabel('Latency per Sample (ms)', fontsize=12)
            ax1.set_title('Latency Comparison (batch=100)', fontsize=14, pad=20)
            ax1.set_xticklabels(model_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3, axis='y')

            for bar, latency in zip(bars1, latencies):
                ax1.text(bar.get_x() + bar.get_width() / 2., latency + 0.01,
                         f'{latency:.3f}', ha='center', va='bottom', fontsize=9)

            # График 2: Throughput vs Size
            ax2.scatter(sizes, throughputs, s=100, alpha=0.6, edgecolors='black')

            # Добавляем метки
            for i, (model, size, throughput) in enumerate(zip(model_names, sizes, throughputs)):
                ax2.annotate(model, (size, throughput), fontsize=9)

            ax2.set_xlabel('Model Size (KB)', fontsize=12)
            ax2.set_ylabel('Throughput (requests/sec)', fontsize=12)
            ax2.set_title('Throughput vs Model Size', fontsize=14, pad=20)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            charts_dir = Path("models/artifacts/deep_benchmark")
            chart_path = charts_dir / 'all_models_comparison.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"[OK] График сохранен: {chart_path}")

        except Exception as e:
            print(f"[WARNING] Не удалось создать график: {e}")

    def generate_performance_report(self):
        """Генерация отчета о производительности."""
        print(f"\n" + "=" * 70)
        print("ГЕНЕРАЦИЯ ОТЧЕТА О ПРОИЗВОДИТЕЛЬНОСТИ")
        print("=" * 70)

        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device_info': self.device_info,
            'benchmark_results': self.results,
            'recommendations': []
        }

        # Анализ результатов
        if 'neural_network' in self.results:
            nn_results = self.results['neural_network']

            # Анализ для batch size 100
            if 100 in nn_results['torch'] and 100 in nn_results['onnx']:
                torch_100 = nn_results['torch'][100]
                onnx_100 = nn_results['onnx'][100]

                speedup_100 = torch_100['time_per_sample_ms'] / onnx_100['time_per_sample_ms']

                report['recommendations'].append({
                    'scenario': 'Online Inference (batch=100)',
                    'torch_latency': f"{torch_100['time_per_sample_ms']:.3f} ms/sample",
                    'onnx_latency': f"{onnx_100['time_per_sample_ms']:.3f} ms/sample",
                    'speedup': f"{speedup_100:.2f}x",
                    'recommendation': 'Используйте ONNX для онлайн-инференса' if speedup_100 > 1.2 else 'Разница незначительна'
                })

        if 'all_onnx' in self.results:
            all_results = self.results['all_onnx']

            # Находим лучшую модель по latency
            best_model = min(all_results.items(), key=lambda x: x[1]['latency_per_sample_ms'])
            worst_model = max(all_results.items(), key=lambda x: x[1]['latency_per_sample_ms'])

            report['recommendations'].append({
                'scenario': 'Best ONNX Model',
                'best_model': best_model[0],
                'best_latency': f"{best_model[1]['latency_per_sample_ms']:.3f} ms/sample",
                'worst_model': worst_model[0],
                'worst_latency': f"{worst_model[1]['latency_per_sample_ms']:.3f} ms/sample",
                'recommendation': f'Для минимальной задержки используйте {best_model[0]}'
            })

        # Сохранение отчета
        report_dir = Path("models/artifacts/deep_benchmark")
        report_dir.mkdir(parents=True, exist_ok=True)

        json_path = report_dir / 'performance_report.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # Текстовый отчет
        txt_path = report_dir / 'performance_summary.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ: PyTorch vs ONNX\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Дата: {report['timestamp']}\n")
            f.write(f"Устройство: {self.device_info['cpu']['cores']} CPU cores\n\n")

            f.write("РЕКОМЕНДАЦИИ ПО ПРОИЗВОДИТЕЛЬНОСТИ:\n")
            f.write("-" * 50 + "\n")

            for rec in report['recommendations']:
                f.write(f"\n• {rec['scenario']}:\n")
                for key, value in rec.items():
                    if key != 'scenario':
                        f.write(f"  {key}: {value}\n")

        print(f"[OK] Отчеты сохранены:")
        print(f"  • JSON: {json_path}")
        print(f"  • TXT: {txt_path}")

        # Вывод краткой сводки
        print(f"\n" + "=" * 70)
        print("ИТОГИ БЕНЧМАРКА ПРОИЗВОДИТЕЛЬНОСТИ")
        print("=" * 70)

        for rec in report['recommendations']:
            print(f"\n• {rec['scenario']}: {rec.get('recommendation', '')}")

        return report

    def run(self):
        """Запуск полного бенчмарка."""
        print("=" * 70)
        print("ЗАПУСК УГЛУБЛЕННОГО БЕНЧМАРКА ПРОИЗВОДИТЕЛЬНОСТИ")
        print("=" * 70)

        # 1. Информация об устройстве
        self.print_device_info()

        # 2. Бенчмарк нейронной сети
        print(f"\n" + "=" * 70)
        print("ЭТАП 1: БЕНЧМАРК НЕЙРОННОЙ СЕТИ")
        nn_results = self.benchmark_neural_network_cpu()

        # 3. Бенчмарк всех ONNX моделей
        print(f"\n" + "=" * 70)
        print("ЭТАП 2: БЕНЧМАРК ВСЕХ ONNX МОДЕЛЕЙ")
        all_results = self.benchmark_all_onnx_models()

        # 4. Генерация отчета
        print(f"\n" + "=" * 70)
        print("ЭТАП 3: ГЕНЕРАЦИЯ ОТЧЕТА")
        report = self.generate_performance_report()

        print(f"\n" + "=" * 70)
        print("БЕНЧМАРК ЗАВЕРШЕН УСПЕШНО!")
        print("=" * 70)

        print(f"\n[INFO] Все отчеты сохранены в models/artifacts/deep_benchmark/")
        print(f"[NEXT] Следующие шаги:")
        print(f"  1. Проверьте корректность конвертации")
        print(f"  2. Запустите нагрузочное тестирование: python scripts/performance/load_testing.py")

        return report


def main():
    """Основная функция."""
    try:
        benchmark = DeepPerformanceBenchmark()
        report = benchmark.run()

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()