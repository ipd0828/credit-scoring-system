#!/usr/bin/env python3
"""
Углубленный бенчмарк производительности моделей на CPU/GPU.
Сравнение PyTorch vs ONNX Runtime на разных инстансах.

scripts/performance/deep_benchmark.py
"""

import numpy as np
import pandas as pd
import time
import json
import psutil
from pathlib import Path
import joblib
import onnxruntime as ort
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


class DeepPerformanceBenchmark:
    """Углубленный бенчмарк производительности на CPU/GPU."""

    def __init__(self):
        self.results = {}
        self.device_info = self._get_device_info()

    def _get_device_info(self):
        """Получение информации об устройствах."""
        info = {
            'cpu': {},
            'gpu': {},
            'memory': {}
        }

        # CPU информация
        info['cpu']['cores'] = psutil.cpu_count(logical=False)
        info['cpu']['threads'] = psutil.cpu_count(logical=True)
        info['cpu']['frequency'] = psutil.cpu_freq().current if psutil.cpu_freq() else 'N/A'

        # GPU информация (если доступно)
        try:
            if torch.cuda.is_available():
                info['gpu']['available'] = True
                info['gpu']['count'] = torch.cuda.device_count()
                info['gpu']['devices'] = []

                for i in range(torch.cuda.device_count()):
                    device_name = torch.cuda.get_device_name(i)
                    device_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3  # GB
                    info['gpu']['devices'].append({
                        'name': device_name,
                        'memory_gb': device_memory,
                        'capability': torch.cuda.get_device_capability(i)
                    })
            else:
                info['gpu']['available'] = False
        except:
            info['gpu']['available'] = False

        # Память
        memory = psutil.virtual_memory()
        info['memory']['total_gb'] = memory.total / 1024 ** 3
        info['memory']['available_gb'] = memory.available / 1024 ** 3
        info['memory']['percent'] = memory.percent

        return info

    def print_device_info(self):
        """Вывод информации об устройствах."""
        print("=" * 70)
        print("ИНФОРМАЦИЯ ОБ УСТРОЙСТВАХ ДЛЯ БЕНЧМАРКА")
        print("=" * 70)

        print(f"\n[CPU] Процессор:")
        print(
            f"  • Ядра: {self.device_info['cpu']['cores']} физических, {self.device_info['cpu']['threads']} логических")
        print(f"  • Частота: {self.device_info['cpu']['frequency']} MHz")

        print(f"\n[GPU] Графические процессоры:")
        if self.device_info['gpu']['available']:
            print(f"  • Доступно: Да ({self.device_info['gpu']['count']} устройств)")
            for i, gpu in enumerate(self.device_info['gpu']['devices']):
                print(f"    Device {i}: {gpu['name']}")
                print(f"      Память: {gpu['memory_gb']:.1f} GB")
                print(f"      Compute Capability: {gpu['capability'][0]}.{gpu['capability'][1]}")
        else:
            print(f"  • Доступно: Нет (только CPU)")

        print(f"\n[MEMORY] Память:")
        print(f"  • Всего: {self.device_info['memory']['total_gb']:.1f} GB")
        print(f"  • Доступно: {self.device_info['memory']['available_gb']:.1f} GB")
        print(f"  • Использовано: {self.device_info['memory']['percent']}%")

        print(f"\n[SOFTWARE] Версии ПО:")
        print(f"  • PyTorch: {torch.__version__}")
        print(f"  • ONNX Runtime: {ort.__version__}")
        print(f"  • NumPy: {np.__version__}")

    def benchmark_neural_network_cpu_vs_onnx(self, num_samples=10000):
        """Бенчмарк нейронной сети: PyTorch CPU vs ONNX CPU."""
        print(f"\n" + "=" * 70)
        print("БЕНЧМАРК: НЕЙРОННАЯ СЕТЬ (PyTorch CPU vs ONNX CPU)")
        print("=" * 70)

        try:
            # Загрузка моделей
            torch_path = Path("models/trained_final/neural_network.pth")
            onnx_path = Path("models/onnx_final/neural_network.onnx")
            scaler_path = Path("models/trained_final/nn_scaler.pkl")

            if not torch_path.exists() or not onnx_path.exists():
                print("[ERROR] Модели не найдены!")
                return None

            # Загрузка и подготовка данных
            print("[LOAD] Подготовка данных...")
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                # Создаем синтетические данные в формате оригинальных данных
                X_synth = np.random.randn(num_samples, 6).astype(np.float32)
                X_scaled = scaler.transform(X_synth)
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
            ort_session_cpu = ort.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )
            input_name = ort_session_cpu.get_inputs()[0].name

            # Тестирование разных batch sizes
            batch_sizes = [1, 10, 50, 100, 500, 1000, 5000]
            results = {'torch_cpu': {}, 'onnx_cpu': {}}

            for batch_size in batch_sizes:
                if batch_size > num_samples:
                    continue

                print(f"\n[BATCH] Размер батча: {batch_size}")
                batch_data = X_scaled[:batch_size]
                batch_tensor = torch.FloatTensor(batch_data)

                # PyTorch CPU
                torch_times = []
                torch_memory_start = psutil.Process().memory_info().rss / 1024 ** 2

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

                torch_memory_end = psutil.Process().memory_info().rss / 1024 ** 2
                torch_memory_used = torch_memory_end - torch_memory_start

                torch_avg = np.mean(torch_times)
                torch_std = np.std(torch_times)

                results['torch_cpu'][batch_size] = {
                    'avg_time_ms': torch_avg * 1000,
                    'std_time_ms': torch_std * 1000,
                    'time_per_sample_ms': torch_avg / batch_size * 1000,
                    'samples_per_second': batch_size / torch_avg,
                    'memory_mb': torch_memory_used,
                    'iterations': 10
                }

                # ONNX CPU
                onnx_times = []
                onnx_memory_start = psutil.Process().memory_info().rss / 1024 ** 2

                # Прогрев
                _ = ort_session_cpu.run(None, {input_name: batch_data})

                # Измерение
                for i in range(10):
                    start = time.perf_counter()
                    _ = ort_session_cpu.run(None, {input_name: batch_data})
                    end = time.perf_counter()
                    onnx_times.append(end - start)

                onnx_memory_end = psutil.Process().memory_info().rss / 1024 ** 2
                onnx_memory_used = onnx_memory_end - onnx_memory_start

                onnx_avg = np.mean(onnx_times)
                onnx_std = np.std(onnx_times)

                results['onnx_cpu'][batch_size] = {
                    'avg_time_ms': onnx_avg * 1000,
                    'std_time_ms': onnx_std * 1000,
                    'time_per_sample_ms': onnx_avg / batch_size * 1000,
                    'samples_per_second': batch_size / onnx_avg,
                    'memory_mb': onnx_memory_used,
                    'iterations': 10
                }

                # Сравнение
                speedup = torch_avg / onnx_avg if onnx_avg > 0 else 0
                memory_saving = torch_memory_used / onnx_memory_used if onnx_memory_used > 0 else 0

                print(f"  PyTorch CPU: {torch_avg / batch_size * 1000:6.3f} ms/sample")
                print(f"  ONNX CPU:    {onnx_avg / batch_size * 1000:6.3f} ms/sample")
                print(f"  Ускорение:   {speedup:.2f}x")
                print(f"  Память:      {torch_memory_used:.1f} MB vs {onnx_memory_used:.1f} MB")

            # Сохранение результатов
            self.results['neural_network_comparison'] = results

            # Визуализация
            self._plot_comparison_chart(results, "Neural Network: PyTorch vs ONNX (CPU)")

            return results

        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return None

    def benchmark_gpu_if_available(self):
        """Бенчмарк на GPU если доступно."""
        if not self.device_info['gpu']['available']:
            print(f"\n[INFO] GPU недоступно, пропускаем GPU бенчмарк")
            return None

        print(f"\n" + "=" * 70)
        print("БЕНЧМАРК: GPU ПРОИЗВОДИТЕЛЬНОСТЬ (если доступно)")
        print("=" * 70)

        try:
            # Загрузка ONNX модели для GPU
            onnx_path = Path("models/onnx_final/neural_network.onnx")
            if not onnx_path.exists():
                print("[ERROR] ONNX модель не найдена!")
                return None

            # Попробуем запустить на CUDA если доступно
            try:
                ort_session_gpu = ort.InferenceSession(
                    str(onnx_path),
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                print(f"[INFO] ONNX Runtime с поддержкой CUDA")
            except:
                print(f"[WARNING] CUDA не доступна в ONNX Runtime")
                return None

            # PyTorch GPU
            print("[TORCH] PyTorch GPU...")

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

            torch_model_gpu = SimpleNN(6).cuda()
            torch_path = Path("models/trained_final/neural_network.pth")
            if torch_path.exists():
                torch_model_gpu.load_state_dict(
                    torch.load(torch_path, map_location=torch.device('cuda'))
                )
            torch_model_gpu.eval()

            # Подготовка данных
            batch_size = 1000
            X_cpu = np.random.randn(batch_size, 6).astype(np.float32)
            X_gpu = torch.FloatTensor(X_cpu).cuda()

            # Тестирование
            results = {}

            # PyTorch GPU
            torch.cuda.synchronize()
            torch_times = []

            for i in range(10):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = torch_model_gpu(X_gpu)
                torch.cuda.synchronize()
                end = time.perf_counter()
                torch_times.append(end - start)

            torch_avg = np.mean(torch_times)
            results['torch_gpu'] = {
                'avg_time_ms': torch_avg * 1000,
                'time_per_sample_ms': torch_avg / batch_size * 1000,
                'samples_per_second': batch_size / torch_avg
            }

            # ONNX GPU
            input_name = ort_session_gpu.get_inputs()[0].name
            onnx_times = []

            for i in range(10):
                start = time.perf_counter()
                _ = ort_session_gpu.run(None, {input_name: X_cpu})
                end = time.perf_counter()
                onnx_times.append(end - start)

            onnx_avg = np.mean(onnx_times)
            results['onnx_gpu'] = {
                'avg_time_ms': onnx_avg * 1000,
                'time_per_sample_ms': onnx_avg / batch_size * 1000,
                'samples_per_second': batch_size / onnx_avg
            }

            print(f"\n[RESULTS] GPU Производительность (batch_size={batch_size}):")
            print(f"  PyTorch GPU: {torch_avg / batch_size * 1000:.3f} ms/sample")
            print(f"  ONNX GPU:    {onnx_avg / batch_size * 1000:.3f} ms/sample")
            print(f"  Ускорение:   {torch_avg / onnx_avg:.2f}x")

            self.results['gpu_comparison'] = results

            return results

        except Exception as e:
            print(f"[ERROR] GPU бенчмарк не удался: {e}")
            return None

    def _plot_comparison_chart(self, results, title):
        """Создание графика сравнения."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Данные
            batch_sizes = sorted(list(results['torch_cpu'].keys()))
            torch_times = [results['torch_cpu'][bs]['time_per_sample_ms'] for bs in batch_sizes]
            onnx_times = [results['onnx_cpu'][bs]['time_per_sample_ms'] for bs in batch_sizes]
            torch_speed = [results['torch_cpu'][bs]['samples_per_second'] for bs in batch_sizes]
            onnx_speed = [results['onnx_cpu'][bs]['samples_per_second'] for bs in batch_sizes]

            # График 1: Latency
            ax1.plot(batch_sizes, torch_times, 'o-', label='PyTorch CPU', linewidth=2, markersize=8)
            ax1.plot(batch_sizes, onnx_times, 's-', label='ONNX CPU', linewidth=2, markersize=8)
            ax1.set_xlabel('Batch Size', fontsize=12)
            ax1.set_ylabel('Latency per Sample (ms)', fontsize=12)
            ax1.set_title('Latency Comparison', fontsize=14, pad=20)
            ax1.set_xscale('log')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(fontsize=11)

            # Добавляем значения на точки
            for i, (bs, torch_t, onnx_t) in enumerate(zip(batch_sizes, torch_times, onnx_times)):
                if i % 2 == 0:  # Каждую вторую точку
                    ax1.text(bs, torch_t + 0.01, f'{torch_t:.2f}', ha='center', va='bottom', fontsize=9)
                    ax1.text(bs, onnx_t - 0.02, f'{onnx_t:.2f}', ha='center', va='top', fontsize=9)

            # График 2: Throughput
            ax2.plot(batch_sizes, torch_speed, 'o-', label='PyTorch CPU', linewidth=2, markersize=8)
            ax2.plot(batch_sizes, onnx_speed, 's-', label='ONNX CPU', linewidth=2, markersize=8)
            ax2.set_xlabel('Batch Size', fontsize=12)
            ax2.set_ylabel('Throughput (samples/sec)', fontsize=12)
            ax2.set_title('Throughput Comparison', fontsize=14, pad=20)
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(fontsize=11)

            # Общий заголовок
            plt.suptitle(title, fontsize=16, y=1.05)
            plt.tight_layout()

            # Сохранение
            charts_dir = Path("models/artifacts/deep_benchmark")
            charts_dir.mkdir(parents=True, exist_ok=True)

            chart_path = charts_dir / 'pytorch_vs_onnx_performance.png'
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
            'benchmarks': self.results,
            'recommendations': []
        }

        # Анализ результатов
        if 'neural_network_comparison' in self.results:
            nn_results = self.results['neural_network_comparison']

            # Анализ для batch size 100 (типичный для онлайн-инференса)
            if 100 in nn_results['torch_cpu'] and 100 in nn_results['onnx_cpu']:
                torch_100 = nn_results['torch_cpu'][100]
                onnx_100 = nn_results['onnx_cpu'][100]

                speedup_100 = torch_100['time_per_sample_ms'] / onnx_100['time_per_sample_ms']

                report['recommendations'].append({
                    'priority': 'high',
                    'scenario': 'Online Inference (batch=100)',
                    'torch_latency': f"{torch_100['time_per_sample_ms']:.3f} ms",
                    'onnx_latency': f"{onnx_100['time_per_sample_ms']:.3f} ms",
                    'speedup': f"{speedup_100:.2f}x",
                    'recommendation': 'Используйте ONNX для онлайн-инференса' if speedup_100 > 1.2 else 'Разница незначительна'
                })

            # Анализ для batch size 1000 (типичный для офлайн-обработки)
            if 1000 in nn_results['torch_cpu'] and 1000 in nn_results['onnx_cpu']:
                torch_1000 = nn_results['torch_cpu'][1000]
                onnx_1000 = nn_results['onnx_cpu'][1000]

                speedup_1000 = torch_1000['time_per_sample_ms'] / onnx_1000['time_per_sample_ms']

                report['recommendations'].append({
                    'priority': 'medium',
                    'scenario': 'Batch Processing (batch=1000)',
                    'torch_latency': f"{torch_1000['time_per_sample_ms']:.3f} ms",
                    'onnx_latency': f"{onnx_1000['time_per_sample_ms']:.3f} ms",
                    'speedup': f"{speedup_1000:.2f}x",
                    'recommendation': 'ONNX эффективен для пакетной обработки' if speedup_1000 > 1.5 else 'Рассмотрите PyTorch для пакетной обработки'
                })

        if 'gpu_comparison' in self.results:
            gpu_results = self.results['gpu_comparison']
            report['recommendations'].append({
                'priority': 'info',
                'scenario': 'GPU Processing',
                'note': 'GPU доступно, но требует дополнительной настройки для production',
                'recommendation': 'Для продакшена используйте CPU версию для стабильности'
            })

        # Общие рекомендации
        report['recommendations'].append({
            'priority': 'high',
            'scenario': 'Production Deployment',
            'recommendation': 'Используйте ONNX Runtime с CPUExecutionProvider для максимальной стабильности'
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
            f.write(f"Устройства: {self.device_info['cpu']['cores']} CPU cores")
            if self.device_info['gpu']['available']:
                f.write(f", {self.device_info['gpu']['count']} GPU(s)")
            f.write("\n\n")

            f.write("РЕКОМЕНДАЦИИ ПО ПРОИЗВОДИТЕЛЬНОСТИ:\n")
            f.write("-" * 50 + "\n")

            for rec in report['recommendations']:
                priority = rec['priority'].upper()
                f.write(f"\n[{priority}] {rec['scenario']}:\n")

                for key, value in rec.items():
                    if key not in ['priority', 'scenario']:
                        f.write(f"  • {key.replace('_', ' ').title()}: {value}\n")

        print(f"[OK] Отчеты сохранены:")
        print(f"  • JSON: {json_path}")
        print(f"  • TXT: {txt_path}")

        # Вывод краткой сводки
        print(f"\n" + "=" * 70)
        print("ИТОГИ БЕНЧМАРКА ПРОИЗВОДИТЕЛЬНОСТИ")
        print("=" * 70)

        for rec in report['recommendations']:
            if rec['priority'] == 'high':
                print(f"\n {rec['scenario']}: {rec['recommendation']}")

        return report

    def run(self):
        """Запуск полного бенчмарка."""
        print("=" * 70)
        print("ЗАПУСК УГЛУБЛЕННОГО БЕНЧМАРКА ПРОИЗВОДИТЕЛЬНОСТИ")
        print("=" * 70)

        # 1. Информация об устройствах
        self.print_device_info()

        # 2. Бенчмарк CPU: PyTorch vs ONNX
        print(f"\n" + "=" * 70)
        print("ЭТАП 1: CPU ПРОИЗВОДИТЕЛЬНОСТЬ (НЕЙРОННАЯ СЕТЬ)")
        cpu_results = self.benchmark_neural_network_cpu_vs_onnx()

        # 3. Бенчмарк GPU (если доступно)
        print(f"\n" + "=" * 70)
        print("ЭТАП 2: GPU ПРОИЗВОДИТЕЛЬНОСТЬ (ЕСЛИ ДОСТУПНО)")
        gpu_results = self.benchmark_gpu_if_available()

        # 4. Генерация отчета
        print(f"\n" + "=" * 70)
        print("ЭТАП 3: ГЕНЕРАЦИЯ ОТЧЕТА")
        report = self.generate_performance_report()

        print(f"\n" + "=" * 70)
        print("БЕНЧМАРК ЗАВЕРШЕН УСПЕШНО!")
        print("=" * 70)

        return report


def main():
    """Основная функция."""
    try:
        benchmark = DeepPerformanceBenchmark()
        report = benchmark.run()

        print(f"\n[INFO] Все отчеты сохранены в models/artifacts/deep_benchmark/")
        print(f"[NEXT] Следующие шаги:")
        print(f"  1. Проверьте корректность конвертации: python scripts/validation/validate_onnx_conversion.py")
        print(f"  2. Запустите нагрузочное тестирование: python scripts/performance/load_testing.py")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()