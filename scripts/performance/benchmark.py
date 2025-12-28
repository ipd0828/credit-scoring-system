#!/usr/bin/env python3
"""
Бенчмарк с правильными метриками для кредитного скоринга (ROC-AUC/AUC).
scripts/performance/benchmark.py
"""

import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import time
import json
import psutil
from pathlib import Path
import joblib
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report


class CreditScoringBenchmark:
    """Бенчмарк с правильными метриками для кредитного скоринга."""

    def __init__(self):
        self.results = {}
        self.model_paths = self._find_all_models()

    def _find_all_models(self):
        """Поиск всех моделей в проекте."""
        print("=" * 60)
        print("ПОИСК МОДЕЛЕЙ В ПРОЕКТЕ")
        print("=" * 60)

        model_paths = {
            'onnx': [],
            'pytorch': [],
            'sklearn': [],
            'optimized': []
        }

        # Поиск ONNX моделей
        search_paths = [
            Path("models/onnx_final"),  # Новая папка с ONNX моделями
            Path("models/onnx"),
            Path("models/optimized"),
            Path("models/nn_models"),
            Path("models/trained_final")  # Новая папка с обученными моделями
        ]

        for search_path in search_paths:
            if search_path.exists():
                print(f"\n[DIR] Проверяем: {search_path}")

                # Ищем ONNX
                onnx_files = list(search_path.glob("*.onnx"))
                for f in onnx_files:
                    model_paths['onnx'].append(f)
                    print(f"  [OK] ONNX: {f.name} ({f.stat().st_size / 1024:.1f} KB)")

                # Ищем PyTorch
                pytorch_files = list(search_path.glob("*.pth"))
                for f in pytorch_files:
                    model_paths['pytorch'].append(f)
                    print(f"  [OK] PyTorch: {f.name} ({f.stat().st_size / 1024:.1f} KB)")

                # Ищем sklearn
                sklearn_files = list(search_path.glob("*.pkl"))
                for f in sklearn_files:
                    # Проверяем, что это не scaler или другие файлы
                    if 'scaler' not in f.stem.lower() and 'threshold' not in f.stem.lower():
                        model_paths['sklearn'].append(f)
                        print(f"  [OK] Sklearn: {f.name} ({f.stat().st_size / 1024:.1f} KB)")

        # Сводка
        print(f"\n[INFO] НАЙДЕНО МОДЕЛЕЙ:")
        print(f"  ONNX: {len(model_paths['onnx'])}")
        print(f"  PyTorch: {len(model_paths['pytorch'])}")
        print(f"  Sklearn: {len(model_paths['sklearn'])}")

        if len(model_paths['onnx']) == 0:
            print(f"\n[WARNING] ONNX модели не найдены!")
            print(f"   Запустите сначала: python scripts/model_training/convert_models_to_onnx.py")

        return model_paths

    def load_test_data(self, num_samples: int = 5000):
        """Загрузка тестовых данных."""
        print(f"\n[LOAD] Загрузка тестовых данных...")

        try:
            data_path = Path("data/processed_fixed")
            if not data_path.exists():
                print(f"[ERROR] Папка с данными не найдена: {data_path}")
                return None, None

            X_test = pd.read_csv(data_path / "X_test.csv")
            y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

            # Анализ распределения классов
            class_distribution = y_test.value_counts(normalize=True)
            print(f"  [INFO] Распределение классов:")
            print(f"    • Класс 0 (погасили): {class_distribution.get(0, 0):.1%}")
            print(f"    • Класс 1 (дефолт): {class_distribution.get(1, 0):.1%}")
            print(
                f"    • Imbalance ratio: {class_distribution.get(1, 0.01) / max(class_distribution.get(0, 0.99), 0.01):.2f}")

            # Ограничиваем количество samples
            if num_samples < len(X_test):
                X_test = X_test.iloc[:num_samples]
                y_test = y_test.iloc[:num_samples]

            print(f"  [OK] Загружено: {X_test.shape[0]} samples, {X_test.shape[1]} features")

            return X_test.values.astype(np.float32), y_test.values

        except Exception as e:
            print(f"[ERROR] Ошибка загрузки данных: {e}")
            # Создаем синтетические данные с реалистичным распределением
            print("  [WARNING] Создаю синтетические данные для теста...")
            # 20% дефолтов (класс 1), 80% хороших заемщиков (класс 0)
            y_synthetic = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])
            X_synthetic = np.random.randn(num_samples, 6).astype(np.float32)
            # Немного коррелируем признаки с целевой переменной
            X_synthetic[:, 0] += y_synthetic * 0.5  # limit_bal коррелирует с дефолтом
            X_synthetic[:, 5] += y_synthetic * 0.3  # pay_new коррелирует с дефолтом

            print(f"  [INFO] Синтетические данные: {np.mean(y_synthetic):.1%} дефолтов")
            return X_synthetic, y_synthetic

    def calculate_metrics(self, y_true, y_pred_proba, threshold=0.5):
        """Расчет всех метрик для кредитного скоринга."""
        # Бинарные предсказания
        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {}

        # Основные метрики
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = 0.5

        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # Дополнительные метрики для несбалансированных данных
        from sklearn.metrics import balanced_accuracy_score, average_precision_score
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)

        # Коэффициент корреляции прогноза с реальностью
        from scipy.stats import pointbiserialr
        try:
            corr, _ = pointbiserialr(y_true, y_pred_proba)
            metrics['pointbiserial_corr'] = corr
        except:
            metrics['pointbiserial_corr'] = 0

        # Коэффициент детерминации для вероятностей
        from sklearn.metrics import r2_score
        try:
            metrics['r2_score'] = r2_score(y_true, y_pred_proba)
        except:
            metrics['r2_score'] = 0

        # Матрица ошибок
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negative'] = int(cm[0, 0])
        metrics['false_positive'] = int(cm[0, 1])
        metrics['false_negative'] = int(cm[1, 0])
        metrics['true_positive'] = int(cm[1, 1])

        # Производные метрики
        metrics['specificity'] = metrics['true_negative'] / max((metrics['true_negative'] + metrics['false_positive']),
                                                                1)
        metrics['false_positive_rate'] = metrics['false_positive'] / max(
            (metrics['false_positive'] + metrics['true_negative']), 1)
        metrics['false_negative_rate'] = metrics['false_negative'] / max(
            (metrics['false_negative'] + metrics['true_positive']), 1)

        # Метрики бизнес-ценности
        # Предположим: FP (ложное отклонение хорошего клиента) стоит 1000 у.е.
        # FN (ложное одобрение плохого клиента) стоит 5000 у.е.
        fp_cost = 1000
        fn_cost = 5000
        metrics['business_cost'] = metrics['false_positive'] * fp_cost + metrics['false_negative'] * fn_cost
        metrics['cost_per_sample'] = metrics['business_cost'] / len(y_true)

        return metrics

    def benchmark_onnx_model(self, onnx_path: Path, model_name: str = None):
        """Бенчмарк конкретной ONNX модели."""
        if model_name is None:
            model_name = onnx_path.stem

        print(f"\n" + "=" * 60)
        print(f"[BENCHMARK] ONNX: {model_name}")
        print("=" * 60)

        try:
            # Проверяем файл
            if not onnx_path.exists():
                print(f"[ERROR] Файл не найден: {onnx_path}")
                return None

            print(f"[FILE] {onnx_path}")
            print(f"[SIZE] {onnx_path.stat().st_size / 1024:.1f} KB")

            # Загружаем ONNX модель
            session = ort.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )

            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            print(f"[INFO] Вход: {input_name}, Выход: {output_name}")
            print(f"[INFO] Формат входа: {session.get_inputs()[0].shape}")

            # Загружаем тестовые данные
            X_test, y_test = self.load_test_data(5000)
            if X_test is None:
                return None

            results = {
                'model_name': model_name,
                'file_path': str(onnx_path),
                'file_size_kb': onnx_path.stat().st_size / 1024,
                'input_shape': session.get_inputs()[0].shape,
                'performance_metrics': {},
                'classification_metrics': {}
            }

            # Тестируем разные batch sizes для производительности
            batch_sizes = [1, 10, 100, 500, 1000]
            batch_results = {}

            for batch_size in batch_sizes:
                if batch_size > len(X_test):
                    print(f"  [SKIP] Пропускаем batch {batch_size} (больше доступных данных)")
                    continue

                batch_data = X_test[:batch_size]

                # Прогрев
                _ = session.run([output_name], {input_name: batch_data})

                # Измерение времени
                times = []
                num_iterations = max(5, 50 // batch_size)  # Адаптивное количество итераций

                for i in range(num_iterations):
                    start_time = time.perf_counter()
                    _ = session.run([output_name], {input_name: batch_data})
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)

                # Статистика производительности
                avg_time = np.mean(times)
                std_time = np.std(times)
                time_per_sample = avg_time / batch_size * 1000  # мс
                samples_per_second = batch_size / avg_time

                batch_results[batch_size] = {
                    'avg_time_ms': avg_time * 1000,
                    'std_time_ms': std_time * 1000,
                    'time_per_sample_ms': time_per_sample,
                    'samples_per_second': samples_per_second,
                    'iterations': num_iterations
                }

                print(f"  [BATCH] {batch_size:4d}: "
                      f"{time_per_sample:6.2f} мс/sample ± {std_time / batch_size * 1000:4.2f}, "
                      f"{samples_per_second:6.0f} samples/sec")

            results['batch_results'] = batch_results

            # Полный inference для оценки качества
            print(f"\n[FULL] Полный inference для оценки качества ({len(X_test)} samples):")

            # Inference для всех данных
            start_time = time.perf_counter()
            predictions = session.run([output_name], {input_name: X_test})
            end_time = time.perf_counter()

            full_time = end_time - start_time
            predictions_array = predictions[0].flatten()

            # Если выход имеет форму (n, 1), преобразуем в (n,)
            if len(predictions_array.shape) == 2:
                predictions_array = predictions_array.reshape(-1)

            # Если выход имеет несколько измерений, берем вероятности класса 1
            if len(predictions_array.shape) > 1 and predictions_array.shape[1] > 1:
                predictions_array = predictions_array[:, 1]  # Вероятность класса 1
            elif np.all(predictions_array <= 1) and np.all(predictions_array >= 0):
                # Это вероятности
                pass
            else:
                # Это логиты или бинарные предсказания
                from scipy.special import expit
                predictions_array = expit(predictions_array)

            # Рассчитываем все метрики
            if y_test is not None:
                metrics = self.calculate_metrics(y_test, predictions_array)
                results['classification_metrics'] = metrics

                print(f"  [METRICS] Ключевые метрики:")
                print(f"    • ROC-AUC:     {metrics['roc_auc']:.4f} (основная метрика)")
                print(f"    • Accuracy:    {metrics['accuracy']:.4f}")
                print(f"    • Precision:   {metrics['precision']:.4f}")
                print(f"    • Recall:      {metrics['recall']:.4f}")
                print(f"    • F1-Score:    {metrics['f1']:.4f}")
                print(f"    • Bal. Acc:    {metrics['balanced_accuracy']:.4f}")
                print(f"    • Avg. Prec:   {metrics['average_precision']:.4f}")

                print(f"\n  [BUSINESS] Бизнес-метрики:")
                print(f"    • Точность (специфичность): {metrics['specificity']:.3f}")
                print(f"    • FPR:  {metrics['false_positive_rate']:.3f}")
                print(f"    • FNR:  {metrics['false_negative_rate']:.3f}")
                print(f"    • Общая стоимость ошибок: {metrics['business_cost']:,.0f} у.е.")
                print(f"    • Стоимость на sample: {metrics['cost_per_sample']:.1f} у.е.")

            results['performance_metrics'] = {
                'total_time_s': full_time,
                'samples_per_second': len(X_test) / full_time,
                'time_per_sample_ms': full_time / len(X_test) * 1000,
                'throughput_rps': len(X_test) / full_time
            }

            print(f"\n  [PERFORMANCE] Производительность:")
            print(f"    • Время:         {full_time:.3f} сек")
            print(f"    • Samples/sec:   {len(X_test) / full_time:.0f}")
            print(f"    • RPS:           {len(X_test) / full_time:.0f}")
            print(f"    • Latency:       {full_time / len(X_test) * 1000:.3f} мс")

            return results

        except Exception as e:
            print(f"[ERROR] Ошибка бенчмарка модели {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def benchmark_all_onnx_models(self):
        """Бенчмарк всех найденных ONNX моделей."""
        print(f"\n" + "=" * 60)
        print("[START] ЗАПУСК БЕНЧМАРКА ВСЕХ ONNX МОДЕЛЕЙ")
        print("=" * 60)

        all_results = {}

        if len(self.model_paths['onnx']) == 0:
            print(f"[WARNING] ONNX модели не найдены!")
            print(f"   Попробуйте создать их:")
            print(f"   1. python scripts/model_training/convert_models_to_onnx.py")
            return None

        for onnx_path in self.model_paths['onnx']:
            print(f"\n" + "-" * 40)
            results = self.benchmark_onnx_model(onnx_path)
            if results:
                all_results[results['model_name']] = results

        return all_results

    def create_comparison_charts(self, results: dict):
        """Создание графиков сравнения метрик."""
        print(f"\n" + "=" * 60)
        print("[CHART] СОЗДАНИЕ ГРАФИКОВ СРАВНЕНИЯ")
        print("=" * 60)

        if not results:
            print("[WARNING] Нет данных для визуализации")
            return

        # Создаем директорию для графиков
        charts_dir = Path("models/artifacts/benchmark_charts")
        charts_dir.mkdir(parents=True, exist_ok=True)

        # Подготовка данных
        model_names = []
        roc_auc_scores = []
        f1_scores = []
        balanced_acc_scores = []
        throughput_scores = []
        business_costs = []

        for model_name, model_data in results.items():
            if 'classification_metrics' in model_data:
                metrics = model_data['classification_metrics']
                model_names.append(model_name)
                roc_auc_scores.append(metrics.get('roc_auc', 0))
                f1_scores.append(metrics.get('f1', 0))
                balanced_acc_scores.append(metrics.get('balanced_accuracy', 0))
                throughput_scores.append(model_data.get('performance_metrics', {}).get('throughput_rps', 0))
                business_costs.append(metrics.get('business_cost', 0))

        if len(model_names) < 2:
            print("[INFO] Недостаточно моделей для сравнения")
            return

        # 1. График ROC-AUC сравнения
        plt.figure(figsize=(12, 8))
        x = np.arange(len(model_names))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # ROC-AUC и F1
        ax1.bar(x - width / 2, roc_auc_scores, width, label='ROC-AUC', color='skyblue', edgecolor='black')
        ax1.bar(x + width / 2, f1_scores, width, label='F1-Score', color='lightcoral', edgecolor='black')
        ax1.set_xlabel('Модель')
        ax1.set_ylabel('Score')
        ax1.set_title('Сравнение ROC-AUC и F1-Score моделей', fontsize=14, pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim([0, 1])

        # Добавляем значения на столбцы
        for i, (roc, f1) in enumerate(zip(roc_auc_scores, f1_scores)):
            ax1.text(i - width / 2, roc + 0.01, f'{roc:.3f}', ha='center', va='bottom', fontsize=9)
            ax1.text(i + width / 2, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom', fontsize=9)

        # Throughput и Business Cost
        ax2_twin = ax2.twinx()

        bars1 = ax2.bar(x - width / 2, throughput_scores, width, label='Throughput (RPS)', color='lightgreen',
                        edgecolor='black')
        bars2 = ax2_twin.bar(x + width / 2, business_costs, width, label='Business Cost', color='gold',
                             edgecolor='black')

        ax2.set_xlabel('Модель')
        ax2.set_ylabel('Throughput (RPS)')
        ax2_twin.set_ylabel('Business Cost')
        ax2.set_title('Производительность и Бизнес-стоимость моделей', fontsize=14, pad=20)
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45, ha='right')

        # Объединяем легенды
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax2.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        chart_path = charts_dir / 'model_comparison_metrics.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] График сравнения метрик сохранен: {chart_path}")

        # 2. Radar chart для сравнения моделей
        self.create_radar_chart(results, charts_dir)

        # 3. Сводная таблица
        self.create_summary_table(results, charts_dir)

        print(f"\n[INFO] Все графики сохранены в: {charts_dir}")

    def create_radar_chart(self, results: dict, charts_dir: Path):
        """Создание radar chart для сравнения моделей."""
        try:
            from matplotlib.patches import Circle, RegularPolygon
            from matplotlib.path import Path as MPath
            from matplotlib.projections.polar import PolarAxes
            from matplotlib.projections import register_projection
            from matplotlib.spines import Spine
            from matplotlib.transforms import Affine2D

            # Выбираем ключевые метрики
            metrics = ['roc_auc', 'f1', 'balanced_accuracy', 'precision', 'recall', 'specificity']
            metric_labels = ['ROC-AUC', 'F1', 'Bal. Acc', 'Precision', 'Recall', 'Specificity']

            model_names = list(results.keys())
            n_models = len(model_names)
            n_metrics = len(metrics)

            if n_models < 2 or n_metrics < 3:
                return

            # Собираем данные
            data = []
            for model_name in model_names:
                model_metrics = results[model_name].get('classification_metrics', {})
                row = [model_metrics.get(m, 0) for m in metrics]
                data.append(row)

            # Нормализуем данные для radar chart (0-1)
            data_norm = np.array(data)

            # Создаем radar chart
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='polar')

            # Углы для каждой метрики
            angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
            angles += angles[:1]  # Замыкаем круг

            # Цвета для моделей
            colors = plt.cm.Set3(np.linspace(0, 1, n_models))

            for idx, (model_name, model_data) in enumerate(zip(model_names, data_norm)):
                values = model_data.tolist()
                values += values[:1]  # Замыкаем круг

                ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
                ax.fill(angles, values, alpha=0.1, color=colors[idx])

            # Настраиваем оси
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels, fontsize=11)
            ax.set_ylim(0, 1)
            ax.set_yticklabels([])
            ax.grid(True, alpha=0.3)

            plt.title('Radar Chart: Сравнение моделей по метрикам', fontsize=14, pad=20)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            plt.tight_layout()

            radar_path = charts_dir / 'radar_chart_comparison.png'
            plt.savefig(radar_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OK] Radar chart сохранен: {radar_path}")

        except Exception as e:
            print(f"[WARNING] Не удалось создать radar chart: {e}")

    def create_summary_table(self, results: dict, charts_dir: Path):
        """Создание сводной таблицы."""
        summary_data = []

        for model_name, model_results in results.items():
            perf = model_results.get('performance_metrics', {})
            metrics = model_results.get('classification_metrics', {})

            summary_data.append({
                'Модель': model_name,
                'ROC-AUC': f"{metrics.get('roc_auc', 0):.4f}",
                'F1-Score': f"{metrics.get('f1', 0):.4f}",
                'Bal. Acc': f"{metrics.get('balanced_accuracy', 0):.4f}",
                'RPS': f"{perf.get('throughput_rps', 0):.0f}",
                'Latency (ms)': f"{perf.get('time_per_sample_ms', 0):.3f}",
                'Размер (KB)': f"{model_results.get('file_size_kb', 0):.1f}",
                'Стоимость (у.е.)': f"{metrics.get('business_cost', 0):,.0f}"
            })

        if summary_data:
            df = pd.DataFrame(summary_data)

            # Сохраняем как CSV
            csv_path = charts_dir / 'performance_summary.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"[OK] CSV таблица сохранена: {csv_path}")

            # Создаем красивую таблицу
            plt.figure(figsize=(14, len(summary_data) * 0.6 + 2))
            ax = plt.subplot(111, frame_on=False)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            cell_text = df.values
            col_labels = df.columns

            # Определяем цвета для ячеек
            cell_colors = []
            for row in cell_text:
                row_colors = []
                for i, cell in enumerate(row):
                    if i == 1:  # ROC-AUC колонка
                        try:
                            value = float(cell)
                            if value >= 0.8:
                                row_colors.append('#d4edda')  # зеленый
                            elif value >= 0.7:
                                row_colors.append('#fff3cd')  # желтый
                            else:
                                row_colors.append('#f8d7da')  # красный
                        except:
                            row_colors.append('white')
                    elif i == 4:  # RPS колонка
                        try:
                            value = float(cell)
                            if value >= 10000:
                                row_colors.append('#d4edda')  # зеленый
                            elif value >= 5000:
                                row_colors.append('#fff3cd')  # желтый
                            else:
                                row_colors.append('#f8d7da')  # красный
                        except:
                            row_colors.append('white')
                    else:
                        row_colors.append('white')
                cell_colors.append(row_colors)

            table = ax.table(cellText=cell_text,
                             colLabels=col_labels,
                             cellLoc='center',
                             loc='center',
                             colColours=['#f0f0f0'] * len(col_labels),
                             cellColours=cell_colors)

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2)

            plt.title('Сводная таблица производительности и качества моделей', fontsize=14, pad=20)
            table_path = charts_dir / 'performance_table.png'
            plt.savefig(table_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OK] Таблица сохранена: {table_path}")

    def generate_final_report(self, results: dict):
        """Генерация финального отчета для кредитного скоринга."""
        print(f"\n" + "=" * 60)
        print("[REPORT] ГЕНЕРАЦИЯ ФИНАЛЬНОГО ОТЧЕТА")
        print("=" * 60)

        report_dir = Path("models/artifacts/benchmark_reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        # Структура отчета
        report = {
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'models_tested': len(results),
            'results': {},
            'recommendations': [],
            'best_model_by_roc_auc': None,
            'best_model_by_business_cost': None,
            'best_model_by_speed': None
        }

        # Добавляем результаты
        for model_name, model_results in results.items():
            report['results'][model_name] = {
                'performance': model_results.get('performance_metrics', {}),
                'classification': model_results.get('classification_metrics', {}),
                'file_size_kb': model_results.get('file_size_kb', 0)
            }

        # Определяем лучшие модели по разным критериям
        if results:
            # 1. Лучшая по ROC-AUC
            best_roc_auc = -1
            best_model_roc_auc = None

            # 2. Лучшая по бизнес-стоимости
            best_business_cost = float('inf')
            best_model_cost = None

            # 3. Лучшая по скорости
            best_speed = -1
            best_model_speed = None

            for model_name, model_results in results.items():
                metrics = model_results.get('classification_metrics', {})
                perf = model_results.get('performance_metrics', {})

                # ROC-AUC
                roc_auc = metrics.get('roc_auc', 0)
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    best_model_roc_auc = model_name

                # Бизнес-стоимость
                business_cost = metrics.get('business_cost', float('inf'))
                if business_cost < best_business_cost:
                    best_business_cost = business_cost
                    best_model_cost = model_name

                # Скорость
                speed = perf.get('throughput_rps', 0)
                if speed > best_speed:
                    best_speed = speed
                    best_model_speed = model_name

            report['best_model_by_roc_auc'] = best_model_roc_auc
            report['best_model_by_business_cost'] = best_model_cost
            report['best_model_by_speed'] = best_model_speed

            # Формируем рекомендации
            report['recommendations'] = [
                {
                    'priority': 'high',
                    'title': 'Для максимального качества (ROC-AUC)',
                    'description': f'Используйте "{best_model_roc_auc}" (ROC-AUC: {best_roc_auc:.4f})'
                },
                {
                    'priority': 'high',
                    'title': 'Для минимальной бизнес-стоимости',
                    'description': f'Используйте "{best_model_cost}" (стоимость: {best_business_cost:,.0f} у.е.)'
                },
                {
                    'priority': 'medium',
                    'title': 'Для максимальной производительности',
                    'description': f'Используйте "{best_model_speed}" (RPS: {best_speed:.0f})'
                },
                {
                    'priority': 'medium',
                    'title': 'Порог классификации',
                    'description': 'Оптимизируйте порог классификации (по умолчанию 0.5) для баланса precision/recall'
                }
            ]

        # Сохраняем JSON отчет
        json_path = report_dir / 'final_benchmark_report.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # Создаем текстовый отчет
        txt_path = report_dir / 'final_benchmark_summary.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ФИНАЛЬНЫЙ ОТЧЕТ ПО БЕНЧМАРКУ МОДЕЛЕЙ (Кредитный скоринг)\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Дата генерации: {report['generated_at']}\n")
            f.write(f"Протестировано моделей: {report['models_tested']}\n\n")

            # Вывод лучших моделей
            f.write("[BEST] ЛУЧШИЕ МОДЕЛИ ПО РАЗНЫМ КРИТЕРИЯМ:\n")
            f.write("-" * 50 + "\n")

            if report['best_model_by_roc_auc']:
                best_model = results[report['best_model_by_roc_auc']]
                metrics = best_model.get('classification_metrics', {})
                f.write(f"\n1. По ROC-AUC (качество): {report['best_model_by_roc_auc']}\n")
                f.write(f"   • ROC-AUC: {metrics.get('roc_auc', 0):.4f}\n")
                f.write(f"   • F1-Score: {metrics.get('f1', 0):.4f}\n")
                f.write(f"   • Precision: {metrics.get('precision', 0):.4f}\n")
                f.write(f"   • Recall: {metrics.get('recall', 0):.4f}\n")

            if report['best_model_by_business_cost']:
                best_model = results[report['best_model_by_business_cost']]
                metrics = best_model.get('classification_metrics', {})
                f.write(f"\n2. По бизнес-стоимости: {report['best_model_by_business_cost']}\n")
                f.write(f"   • Общая стоимость: {metrics.get('business_cost', 0):,.0f} у.е.\n")
                f.write(f"   • Стоимость на sample: {metrics.get('cost_per_sample', 0):.1f} у.е.\n")
                f.write(f"   • FPR: {metrics.get('false_positive_rate', 0):.3f}\n")
                f.write(f"   • FNR: {metrics.get('false_negative_rate', 0):.3f}\n")

            if report['best_model_by_speed']:
                best_model = results[report['best_model_by_speed']]
                perf = best_model.get('performance_metrics', {})
                f.write(f"\n3. По производительности: {report['best_model_by_speed']}\n")
                f.write(f"   • RPS: {perf.get('throughput_rps', 0):.0f}\n")
                f.write(f"   • Latency: {perf.get('time_per_sample_ms', 0):.3f} мс\n")
                f.write(f"   • Размер: {best_model.get('file_size_kb', 0):.1f} KB\n")

            f.write("\n[INFO] ДЕТАЛЬНЫЕ РЕКОМЕНДАЦИИ:\n")
            f.write("-" * 50 + "\n")

            for rec in report['recommendations']:
                priority_icon = "[HIGH]" if rec['priority'] == 'high' else "[MEDIUM]" if rec[
                                                                                             'priority'] == 'medium' else "[INFO]"
                f.write(f"\n{priority_icon} {rec['title']}\n")
                f.write(f"   {rec['description']}\n")

        print(f"[OK] Отчеты сохранены:")
        print(f"   [JSON] {json_path}")
        print(f"   [TXT] {txt_path}")

        # Вывод краткой сводки в консоль
        print(f"\n" + "=" * 60)
        print("[BEST] ИТОГИ БЕНЧМАРКА (Кредитный скоринг)")
        print("=" * 60)

        if report['best_model_by_roc_auc']:
            best = results[report['best_model_by_roc_auc']]
            metrics = best.get('classification_metrics', {})
            print(f"\n Лучшая по ROC-AUC: {report['best_model_by_roc_auc']}")
            print(f"   • ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
            print(f"   • F1-Score: {metrics.get('f1', 0):.4f}")

        if report['best_model_by_business_cost']:
            best = results[report['best_model_by_business_cost']]
            metrics = best.get('classification_metrics', {})
            print(f"\n Лучшая по бизнес-стоимости: {report['best_model_by_business_cost']}")
            print(f"   • Стоимость ошибок: {metrics.get('business_cost', 0):,.0f} у.е.")

        if report['best_model_by_speed']:
            best = results[report['best_model_by_speed']]
            perf = best.get('performance_metrics', {})
            print(f"\n Лучшая по скорости: {report['best_model_by_speed']}")
            print(f"   • RPS: {perf.get('throughput_rps', 0):.0f}")
            print(f"   • Latency: {perf.get('time_per_sample_ms', 0):.3f} мс")

    def run(self):
        """Запуск полного бенчмарка."""
        print("=" * 60)
        print("[START] ЗАПУСК КОМПЛЕКСНОГО БЕНЧМАРКА (Кредитный скоринг)")
        print("=" * 60)

        all_results = {}

        # 1. Бенчмарк ONNX моделей
        print(f"\n1. ТЕСТИРУЕМ ONNX МОДЕЛИ")
        onnx_results = self.benchmark_all_onnx_models()
        if onnx_results:
            all_results.update(onnx_results)

        # 2. Создаем визуализации
        if all_results:
            print(f"\n2. СОЗДАЕМ ВИЗУАЛИЗАЦИИ")
            self.create_comparison_charts(all_results)

            # 3. Генерируем отчет
            print(f"\n3. ГЕНЕРИРУЕМ ОТЧЕТ")
            self.generate_final_report(all_results)

            # Сохраняем все результаты
            results_dir = Path("models/artifacts/benchmark_results")
            results_dir.mkdir(parents=True, exist_ok=True)

            results_path = results_dir / "all_benchmark_results.pkl"
            joblib.dump(all_results, results_path)

            print(f"\n" + "=" * 60)
            print("[SUCCESS] БЕНЧМАРК УСПЕШНО ЗАВЕРШЕН!")
            print("=" * 60)
            print(f"\n[INFO] Результаты сохранены в:")
            print(f"   • models/artifacts/benchmark_results/")
            print(f"   • models/artifacts/benchmark_charts/")
            print(f"   • models/artifacts/benchmark_reports/")

            return all_results
        else:
            print(f"\n[ERROR] Не удалось получить результаты бенчмарка")
            return None


def main():
    """Основная функция."""
    try:
        benchmark = CreditScoringBenchmark()
        results = benchmark.run()

        if results:
            print(f"\n[RECOMMEND] РЕКОМЕНДАЦИИ ДЛЯ ПРОДАКШЕНА:")
            print("-" * 40)
            print("1. Используйте ROC-AUC как основную метрику (несбалансированные данные)")
            print("2. Оптимизируйте порог классификации для вашей бизнес-логики")
            print("3. Balance precision/recall в зависимости от стоимости ошибок")
            print("4. Модель с ROC-AUC > 0.7 считается хорошей для кредитного скоринга")
            print("5. Используйте ONNX формат для production")

    except Exception as e:
        print(f"[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()