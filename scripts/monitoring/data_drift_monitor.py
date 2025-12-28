# scripts/monitoring/data_drift_monitor.py
"""
Мониторинг дрифта данных с использованием Evidently AI.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import TargetDriftPreset
from evidently.metrics import DatasetDriftMetric

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDriftMonitor:
    """Монитор дрифта данных для кредитного скоринга."""

    def __init__(self, reference_data_path: str, report_dir: str = "monitoring/reports"):
        """
        Инициализация монитора дрифта.

        Args:
            reference_data_path: Путь к референсным данным
            report_dir: Директория для отчетов
        """
        self.reference_data = pd.read_csv(reference_data_path)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Определение колонок для мониторинга
        self.column_mapping = ColumnMapping(
            target='default_payment_next_month',
            prediction='prediction',
            numerical_features=['limit_bal', 'age'],
            categorical_features=['sex', 'marriage_new', 'pay_new', 'education_new'],
            datetime_features=None,
            task='classification'
        )

        logger.info(f"Инициализирован монитор дрифта с {len(self.reference_data)} референсными образцами")

    def check_data_drift(self, current_data: pd.DataFrame) -> dict:
        """
        Проверка дрифта между текущими и референсными данными.

        Args:
            current_data: Текущие данные для сравнения

        Returns:
            dict: Результаты проверки дрифта
        """
        logger.info(f"Проверка дрифта для {len(current_data)} текущих образцов")

        # Создание отчета
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(),
            DatasetDriftMetric()
        ])

        # Вычисление метрик
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )

        # Получение результатов
        report_dict = report.as_dict()

        # Извлечение ключевых метрик
        drift_metrics = {
            'timestamp': datetime.now().isoformat(),
            'samples_current': len(current_data),
            'samples_reference': len(self.reference_data),
            'dataset_drift': report_dict['metrics'][0]['result']['dataset_drift'],
            'number_of_drifted_columns': report_dict['metrics'][0]['result']['number_of_drifted_columns'],
            'share_of_drifted_columns': report_dict['metrics'][0]['result']['share_of_drifted_columns'],
            'drift_by_columns': {}
        }

        # Детали по колонкам
        for metric in report_dict['metrics']:
            if 'result' in metric and 'drift_by_columns' in metric['result']:
                for col_name, col_metrics in metric['result']['drift_by_columns'].items():
                    if 'drift_score' in col_metrics:
                        drift_metrics['drift_by_columns'][col_name] = {
                            'drift_score': col_metrics['drift_score'],
                            'drift_detected': col_metrics['drift_detected']
                        }

        # Сохранение отчета
        self._save_report(report, drift_metrics, current_data)

        return drift_metrics

    def _save_report(self, report, drift_metrics: dict, current_data: pd.DataFrame):
        """Сохранение отчета дрифта."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"drift_report_{timestamp}"

        # Сохранение JSON отчета
        json_path = report_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(drift_metrics, f, indent=2, default=str)

        # Сохранение HTML отчета
        html_path = report_path.with_suffix('.html')
        report.save_html(str(html_path))

        # Сохранение текущих данных
        data_path = report_path.with_suffix('.csv')
        current_data.to_csv(data_path, index=False)

        logger.info(f"Отчет дрифта сохранен: {html_path}")
        logger.info(f"Метрики дрифта: {json.dumps(drift_metrics, indent=2)}")

    def generate_alert(self, drift_metrics: dict, threshold: float = 0.5) -> dict:
        """
        Генерация алерта на основе метрик дрифта.

        Args:
            drift_metrics: Метрики дрифта
            threshold: Порог для алерта (доля дрифтующих колонок)

        Returns:
            dict: Алерт или None если дрифт в пределах нормы
        """
        share_drifted = drift_metrics['share_of_drifted_columns']

        if share_drifted > threshold:
            alert = {
                'severity': 'CRITICAL' if share_drifted > 0.7 else 'WARNING',
                'message': f'Обнаружен дрифт данных: {share_drifted:.1%} колонок изменились',
                'timestamp': drift_metrics['timestamp'],
                'share_drifted': share_drifted,
                'drifted_columns': [
                    col for col, metrics in drift_metrics['drift_by_columns'].items()
                    if metrics['drift_detected']
                ]
            }

            logger.warning(f"АЛЕРТ: {alert['message']}")
            return alert

        logger.info(f"Дрифт в пределах нормы: {share_drifted:.1%}")
        return None


def main():
    """Основная функция для мониторинга дрифта."""
    # Пути к данным
    project_root = Path(__file__).parent.parent
    reference_path = project_root / "data" / "processed_fixed" / "train.csv"
    current_path = project_root / "data" / "processed_fixed" / "test.csv"

    if not reference_path.exists() or not current_path.exists():
        logger.error("Данные не найдены. Сначала запустите пайплайн подготовки данных.")
        return

    # Инициализация монитора
    monitor = DataDriftMonitor(str(reference_path))

    # Загрузка текущих данных
    current_data = pd.read_csv(current_path)

    # Проверка дрифта
    drift_metrics = monitor.check_data_drift(current_data)

    # Генерация алерта
    alert = monitor.generate_alert(drift_metrics)

    if alert:
        # Здесь можно добавить отправку алерта (Slack, Email, etc.)
        print(f"🚨 АЛЕРТ: {alert['message']}")
        print(f"   Серьезность: {alert['severity']}")
        print(f"   Дрифтующие колонки: {', '.join(alert['drifted_columns'])}")


if __name__ == "__main__":
    main()