# scripts/monitoring/concept_drift_monitor.py
"""
Мониторинг концептуального дрифта (изменения в отношениях фич-таргет).
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from evidently import ColumnMapping
from evidently.metric_preset import ClassificationPreset
from evidently.report import Report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConceptDriftMonitor:
    """Монитор концептуального дрифта."""

    def __init__(self, model, reference_data_path: str):
        """
        Args:
            model: Обученная ML модель
            reference_data_path: Путь к референсным данным
        """
        self.model = model
        self.reference_data = pd.read_csv(reference_data_path)
        self.report_dir = Path("monitoring/reports/concept_drift")
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Предсказания на референсных данных
        X_ref = self.reference_data.drop('default_payment_next_month', axis=1)
        self.reference_data['prediction'] = self.model.predict_proba(X_ref)[:, 1]

    def check_concept_drift(self, current_data: pd.DataFrame) -> dict:
        """
        Проверка концептуального дрифта.

        Args:
            current_data: Текущие данные с реальными таргетами

        Returns:
            dict: Метрики концептуального дрифта
        """
        # Предсказания на текущих данных
        X_curr = current_data.drop('default_payment_next_month', axis=1)
        current_data['prediction'] = self.model.predict_proba(X_curr)[:, 1]

        # Создание отчета
        column_mapping = ColumnMapping(
            target='default_payment_next_month',
            prediction='prediction',
            task='classification'
        )

        report = Report(metrics=[
            ClassificationPreset()
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )

        # Извлечение метрик
        report_dict = report.as_dict()
        metrics = report_dict['metrics'][0]['result']

        # Дополнительные метрики
        metrics['current_auc'] = roc_auc_score(
            current_data['default_payment_next_month'],
            current_data['prediction']
        )
        metrics['reference_auc'] = roc_auc_score(
            self.reference_data['default_payment_next_month'],
            self.reference_data['prediction']
        )
        metrics['auc_drift'] = abs(metrics['current_auc'] - metrics['reference_auc'])

        # Сохранение отчета
        self._save_report(report, metrics)

        return metrics

    def _save_report(self, report, metrics: dict):
        """Сохранение отчета концептуального дрифта."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"concept_drift_{timestamp}.html"
        report.save_html(str(report_path))

        # Логирование метрик
        logger.info(f"Concept Drift Report saved: {report_path}")
        logger.info(f"AUC Reference: {metrics['reference_auc']:.3f}")
        logger.info(f"AUC Current: {metrics['current_auc']:.3f}")
        logger.info(f"AUC Drift: {metrics['auc_drift']:.3f}")


# Интеграция в API для мониторинга в реальном времени