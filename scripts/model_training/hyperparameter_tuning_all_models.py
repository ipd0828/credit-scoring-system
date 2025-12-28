#!/usr/bin/env python3
"""
Скрипт для подбора гиперпараметров всех 4 моделей на кастомных признаках.
Обновленная версия для 4 моделей: LogReg, RandomForest, LightGBM, Neural Network
С добавленной обработкой дисбаланса классов для LightGBM
АВТОМАТИЧЕСКИЙ РЕЖИМ - без интерактивных вопросов
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
    classification_report
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings("ignore")

# Пытаемся импортировать LightGBM
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM не установлен. Установите: pip install lightgbm")

# Пытаемся импортировать PyTorch для нейронной сети
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch не установлен. Установите: pip install torch")


def load_processed_data():
    """Загрузка обработанных данных для кастомных признаков."""
    print("Загрузка обработанных данных...")

    data_path = Path("data/processed_fixed")

    if not data_path.exists():
        raise FileNotFoundError(f"Папка с данными не найдена: {data_path}")

    X_train = pd.read_csv(data_path / "X_train.csv")
    X_val = pd.read_csv(data_path / "X_val.csv")
    X_test = pd.read_csv(data_path / "X_test.csv")
    y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
    y_val = pd.read_csv(data_path / "y_val.csv").squeeze()
    y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

    # Анализ дисбаланса классов
    print(f"\nАНАЛИЗ ДИСБАЛАНСА КЛАССОВ:")
    print(f"Обучающая выборка (y_train):")
    class_counts_train = y_train.value_counts()
    class_ratio_train = class_counts_train[0] / class_counts_train[1]
    print(f"  Класс 0 (хорошие): {class_counts_train[0]} ({class_counts_train[0] / len(y_train) * 100:.1f}%)")
    print(f"  Класс 1 (плохие): {class_counts_train[1]} ({class_counts_train[1] / len(y_train) * 100:.1f}%)")
    print(f"  Соотношение (0:1): {class_ratio_train:.2f}:1")

    print(f"\nВалидационная выборка (y_val):")
    class_counts_val = y_val.value_counts()
    class_ratio_val = class_counts_val[0] / class_counts_val[1]
    print(f"  Класс 0 (хорошие): {class_counts_val[0]} ({class_counts_val[0] / len(y_val) * 100:.1f}%)")
    print(f"  Класс 1 (плохие): {class_counts_val[1]} ({class_counts_val[1] / len(y_val) * 100:.1f}%)")
    print(f"  Соотношение (0:1): {class_ratio_val:.2f}:1")

    print(f"\nТестовая выборка (y_test):")
    class_counts_test = y_test.value_counts()
    class_ratio_test = class_counts_test[0] / class_counts_test[1]
    print(f"  Класс 0 (хорошие): {class_counts_test[0]} ({class_counts_test[0] / len(y_test) * 100:.1f}%)")
    print(f"  Класс 1 (плохие): {class_counts_test[1]} ({class_counts_test[1] / len(y_test) * 100:.1f}%)")
    print(f"  Соотношение (0:1): {class_ratio_test:.2f}:1")

    print(f"\nЗагружено:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_val:   {y_val.shape}")
    print(f"  y_test:  {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_param_grids():
    """Определение сеток параметров для всех 4 моделей."""
    print("Определение сеток параметров для 4 моделей...")

    param_grids = {}

    # 1. Logistic Regression
    param_grids["LogisticRegression"] = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ['l1', 'l2', 'elasticnet', None],
        "solver": ['liblinear', 'saga'],
        "max_iter": [100, 500, 1000],
        "class_weight": [None, 'balanced']
    }

    # 2. Random Forest
    param_grids["RandomForestClassifier"] = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ['sqrt', 'log2', None],
        "bootstrap": [True, False],
        "class_weight": [None, 'balanced', 'balanced_subsample']
    }

    # 3. LightGBM - ОБНОВЛЕННАЯ СЕТКА С ФОКУСОМ НА ДИСБАЛАНС КЛАССОВ
    if LIGHTGBM_AVAILABLE:
        param_grids["LGBMClassifier"] = {
            # Основные параметры
            "n_estimators": [100, 300, 500],
            "learning_rate": [0.01, 0.001, 0.0001],
            "num_leaves": [5, 10, 20],
            "max_depth": [-1, 5, 8, 12],

            # Параметры для борьбы с дисбалансом
            "min_child_samples": [10, 20, 50],
            "min_child_weight": [0.001, 0.01, 0.1],
            "min_split_gain": [0.0, 0.1],

            # Регуляризация
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5],

            # Сэмплирование
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "subsample_freq": [0],

            # Важные параметры для дисбаланса
            "scale_pos_weight": [1, 3, 5],  # Вес для положительного класса (3.52:1 соотношение)
            "is_unbalance": [True, False],
            "boosting_type": ['gbdt', 'dart'],
        }

    # 4. Neural Network - отдельная обработка
    param_grids["NeuralNetwork"] = {
        "hidden_layers": [[32, 16], [128, 64]],
        "learning_rate": [0.001],
        "batch_size": [32],
        "epochs": [70],
        "dropout_rate": [0.0, 0.2]
    }

    print(f"Создано {len(param_grids)} сеток параметров для 4 моделей")
    return param_grids


def tune_lightgbm_with_imbalance(X_train, y_train, X_val, y_val):
    """Подбор гиперпараметров для LightGBM с фокусом на дисбаланс классов."""
    print("\n" + "=" * 40)
    print("ТЮНИНГ LIGHTGBM С ОБРАБОТКОЙ ДИСБАЛАНСА КЛАССОВ")
    print("=" * 40)

    if not LIGHTGBM_AVAILABLE:
        print("LightGBM недоступен. Пропускаем тюнинг.")
        return None, {}

    # Анализ дисбаланса классов
    class_counts = y_train.value_counts()
    class_ratio = class_counts[0] / class_counts[1]
    print(f"Дисбаланс классов в обучающей выборке:")
    print(f"  Класс 0 (хорошие): {class_counts[0]} ({class_counts[0] / len(y_train) * 100:.1f}%)")
    print(f"  Класс 1 (плохие): {class_counts[1]} ({class_counts[1] / len(y_train) * 100:.1f}%)")
    print(f"  Соотношение: {class_ratio:.2f}:1")

    # Расчет scale_pos_weight
    scale_pos_weight_value = class_counts[0] / class_counts[1]
    print(f"  Автоматически установлен scale_pos_weight: {scale_pos_weight_value:.2f}")

    # Используем оптимизированную сетку параметров
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.001, 0.1],
        "num_leaves": [5, 20],
        "max_depth": [-1, 5, 10],
        "min_child_samples": [10, 20, 50],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [0, 0.1],
        "scale_pos_weight": [scale_pos_weight_value, scale_pos_weight_value * 0.7, scale_pos_weight_value * 1.3],
        "is_unbalance": [True],
        "boosting_type": ['gbdt', 'dart'],
    }

    # Используем RandomizedSearchCV с фокусом на F1-score
    print("Запуск поиска гиперпараметров с фокусом на F1-score...")

    search = RandomizedSearchCV(
        LGBMClassifier(
            random_state=42,
            verbose=-1,
            n_jobs=-1,
            n_estimators=100
        ),
        param_distributions=param_grid,
        n_iter=15,  # Проверяем 15 комбинаций
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='f1',  # Фокус на F1-score для дисбаланса
        n_jobs=-1,
        verbose=1,
        random_state=42,
        error_score='raise'  # Будем видеть ошибки явно
    )

    try:
        search.fit(X_train, y_train)

        print(f"\nЛучшие параметры:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Лучший F1-score: {search.best_score_:.4f}")

        # Оценка на валидации
        y_pred = search.best_estimator_.predict(X_val)
        y_proba = search.best_estimator_.predict_proba(X_val)[:, 1]

        val_f1 = f1_score(y_val, y_pred, zero_division=0)
        val_auc = roc_auc_score(y_val, y_proba)
        val_accuracy = accuracy_score(y_val, y_pred)
        val_precision = precision_score(y_val, y_pred, zero_division=0)
        val_recall = recall_score(y_val, y_pred, zero_division=0)

        print(f"\nНа валидационной выборке:")
        print(f"  F1-score:   {val_f1:.4f}")
        print(f"  ROC-AUC:    {val_auc:.4f}")
        print(f"  Accuracy:   {val_accuracy:.4f}")
        print(f"  Precision:  {val_precision:.4f}")
        print(f"  Recall:     {val_recall:.4f}")

        # Выводим classification report для анализа
        print(f"\nClassification Report:")
        report = classification_report(y_val, y_pred, zero_division=0)
        print(report)

        # Обучение финальной модели на объединенных данных
        print("\nОбучение финальной модели на train+val данных...")
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])

        # Увеличиваем n_estimators для финальной модели
        final_params = search.best_params_.copy()
        if 'n_estimators' in final_params:
            final_params['n_estimators'] = min(300, final_params['n_estimators'] * 1.5)
        else:
            final_params['n_estimators'] = 200

        final_model = LGBMClassifier(
            **final_params,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )

        final_model.fit(X_train_val, y_train_val)

        return final_model, final_params

    except Exception as e:
        print(f"Ошибка при поиске гиперпараметров: {e}")
        print("Пробуем упрощенный подход...")
        return tune_lightgbm_simple(X_train, y_train, X_val, y_val)


def tune_lightgbm_simple(X_train, y_train, X_val, y_val):
    """Упрощенная версия тюнинга LightGBM."""
    print("\n" + "=" * 40)
    print("ТЮНИНГ LIGHTGBM (упрощенная версия)")
    print("=" * 40)

    if not LIGHTGBM_AVAILABLE:
        print("LightGBM недоступен. Пропускаем тюнинг.")
        return None, {}

    # Базовая сетка параметров
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "num_leaves": [31, 63],
        "max_depth": [-1, 5],
        "min_child_samples": [20, 50],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [0, 0.1],
    }

    # Добавляем параметры для дисбаланса
    class_counts = y_train.value_counts()
    if len(class_counts) == 2:
        scale_pos_weight = class_counts[0] / class_counts[1]
        print(f"  Автоматический scale_pos_weight: {scale_pos_weight:.2f}")
        param_grid["scale_pos_weight"] = [1, scale_pos_weight]
        param_grid["is_unbalance"] = [True]

    search = RandomizedSearchCV(
        LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
        param_distributions=param_grid,
        n_iter=10,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    print("Запуск поиска гиперпараметров...")
    search.fit(X_train, y_train)

    print(f"\nЛучшие параметры:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Лучший ROC-AUC: {search.best_score_:.4f}")

    # Оценка на валидационной выборке
    y_pred = search.best_estimator_.predict(X_val)
    y_proba = search.best_estimator_.predict_proba(X_val)[:, 1]

    val_auc = roc_auc_score(y_val, y_proba)
    val_accuracy = accuracy_score(y_val, y_pred)
    val_f1 = f1_score(y_val, y_pred, zero_division=0)
    val_precision = precision_score(y_val, y_pred, zero_division=0)
    val_recall = recall_score(y_val, y_pred, zero_division=0)

    print(f"\nНа валидационной выборке:")
    print(f"  ROC-AUC:    {val_auc:.4f}")
    print(f"  Accuracy:   {val_accuracy:.4f}")
    print(f"  F1-Score:   {val_f1:.4f}")
    print(f"  Precision:  {val_precision:.4f}")
    print(f"  Recall:     {val_recall:.4f}")

    # Выводим classification report для анализа
    print(f"\nClassification Report:")
    report = classification_report(y_val, y_pred, zero_division=0)
    print(report)

    return search.best_estimator_, search.best_params_


def tune_logistic_regression(X_train, y_train, X_val, y_val):
    """Подбор гиперпараметров для Logistic Regression."""
    print("\n" + "=" * 40)
    print("ТЮНИНГ LOGISTIC REGRESSION")
    print("=" * 40)

    param_grid = get_param_grids()["LogisticRegression"]

    # Используем RandomizedSearchCV для скорости
    search = RandomizedSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_distributions=param_grid,
        n_iter=20,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    print("Запуск поиска гиперпараметров...")
    search.fit(X_train, y_train)

    print(f"\nЛучшие параметры:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Лучший ROC-AUC: {search.best_score_:.4f}")

    # Оценка на валидационной выборке
    y_pred = search.best_estimator_.predict(X_val)
    y_proba = search.best_estimator_.predict_proba(X_val)[:, 1]

    val_auc = roc_auc_score(y_val, y_proba)
    val_accuracy = accuracy_score(y_val, y_pred)
    val_f1 = f1_score(y_val, y_pred, zero_division=0)

    print(f"\nНа валидационной выборке:")
    print(f"  ROC-AUC: {val_auc:.4f}")
    print(f"  Accuracy: {val_accuracy:.4f}")
    print(f"  F1-Score: {val_f1:.4f}")

    return search.best_estimator_, search.best_params_


def tune_random_forest(X_train, y_train, X_val, y_val):
    """Подбор гиперпараметров для Random Forest."""
    print("\n" + "=" * 40)
    print("ТЮНИНГ RANDOM FOREST")
    print("=" * 40)

    param_grid = get_param_grids()["RandomForestClassifier"]

    # Используем RandomizedSearchCV
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_grid,
        n_iter=15,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    print("Запуск поиска гиперпараметров...")
    search.fit(X_train, y_train)

    print(f"\nЛучшие параметры:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Лучший ROC-AUC: {search.best_score_:.4f}")

    # Оценка на валидационной выборке
    y_pred = search.best_estimator_.predict(X_val)
    y_proba = search.best_estimator_.predict_proba(X_val)[:, 1]

    val_auc = roc_auc_score(y_val, y_proba)
    val_accuracy = accuracy_score(y_val, y_pred)
    val_f1 = f1_score(y_val, y_pred, zero_division=0)

    print(f"\nНа валидационной выборке:")
    print(f"  ROC-AUC: {val_auc:.4f}")
    print(f"  Accuracy: {val_accuracy:.4f}")
    print(f"  F1-Score: {val_f1:.4f}")

    return search.best_estimator_, search.best_params_


class SimpleNN(nn.Module):
    """Простая нейронная сеть с настраиваемой архитектурой."""

    def __init__(self, input_size, hidden_layers, dropout_rate=0.0):
        super(SimpleNN, self).__init__()

        layers = []
        prev_size = input_size

        # Добавляем скрытые слои
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Выходной слой
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def tune_neural_network(X_train, y_train, X_val, y_val):
    """Подбор гиперпараметров для нейронной сети."""
    print("\n" + "=" * 40)
    print("ТЮНИНГ NEURAL NETWORK")
    print("=" * 40)

    if not TORCH_AVAILABLE:
        print("PyTorch недоступен. Пропускаем тюнинг нейронной сети.")
        return None, {}

    param_grid = get_param_grids()["NeuralNetwork"]
    input_size = X_train.shape[1]

    best_model = None
    best_params = {}
    best_val_auc = 0.0

    # Масштабирование данных
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    # Конвертация в тензоры
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1)

    # Создаем датасеты
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Упрощенный поиск - проверяем только 4 комбинации для скорости
    trial_count = 0
    for hidden_layers in [[32, 16], [64, 32]]:  # Только 2 варианта
        for dropout_rate in [0.0, 0.2]:  # Только 2 варианта
            trial_count += 1
            print(f"\nТренировка {trial_count}: hidden_layers={hidden_layers}, dropout={dropout_rate}")

            # Создаем модель
            model = SimpleNN(input_size, hidden_layers, dropout_rate)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # DataLoader
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            # Обучение (уменьшено до 70 эпох)
            model.train()
            for epoch in range(70):
                epoch_loss = 0.0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    print(f"    Эпоха {epoch + 1}/70, Loss: {epoch_loss / len(train_loader):.4f}")

            # Оценка на валидации
            model.eval()
            with torch.no_grad():
                outputs = model(X_val_tensor)
                y_proba = outputs.numpy().flatten()
                y_pred = (y_proba >= 0.5).astype(int)

            val_auc = roc_auc_score(y_val, y_proba)
            val_accuracy = accuracy_score(y_val, y_pred)
            val_f1 = f1_score(y_val, y_pred, zero_division=0)

            print(f"    Val AUC: {val_auc:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

            # Сохраняем лучшую модель
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model = model
                best_params = {
                    "hidden_layers": hidden_layers,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 70,
                    "dropout_rate": dropout_rate,
                    "input_size": input_size
                }

    print(f"\nЛучшие параметры нейронной сети:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Лучший ROC-AUC на валидации: {best_val_auc:.4f}")

    return best_model, best_params


def evaluate_tuned_models(tuned_models, X_test, y_test):
    """Оценка настроенных моделей на тестовых данных."""
    print("\n" + "=" * 60)
    print("ОЦЕНКА НАСТРОЕННЫХ МОДЕЛЕЙ НА ТЕСТОВЫХ ДАННЫХ")
    print("=" * 60)

    results = []

    for model_name, (model, params) in tuned_models.items():
        if model is None:
            continue

        print(f"\nОценка {model_name}...")

        try:
            # Предсказания
            if model_name == "NeuralNetwork":
                # Масштабирование для нейронной сети
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_test_scaled = scaler.fit_transform(X_test).astype(np.float32)
                X_test_tensor = torch.FloatTensor(X_test_scaled)

                model.eval()
                with torch.no_grad():
                    outputs = model(X_test_tensor)
                    y_proba = outputs.numpy().flatten()
                    y_pred = (y_proba >= 0.5).astype(int)
            else:
                # Для sklearn моделей
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)

            print(f"{model_name}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-score:  {f1:.4f}")
            print(f"  ROC-AUC:   {roc_auc:.4f}")

            # Кросс-валидация (только для sklearn моделей)
            if model_name != "NeuralNetwork":
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(
                    model, X_test, y_test, cv=cv, scoring="roc_auc"
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                print(f"  CV AUC:    {cv_mean:.4f} ± {cv_std:.4f}")
            else:
                cv_mean = roc_auc
                cv_std = 0.0

            results.append({
                "model": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "cv_auc_mean": cv_mean,
                "cv_auc_std": cv_std,
                "best_params": params
            })

            # Сохраняем настроенную модель
            models_dir = Path("models/trained_final")
            models_dir.mkdir(parents=True, exist_ok=True)

            if model_name == "NeuralNetwork":
                model_path = models_dir / "tuned_neural_network.pth"
                torch.save(model.state_dict(), model_path)
                print(f"  Настроенная модель сохранена: {model_path}")

                # Сохраняем параметры
                params_path = models_dir / "tuned_neural_network_params.pkl"
                joblib.dump(params, params_path)
            else:
                model_path = models_dir / f"tuned_{model_name.lower()}.pkl"
                joblib.dump(model, model_path)
                print(f"  Настроенная модель сохранена: {model_path}")

        except Exception as e:
            print(f"  Ошибка при оценке {model_name}: {e}")
            continue

    return results


def save_tuning_results(results):
    """Сохранение результатов настройки."""
    print("\nСохранение результатов настройки...")

    if not results:
        print("  Нет результатов для сохранения")
        return None

    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("roc_auc", ascending=False)

    # Сохраняем результаты
    output_dir = Path("models/trained_final")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "tuned_models_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"  Результаты сохранены: {results_path}")

    # Сохраняем лучшую настроенную модель
    if not results_df.empty:
        best_model_name = results_df.iloc[0]["model"]

        if best_model_name == "NeuralNetwork":
            best_model_path = output_dir / "tuned_neural_network.pth"
            best_params_path = output_dir / "tuned_neural_network_params.pkl"

            # Для нейронной сети сохраняем оба файла
            import shutil
            final_best_path = output_dir / "best_tuned_model.pth"
            if best_model_path.exists():
                shutil.copy2(best_model_path, final_best_path)
                print(f"  Лучшая настроенная модель сохранена: {final_best_path}")
        else:
            best_model_path = output_dir / f"tuned_{best_model_name.lower()}.pkl"
            final_best_path = output_dir / "best_tuned_model.pkl"

            import shutil
            if best_model_path.exists():
                shutil.copy2(best_model_path, final_best_path)
                print(f"  Лучшая настроенная модель сохранена: {final_best_path}")

        print(f"  Лучшая настроенная модель: {best_model_name}")
        print(f"  ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f}")
        print(f"  Accuracy: {results_df.iloc[0]['accuracy']:.4f}")
        print(f"  F1-Score: {results_df.iloc[0]['f1_score']:.4f}")

    return results_df


def main():
    """Основная функция."""
    try:
        # Загрузка данных
        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()

        print(f"\n{'=' * 60}")
        print("ПОДБОР ГИПЕРПАРАМЕТРОВ ДЛЯ 4 МОДЕЛЕЙ")
        print("=" * 60)
        print("Модели: Logistic Regression, Random Forest, LightGBM, Neural Network")
        print("АВТОМАТИЧЕСКИЙ РЕЖИМ - используется продвинутый тюнинг для LightGBM")

        tuned_models = {}

        # 1. Тюнинг Logistic Regression
        try:
            print(f"\n{'=' * 60}")
            lr_model, lr_params = tune_logistic_regression(X_train, y_train, X_val, y_val)
            tuned_models["LogisticRegression"] = (lr_model, lr_params)
        except Exception as e:
            print(f"Ошибка при тюнинге Logistic Regression: {e}")

        # 2. Тюнинг Random Forest
        try:
            print(f"\n{'=' * 60}")
            rf_model, rf_params = tune_random_forest(X_train, y_train, X_val, y_val)
            tuned_models["RandomForestClassifier"] = (rf_model, rf_params)
        except Exception as e:
            print(f"Ошибка при тюнинге Random Forest: {e}")

        # 3. Тюнинг LightGBM - ВСЕГДА используем продвинутую версию
        if LIGHTGBM_AVAILABLE:
            try:
                print(f"\n{'=' * 60}")
                print("ЗАПУСК ПРОДВИНУТОГО ТЮНИНГА LIGHTGBM")
                print("(автоматическая обработка дисбаланса классов)")

                lgb_model, lgb_params = tune_lightgbm_with_imbalance(X_train, y_train, X_val, y_val)
                tuned_models["LGBMClassifier"] = (lgb_model, lgb_params)

            except Exception as e:
                print(f"Ошибка при тюнинге LightGBM: {e}")
                import traceback
                traceback.print_exc()
                print("\nПробуем упрощенный тюнинг...")
                try:
                    lgb_model, lgb_params = tune_lightgbm_simple(X_train, y_train, X_val, y_val)
                    tuned_models["LGBMClassifier"] = (lgb_model, lgb_params)
                except Exception as e2:
                    print(f"Ошибка при упрощенном тюнинге LightGBM: {e2}")
        else:
            print("\nLightGBM недоступен, пропускаем тюнинг.")

        # 4. Тюнинг Neural Network (упрощенный для скорости)
        if TORCH_AVAILABLE:
            try:
                print(f"\n{'=' * 60}")
                nn_model, nn_params = tune_neural_network(X_train, y_train, X_val, y_val)
                tuned_models["NeuralNetwork"] = (nn_model, nn_params)
            except Exception as e:
                print(f"Ошибка при тюнинге Neural Network: {e}")
        else:
            print("\nPyTorch недоступен, пропускаем тюнинг нейронной сети.")

        # Оценка всех настроенных моделей
        print(f"\n{'=' * 60}")
        print("ОЦЕНКА ВСЕХ НАСТРОЕННЫХ МОДЕЛЕЙ")
        print("=" * 60)

        if tuned_models:
            evaluation_results = evaluate_tuned_models(tuned_models, X_test, y_test)

            # Сохранение результатов
            results_df = save_tuning_results(evaluation_results)

            print(f"\n{'=' * 60}")
            print("ПОДБОР ГИПЕРПАРАМЕТРОВ ЗАВЕРШЕН УСПЕШНО")
            print("=" * 60)

            if results_df is not None and not results_df.empty:
                print(f"\nРЕЙТИНГ МОДЕЛЕЙ ПО ROC-AUC:")
                for i, (_, row) in enumerate(results_df.iterrows(), 1):
                    print(f"{i}. {row['model']}: ROC-AUC={row['roc_auc']:.4f}, "
                          f"Accuracy={row['accuracy']:.4f}, F1={row['f1_score']:.4f}")

                best_model = results_df.iloc[0]
                print(f"\n ЛУЧШАЯ МОДЕЛЬ: {best_model['model']}")
                print(f"   ROC-AUC: {best_model['roc_auc']:.4f}")
                print(f"   Accuracy: {best_model['accuracy']:.4f}")
                print(f"   F1-Score: {best_model['f1_score']:.4f}")
        else:
            print(" Не удалось настроить ни одну модель")

    except Exception as e:
        print(f"\n Критическая ошибка в пайплайне тюнинга: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()