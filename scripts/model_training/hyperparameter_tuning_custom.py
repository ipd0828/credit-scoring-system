"""
Скрипт для подбора гиперпараметров моделей на кастомных признаках.
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings("ignore")

# Пытаемся импортировать CatBoost
try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost не установлен. Установите: pip install catboost")


def load_processed_data():
    """Загрузка обработанных данных для кастомных признаков."""
    print("Загрузка обработанных данных...")

    data_path = Path("data/processed_custom")

    if not data_path.exists():
        raise FileNotFoundError(f"Папка с данными не найдена: {data_path}")

    X_train = pd.read_csv(data_path / "X_train.csv")
    X_test = pd.read_csv(data_path / "X_test.csv")
    y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
    y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

    print(f"Загружено:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def fix_categorical_features(X_train, X_test):
    """Исправляет типы категориальных признаков для CatBoost."""
    categorical_features = ['sex', 'marriage_new', 'pay_new', 'education_new']

    for feature in categorical_features:
        if feature in X_train.columns:
            # Преобразуем в целые числа, затем в строки (для CatBoost)
            X_train[feature] = X_train[feature].astype(int).astype(str)
            X_test[feature] = X_test[feature].astype(int).astype(str)

    print("Категориальные признаки преобразованы для CatBoost")  # Убрали emoji
    return X_train, X_test


def get_param_grids():
    """Определение сеток параметров для моделей."""
    print("Определение сеток параметров...")

    param_grids = {}

    # Logistic Regression
    param_grids['LogisticRegression'] = {
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['liblinear'],
        'classifier__max_iter': [1000]
    }

    # Random Forest
    param_grids['RandomForestClassifier'] = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    # CatBoost
    if CATBOOST_AVAILABLE:
        param_grids['CatBoostClassifier'] = {
            'classifier__iterations': [100, 200],
            'classifier__depth': [4, 6],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__l2_leaf_reg': [3, 5]
        }

    print(f"Создано {len(param_grids)} сеток параметров")
    return param_grids


def create_models():
    """Создание базовых моделей для настройки."""
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced'),
        'RandomForestClassifier': RandomForestClassifier(random_state=42, class_weight='balanced'),
    }

    if CATBOOST_AVAILABLE:
        models['CatBoostClassifier'] = CatBoostClassifier(
            random_state=42,
            verbose=False,
            thread_count=-1,
            # Добавленные настройки
            loss_function='Logloss',
            auto_class_weights='Balanced',
            eval_metric='AUC'
        )

    return models


def get_categorical_features(X_train):
    """Определение категориальных признаков для CatBoost."""
    categorical_features = ['sex', 'marriage_new', 'pay_new', 'education_new']
    cat_features_indices = [i for i, col in enumerate(X_train.columns) if col in categorical_features]
    return cat_features_indices


def tune_hyperparameters(X_train, y_train):
    """Подбор гиперпараметров для моделей."""
    print("\n" + "=" * 60)
    print("ПОИСК ПО СЕТКЕ ПАРАМЕТРОВ")
    print("=" * 60)

    param_grids = get_param_grids()
    models = create_models()
    best_models = {}
    results = []

    # Получаем индексы категориальных признаков для CatBoost
    cat_features_indices = get_categorical_features(X_train)

    for model_name, model in models.items():
        if model_name not in param_grids:
            continue

        print(f"\nПоиск параметров для {model_name}...")

        # Создаем pipeline
        pipeline = Pipeline([
            ('classifier', model)
        ])

        # Настройка гиперпараметров
        grid_search = GridSearchCV(
            pipeline,
            param_grids[model_name],
            cv=2,  # Уменьшаем для скорости
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        # Особый подход для CatBoost с категориальными признаками
        if model_name == 'CatBoostClassifier' and CATBOOST_AVAILABLE:
            try:
                # Для CatBoost используем прямой вызов без Pipeline
                catboost_model = CatBoostClassifier(
                    random_state=42,
                    verbose=False,
                    thread_count=-1,
                    loss_function='Logloss',
                    auto_class_weights='Balanced',
                    eval_metric='AUC'
                )

                catboost_grid_search = GridSearchCV(
                    catboost_model,
                    {
                        'iterations': [100, 200],
                        'depth': [4, 6],
                        'learning_rate': [0.05, 0.1],
                        'l2_leaf_reg': [3, 5]
                    },
                    cv=2,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )

                # Преобразуем категориальные признаки в правильный формат
                X_train_fixed = X_train.copy()
                categorical_features = ['sex', 'marriage_new', 'pay_new', 'education_new']
                for feature in categorical_features:
                    if feature in X_train_fixed.columns:
                        X_train_fixed[feature] = X_train_fixed[feature].astype(str)

                # Обучаем с категориальными признаками
                catboost_grid_search.fit(X_train_fixed, y_train, cat_features=cat_features_indices)
                best_models[model_name] = catboost_grid_search.best_estimator_
                best_score = catboost_grid_search.best_score_
                best_params = catboost_grid_search.best_params_

                print(f"{model_name} завершён:")
                print(f"  Лучший AUC: {best_score:.4f}")
                print(f"  Лучшие параметры: {best_params}")

                results.append({
                    'model': model_name,
                    'best_score': best_score,
                    'best_params': best_params,
                    'best_estimator': best_models[model_name]
                })

            except Exception as e:
                print(f"Ошибка при настройке CatBoost: {e}")  # Убрали emoji
                print("Пропускаем CatBoost и продолжаем с другими моделями")  # Убрали emoji
                continue

        else:
            # Стандартный подход для других моделей
            try:
                grid_search.fit(X_train, y_train)
                best_models[model_name] = grid_search.best_estimator_
                best_score = grid_search.best_score_
                best_params = grid_search.best_params_

                print(f"{model_name} завершён:")
                print(f"  Лучший AUC: {best_score:.4f}")
                print(f"  Лучшие параметры: {best_params}")

                results.append({
                    'model': model_name,
                    'best_score': best_score,
                    'best_params': best_params,
                    'best_estimator': best_models[model_name]
                })
            except Exception as e:
                print(f"Ошибка при настройке {model_name}: {e}")  # Убрали emoji
                continue

    return best_models, results


def evaluate_tuned_models(best_models, X_test, y_test):
    """Оценка настроенных моделей на тестовых данных."""
    print("\n" + "=" * 60)
    print("ОЦЕНКА НАСТРОЕННЫХ МОДЕЛЕЙ")
    print("=" * 60)

    results = []

    # Получаем индексы категориальных признаков для CatBoost
    cat_features_indices = get_categorical_features(X_test)

    for model_name, model in best_models.items():
        print(f"\nОценка {model_name}...")

        try:
            # Предсказания
            if model_name == 'CatBoostClassifier':
                # Для CatBoost преобразуем категориальные признаки
                X_test_fixed = X_test.copy()
                categorical_features = ['sex', 'marriage_new', 'pay_new', 'education_new']
                for feature in categorical_features:
                    if feature in X_test_fixed.columns:
                        X_test_fixed[feature] = X_test_fixed[feature].astype(str)

                y_pred = model.predict(X_test_fixed)
                y_proba = model.predict_proba(X_test_fixed)[:, 1]
            else:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)

            # Кросс-валидация
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            if model_name == 'CatBoostClassifier':
                cv_scores = []
                for train_idx, val_idx in cv.split(X_test, y_test):
                    X_train_cv, X_val_cv = X_test.iloc[train_idx], X_test.iloc[val_idx]
                    y_train_cv, y_val_cv = y_test.iloc[train_idx], y_test.iloc[val_idx]

                    # Преобразуем категориальные признаки для CV
                    X_train_cv_fixed = X_train_cv.copy()
                    X_val_cv_fixed = X_val_cv.copy()
                    for feature in categorical_features:
                        if feature in X_train_cv_fixed.columns:
                            X_train_cv_fixed[feature] = X_train_cv_fixed[feature].astype(str)
                            X_val_cv_fixed[feature] = X_val_cv_fixed[feature].astype(str)

                    # Создаем временную модель для CV
                    temp_model = CatBoostClassifier(**model.get_params())
                    temp_model.fit(X_train_cv_fixed, y_train_cv, cat_features=cat_features_indices, verbose=False)
                    y_proba_cv = temp_model.predict_proba(X_val_cv_fixed)[:, 1]
                    cv_score = roc_auc_score(y_val_cv, y_proba_cv)
                    cv_scores.append(cv_score)

                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
            else:
                cv_scores = cross_val_score(model, X_test, y_test, cv=cv, scoring='roc_auc')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

            print(f"{model_name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-score: {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
            print(f"  CV AUC: {cv_mean:.4f} ± {cv_std:.4f}")

            results.append({
                'model': model_name,
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_auc_mean': cv_mean,
                'cv_auc_std': cv_std
            })

            # Сохраняем настроенную модель
            models_dir = Path("models/trained_custom")
            models_dir.mkdir(parents=True, exist_ok=True)

            model_path = models_dir / f"tuned_{model_name.lower()}.pkl"
            joblib.dump(model, model_path)
            print(f"  Настроенная модель сохранена: {model_path}")

        except Exception as e:
            print(f"  Ошибка при оценке {model_name}: {e}")  # Убрали emoji
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
    results_df = results_df.sort_values('roc_auc', ascending=False)

    # Сохраняем результаты
    output_dir = Path("models/trained_custom")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "tuned_models_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"  Результаты сохранены: {results_path}")

    # Сохраняем лучшую настроенную модель
    best_model_name = results_df.iloc[0]['model']
    best_model_path = output_dir / f"tuned_{best_model_name.lower()}.pkl"
    final_best_path = output_dir / "best_tuned_model.pkl"

    # Копируем лучшую модель
    import shutil
    shutil.copy2(best_model_path, final_best_path)
    print(f"  Лучшая настроенная модель сохранена: {final_best_path}")
    print(f"  Лучшая настроенная модель: {best_model_name} (ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f})")

    return results_df


def main():
    """Основная функция."""
    try:
        # Загрузка данных
        X_train, X_test, y_train, y_test = load_processed_data()

        # Исправляем категориальные признаки для CatBoost
        X_train_fixed, X_test_fixed = fix_categorical_features(X_train, X_test)

        # Подбор гиперпараметров
        best_models, tuning_results = tune_hyperparameters(X_train_fixed, y_train)

        # Оценка настроенных моделей
        evaluation_results = evaluate_tuned_models(best_models, X_test_fixed, y_test)

        # Сохранение результатов
        results_df = save_tuning_results(evaluation_results)

        print("\n" + "=" * 60)
        print("ПОДБОР ГИПЕРПАРАМЕТРОВ ЗАВЕРШЕН УСПЕШНО")
        print("=" * 60)

        if results_df is not None:
            print(f"\nЛучшая настроенная модель: {results_df.iloc[0]['model']}")
            print(f"ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f}")
        else:
            print("Нет успешно настроенных моделей")

    except Exception as e:
        print(f"Ошибка: {e}")
        raise


if __name__ == "__main__":
    main()