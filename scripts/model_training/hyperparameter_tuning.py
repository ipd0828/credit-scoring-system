"""
Скрипт для подбора гиперпараметров моделей кредитного скоринга.

Этот скрипт выполняет:
1. Загрузку обработанных данных
2. Определение сеток параметров для различных моделей
3. Поиск лучших гиперпараметров с помощью GridSearchCV
4. Оценку качества настроенных моделей
5. Сохранение лучших моделей
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple
import joblib
import warnings
from tqdm import tqdm

# Импорты для машинного обучения
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score
)

# Импорты для визуализации
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


def load_processed_data(data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Загружает обработанные данные.
    
    Args:
        data_dir: Папка с обработанными данными
    
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    data_path = Path(data_dir)
    
    print("Загрузка обработанных данных...")
    
    # Загружаем данные
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


def create_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Создает препроцессор для обработки признаков.
    
    Args:
        X_train: Обучающие данные для определения типов признаков
    
    Returns:
        ColumnTransformer: Настроенный препроцессор
    """
    print("\nСоздание препроцессора...")
    
    # Определяем типы признаков
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ]
    )
    
    print(f"Препроцессор создан для {len(numeric_features)} числовых и {len(categorical_features)} категориальных признаков")
    
    return preprocessor


def define_parameter_grids() -> List[Dict[str, Any]]:
    """
    Определяет упрощенные сетки параметров для логистической регрессии и случайного леса.
    
    Returns:
        List[Dict]: Список словарей с параметрами для GridSearchCV
    """
    print("\nОпределение упрощенных сеток параметров...")
    
    param_grids = [
        # Logistic Regression - упрощенная сетка
        {
            'classifier': [LogisticRegression(max_iter=1000, random_state=42)],
            'classifier__C': [0.1, 1, 10],  # Уменьшено с 5 до 3 значений
            'classifier__penalty': ['l2'],  # Только L2 регуляризация
            'classifier__solver': ['liblinear']  # Только liblinear
        },
        
        # Random Forest - упрощенная сетка
        {
            'classifier': [RandomForestClassifier(random_state=42)],
            'classifier__n_estimators': [50, 100],  # Уменьшено с 3 до 2 значений
            'classifier__max_depth': [10, 15],  # Уменьшено с 4 до 2 значений
            'classifier__min_samples_split': [2, 5],  # Уменьшено с 3 до 2 значений
            'classifier__min_samples_leaf': [1, 2]  # Уменьшено с 3 до 2 значений
        }
    ]
    
    print(f"Создано {len(param_grids)} упрощенных сеток параметров:")
    for i, grid in enumerate(param_grids):
        classifier_name = grid['classifier'][0].__class__.__name__
        print(f"  {i+1}. {classifier_name}")
    
    return param_grids


def perform_grid_search(X_train: pd.DataFrame, y_train: pd.Series, 
                       preprocessor: ColumnTransformer, 
                       param_grids: List[Dict[str, Any]], 
                       cv: int = 2, n_jobs: int = -1) -> Dict[str, Any]:
    """
    Выполняет поиск по сетке параметров.
    
    Args:
        X_train: Обучающие признаки
        y_train: Обучающая целевая переменная
        preprocessor: Препроцессор
        param_grids: Список сеток параметров
        cv: Количество фолдов для кросс-валидации
        n_jobs: Количество параллельных процессов
    
    Returns:
        Dict: Результаты поиска по сетке
    """
    print(f"\n" + "="*60)
    print("ПОИСК ПО СЕТКЕ ПАРАМЕТРОВ")
    print("="*60)
    
    results = {}
    
    for i, param_grid in enumerate(tqdm(param_grids, desc="Поиск параметров", unit="модель")):
        classifier_name = param_grid['classifier'][0].__class__.__name__
        print(f"\nПоиск параметров для {classifier_name}...")
        
        try:
            # Создаем пайплайн
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', param_grid['classifier'][0])
            ])
            
            # Выполняем поиск по сетке
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=n_jobs,
                verbose=1
            )
            
            # Обучаем
            grid_search.fit(X_train, y_train)
            
            # Сохраняем результаты
            results[classifier_name] = {
                'best_estimator': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            print(f"{classifier_name} завершён:")
            print(f"  Лучший AUC: {grid_search.best_score_:.4f}")
            print(f"  Лучшие параметры: {grid_search.best_params_}")
            
        except Exception as e:
            print(f"Ошибка при поиске параметров для {classifier_name}: {e}")
            results[classifier_name] = {'error': str(e)}
    
    return results


def evaluate_tuned_models(results: Dict[str, Any], X_test: pd.DataFrame, 
                         y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
    """
    Оценивает качество настроенных моделей на тестовых данных.
    
    Args:
        results: Результаты поиска по сетке
        X_test: Тестовые признаки
        y_test: Тестовая целевая переменная
    
    Returns:
        Dict: Результаты оценки
    """
    print(f"\n" + "="*60)
    print("ОЦЕНКА НАСТРОЕННЫХ МОДЕЛЕЙ")
    print("="*60)
    
    evaluation_results = {}
    
    for model_name, model_results in results.items():
        if 'error' in model_results:
            print(f"Пропускаем {model_name} из-за ошибки")
            continue
        
        print(f"\nОценка {model_name}...")
        
        try:
            best_model = model_results['best_estimator']
            
            # Предсказания
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Вычисляем метрики
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }
            
            # Кросс-валидация на полных данных
            cv_scores = cross_val_score(best_model, X_test, y_test, cv=5, scoring='roc_auc')
            metrics['cv_auc_mean'] = cv_scores.mean()
            metrics['cv_auc_std'] = cv_scores.std()
            
            evaluation_results[model_name] = {
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_proba,
                'best_params': model_results['best_params'],
                'cv_score': model_results['best_score']
            }
            
            print(f"{model_name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-score: {metrics['f1']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
            
        except Exception as e:
            print(f"Ошибка при оценке {model_name}: {e}")
            evaluation_results[model_name] = {'error': str(e)}
    
    return evaluation_results


def create_hyperparameter_plots(results: Dict[str, Any], output_dir: str = "models/artifacts") -> None:
    """
    Создает графики для анализа гиперпараметров.
    
    Args:
        results: Результаты поиска по сетке
        output_dir: Папка для сохранения
    """
    print("\nСоздание графиков анализа гиперпараметров...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for model_name, model_results in results.items():
        if 'error' in model_results or 'cv_results' not in model_results:
            continue
        
        try:
            cv_results = model_results['cv_results']
            
            # Создаем график зависимости от параметров
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            # Получаем параметры для анализа
            param_names = [key for key in cv_results.keys() if key.startswith('param_')]
            
            if len(param_names) >= 4:
                param_names = param_names[:4]
            else:
                # Дополняем до 4 параметров
                while len(param_names) < 4:
                    param_names.append(None)
            
            for i, param_name in enumerate(param_names):
                if param_name is None:
                    axes[i].remove()
                    continue
                
                ax = axes[i]
                param_values = cv_results[param_name]
                mean_scores = cv_results['mean_test_score']
                
                # Создаем график
                if isinstance(param_values[0], (int, float)):
                    # Числовой параметр
                    unique_params = sorted(set(param_values))
                    param_scores = []
                    for param in unique_params:
                        mask = param_values == param
                        param_scores.append(mean_scores[mask].mean())
                    
                    ax.plot(unique_params, param_scores, 'o-')
                    ax.set_xlabel(param_name.replace('param_', ''))
                    ax.set_ylabel('CV Score')
                    ax.set_title(f'{param_name.replace("param_", "").replace("_", " ").title()}')
                    ax.grid(True, alpha=0.3)
                else:
                    # Категориальный параметр
                    unique_params = list(set(param_values))
                    param_scores = []
                    for param in unique_params:
                        mask = param_values == param
                        param_scores.append(mean_scores[mask].mean())
                    
                    ax.bar(range(len(unique_params)), param_scores)
                    ax.set_xticks(range(len(unique_params)))
                    ax.set_xticklabels(unique_params, rotation=45)
                    ax.set_ylabel('CV Score')
                    ax.set_title(f'{param_name.replace("param_", "").replace("_", " ").title()}')
                    ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'Анализ гиперпараметров: {model_name}', fontsize=16)
            plt.tight_layout()
            
            # Сохраняем график
            plot_path = output_path / f"hyperparameter_analysis_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"График сохранен: {plot_path}")
            
        except Exception as e:
            print(f"Ошибка при создании графика для {model_name}: {e}")


def create_comparison_plot(evaluation_results: Dict[str, Dict[str, Any]], 
                          output_dir: str = "models/artifacts") -> None:
    """
    Создает график сравнения настроенных моделей.
    
    Args:
        evaluation_results: Результаты оценки
        output_dir: Папка для сохранения
    """
    print("\nСоздание графика сравнения настроенных моделей...")
    
    # Подготавливаем данные
    model_names = []
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_values = {metric: [] for metric in metrics}
    
    for model_name, model_results in evaluation_results.items():
        if 'error' not in model_results and 'metrics' in model_results:
            model_names.append(model_name)
            for metric in metrics:
                metric_values[metric].append(model_results['metrics'][metric])
    
    if not model_names:
        print("Нет данных для создания графика сравнения")
        return
    
    # Создаем график
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(model_names, metric_values[metric])
        ax.set_title(f'{metric.upper()}')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, metric_values[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    # Удаляем лишний subplot
    axes[5].remove()
    
    plt.suptitle('Сравнение настроенных моделей', fontsize=16)
    plt.tight_layout()
    
    # Сохраняем график
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path = output_path / "tuned_models_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"График сохранен: {plot_path}")


def save_tuned_models(results: Dict[str, Any], evaluation_results: Dict[str, Dict[str, Any]], 
                     output_dir: str = "models/trained") -> None:
    """
    Сохраняет настроенные модели и результаты.
    
    Args:
        results: Результаты поиска по сетке
        evaluation_results: Результаты оценки
        output_dir: Папка для сохранения
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nСохранение настроенных моделей в {output_path}...")
    
    # Сохраняем каждую настроенную модель
    for model_name, model_results in results.items():
        if 'error' not in model_results and 'best_estimator' in model_results:
            model_path = output_path / f"tuned_{model_name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model_results['best_estimator'], model_path)
            print(f"  {model_name} -> {model_path}")
    
    # Сохраняем результаты оценки
    if evaluation_results:
        results_data = []
        for model_name, model_results in evaluation_results.items():
            if 'error' not in model_results and 'metrics' in model_results:
                row = {'model': model_name}
                row.update(model_results['metrics'])
                row.update(model_results['best_params'])
                results_data.append(row)
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(output_path / "tuned_models_results.csv", index=False)
            print(f"  Результаты -> {output_path / 'tuned_models_results.csv'}")
    
    # Сохраняем лучшую модель
    if evaluation_results:
        best_model_name = max(
            [name for name, results in evaluation_results.items() 
             if 'error' not in results and 'metrics' in results],
            key=lambda x: evaluation_results[x]['metrics']['roc_auc']
        )
        
        if best_model_name in results and 'best_estimator' in results[best_model_name]:
            best_model = results[best_model_name]['best_estimator']
            joblib.dump(best_model, output_path / "best_tuned_model.pkl")
            print(f"  Лучшая модель ({best_model_name}) -> {output_path / 'best_tuned_model.pkl'}")


def print_final_results(evaluation_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Выводит финальные результаты сравнения настроенных моделей.
    
    Args:
        evaluation_results: Результаты оценки
    """
    print("\n" + "="*60)
    print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ НАСТРОЕННЫХ МОДЕЛЕЙ")
    print("="*60)
    
    if not evaluation_results:
        print("Нет результатов для отображения.")
        return
    
    # Создаем DataFrame с результатами
    results_data = []
    for model_name, model_results in evaluation_results.items():
        if 'error' not in model_results and 'metrics' in model_results:
            row = {'model': model_name}
            row.update(model_results['metrics'])
            results_data.append(row)
    
    if not results_data:
        print("Нет валидных результатов для отображения.")
        return
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('roc_auc', ascending=False)
    
    print("\nСравнение настроенных моделей:")
    print(results_df.round(4))
    
    # Выводим лучшую модель
    if 'roc_auc' in results_df.columns:
        best_model = results_df['roc_auc'].idxmax()
        best_auc = results_df.loc[best_model, 'roc_auc']
        print(f"\nЛучшая настроенная модель: {results_df.loc[best_model, 'model']} (ROC-AUC: {best_auc:.4f})")


def main():
    """Основная функция для запуска подбора гиперпараметров."""
    # Загружаем данные
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Создаем препроцессор
    preprocessor = create_preprocessor(X_train)
    
    # Определяем сетки параметров
    param_grids = define_parameter_grids()
    
    # Выполняем поиск по сетке
    grid_search_results = perform_grid_search(X_train, y_train, preprocessor, param_grids)
    
    # Оцениваем настроенные модели
    evaluation_results = evaluate_tuned_models(grid_search_results, X_test, y_test)
    
    # Создаем визуализации
    create_hyperparameter_plots(grid_search_results)
    create_comparison_plot(evaluation_results)
    
    # Сохраняем результаты
    save_tuned_models(grid_search_results, evaluation_results)
    
    # Выводим финальные результаты
    print_final_results(evaluation_results)
    
    print("\n" + "="*60)
    print("ПОДБОР ГИПЕРПАРАМЕТРОВ ЗАВЕРШЕН УСПЕШНО")
    print("="*60)


if __name__ == "__main__":
    main()
