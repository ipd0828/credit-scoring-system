#!/usr/bin/env python3
"""
Обучение всех 4 моделей: LogReg, RandomForest, LightGBM, Neural Network
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Добавляем корневую папку проекта
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Импорт моделей
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def load_training_data():
    """Загрузка обучающих данных."""
    print("=" * 60)
    print("ЗАГРУЗКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
    print("=" * 60)

    data_path = Path("data/processed_fixed")

    X_train = pd.read_csv(data_path / "X_train.csv")
    X_val = pd.read_csv(data_path / "X_val.csv")
    X_test = pd.read_csv(data_path / "X_test.csv")
    y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
    y_val = pd.read_csv(data_path / "y_val.csv").squeeze()
    y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

    print(f"[OK] Данные загружены:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_val:   {X_val.shape}")
    print(f"   X_test:  {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Обучение Logistic Regression."""
    print("\n" + "=" * 40)
    print("ОБУЧЕНИЕ LOGISTIC REGRESSION")
    print("=" * 40)

    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        solver='liblinear'
    )

    print("Обучение модели...")
    model.fit(X_train, y_train)

    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"[OK] Обучение завершено:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")

    # Сохранение
    models_dir = Path("models/trained_final")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "logistic_regression.pkl"
    joblib.dump(model, model_path)
    print(f"   Сохранена: {model_path}")

    return model, accuracy, roc_auc, y_pred, y_proba


def train_random_forest(X_train, y_train, X_test, y_test):
    """Обучение Random Forest."""
    print("\n" + "=" * 40)
    print("ОБУЧЕНИЕ RANDOM FOREST")
    print("=" * 40)

    model = RandomForestClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced_subsample',
        n_jobs=-1
    )

    print("Обучение модели...")
    model.fit(X_train, y_train)

    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"[OK] Обучение завершено:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
    print(f"   Важность признаков:")
    for i, (feature, importance) in enumerate(zip(X_train.columns, model.feature_importances_)):
        print(f"     {i + 1:2d}. {feature}: {importance:.4f}")

    # Сохранение
    models_dir = Path("models/trained_final")
    model_path = models_dir / "random_forest.pkl"
    joblib.dump(model, model_path)
    print(f"   Сохранена: {model_path}")

    return model, accuracy, roc_auc, y_pred, y_proba


def train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    """Обучение LightGBM."""
    print("\n" + "=" * 40)
    print("ОБУЧЕНИЕ LIGHTGBM")
    print("=" * 40)

    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] LightGBM не установлен. Установите: pip install lightgbm")
        return None, 0, 0, None, None

    # Используем sklearn API для LightGBM (более стабильный)
    try:
        from lightgbm import LGBMClassifier
    except:
        print("[ERROR] Не удалось импортировать LGBMClassifier")
        return None, 0, 0, None, None

    print("Обучение модели...")

    # Используем sklearn-совместимый API
    model = LGBMClassifier(
        objective='binary',
        n_estimators=200,
        learning_rate=0.001,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=25122025,
        n_jobs=-1,
        verbose=-1,
        class_weight='balanced'
    )

    # Обучение с валидацией
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"[OK] Обучение завершено:")
    print(f"   Лучшая итерация: {model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators}")
    print(f"   Accuracy:        {accuracy:.4f}")
    print(f"   ROC-AUC:         {roc_auc:.4f}")

    # Сохранение
    models_dir = Path("models/trained_final")
    model_path = models_dir / "lightgbm.pkl"
    joblib.dump(model, model_path)
    print(f"   Сохранена: {model_path}")

    return model, accuracy, roc_auc, y_pred, y_proba


# Для нейронной сети
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch не установлен. Нейронная сеть не будет обучена")


class SimpleNN(nn.Module):
    """Простая нейронная сеть."""

    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
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


def train_neural_network(X_train, y_train, X_val, y_val, X_test, y_test):
    """Обучение нейронной сети."""
    print("\n" + "=" * 40)
    print("ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ")
    print("=" * 40)

    if not TORCH_AVAILABLE:
        print("[WARNING] PyTorch не установлен. Пропускаем нейронную сеть.")
        return None, 0, 0, None, None

    # Масштабирование данных
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # Конвертация в тензоры
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)

    # Создание модели
    model = SimpleNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Обучение
    print("Обучение модели...")
    model.train()
    epochs = 100
    batch_size = 64

    for epoch in range(epochs):
        # Пакетное обучение
        permutation = torch.randperm(X_train_tensor.size()[0])

        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = X_train_tensor[indices]
            batch_y = y_train_tensor[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Логирование каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            print(f"   Эпоха [{epoch + 1}/{epochs}] - Loss: {loss.item():.4f}")

    # Оценка
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        y_proba = outputs.numpy().flatten()
        y_pred = (y_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"[OK] Обучение завершено:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")

    # Сохранение
    models_dir = Path("models/trained_final")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "neural_network.pth"
    torch.save(model.state_dict(), model_path)

    scaler_path = models_dir / "nn_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    print(f"   Модель сохранена: {model_path}")
    print(f"   Scaler сохранен: {scaler_path}")

    return model, accuracy, roc_auc, y_pred, y_proba


def create_confusion_matrix(y_true, y_pred, model_name, output_dir="models/artifacts"):
    """
    Создание и сохранение confusion matrix.

    Args:
        y_true: истинные значения
        y_pred: предсказанные значения
        model_name: имя модели
        output_dir: директория для сохранения
    """
    # Создаем директорию если не существует
    output_path = Path(output_dir) / "confusion_matrices"
    output_path.mkdir(parents=True, exist_ok=True)

    # Создаем confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Создаем красивый график
    plt.figure(figsize=(10, 8))

    # Используем seaborn для тепловой карты
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Good (0)', 'Bad (1)'],
                yticklabels=['Good (0)', 'Bad (1)'])

    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)

    # Добавляем метрики в заголовок
    accuracy = accuracy_score(y_true, y_pred)
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Информация о матрице
    info_text = f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}"
    plt.figtext(0.02, 0.02, info_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    # Сохраняем изображение
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    filepath = output_path / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Confusion matrix сохранена: {filepath}")

    # Выводим матрицу в консоль
    print(f"\n  Confusion Matrix для {model_name}:")
    print(f"  {'-' * 40}")
    print(f"  |{'Predicted':^20}|")
    print(f"  {'-' * 40}")
    print(f"  |{'True/Actual':^10}| Good (0) | Bad (1) |")
    print(f"  {'-' * 40}")
    print(f"  | Good (0)   | {cm[0, 0]:^8} | {cm[0, 1]:^7} |")
    print(f"  | Bad (1)    | {cm[1, 0]:^8} | {cm[1, 1]:^7} |")
    print(f"  {'-' * 40}")
    print(f"  TP (True Positive): {cm[1, 1]}")
    print(f"  TN (True Negative): {cm[0, 0]}")
    print(f"  FP (False Positive): {cm[0, 1]}")
    print(f"  FN (False Negative): {cm[1, 0]}")

    return cm, accuracy, precision, recall, f1


def compare_models(results):
    """Сравнение моделей."""
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)

    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Модель': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}",
            'Успешно': 'OK' if metrics['accuracy'] > 0 else 'FAIL'
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # Определение лучшей модели
    if comparison_data:
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        print(f"\n[BEST] ЛУЧШАЯ МОДЕЛЬ: {best_model}")
        print(f"   Accuracy: {results[best_model]['accuracy']:.4f}")
        print(f"   ROC-AUC:  {results[best_model]['roc_auc']:.4f}")

        # Создаем confusion matrix для лучшей модели
        print(f"\n" + "=" * 40)
        print(f"АНАЛИЗ ЛУЧШЕЙ МОДЕЛИ: {best_model}")
        print("=" * 40)

        y_true = results[best_model]['y_true']
        y_pred = results[best_model]['y_pred']

        # Создаем confusion matrix
        cm, acc, prec, rec, f1 = create_confusion_matrix(
            y_true, y_pred, best_model
        )

        # Сохраняем детализированный отчет о лучшей модели
        models_dir = Path("models/trained_final")
        best_model_report = {
            'model_name': best_model,
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'roc_auc': float(results[best_model]['roc_auc']),
            'confusion_matrix': cm.tolist(),
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Сохраняем отчет в JSON
        report_path = models_dir / "best_model_report.json"
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(best_model_report, f, indent=2, ensure_ascii=False)

        print(f"  Детальный отчет сохранен: {report_path}")

        # Сохранение лучшей модели
        best_model_path = Path("models/trained_final/best_model.pkl")
        if best_model == "Logistic Regression":
            joblib.dump(results[best_model]['model'], best_model_path)
        elif best_model == "Random Forest":
            joblib.dump(results[best_model]['model'], best_model_path)
        elif best_model == "LightGBM":
            joblib.dump(results[best_model]['model'], best_model_path)
        elif best_model == "Neural Network":
            torch.save(results[best_model]['model'].state_dict(),
                       Path("models/trained_final/best_model.pth"))

        print(f"   Модель сохранена как: {best_model_path}")

    return comparison_df


def main():
    """Основная функция."""
    print("=" * 60)
    print("ОБУЧЕНИЕ 4 МОДЕЛЕЙ: LogReg, RandomForest, LightGBM, NN")
    print("=" * 60)

    # Загрузка данных
    X_train, X_val, X_test, y_train, y_val, y_test = load_training_data()

    results = {}

    # 1. Logistic Regression
    try:
        print("\n" + "-" * 40)
        print("1. Обучаем Logistic Regression...")
        lr_model, lr_acc, lr_auc, lr_pred, lr_proba = train_logistic_regression(
            X_train, y_train, X_test, y_test
        )
        results['Logistic Regression'] = {
            'model': lr_model,
            'accuracy': lr_acc,
            'roc_auc': lr_auc,
            'y_pred': lr_pred,
            'y_true': y_test,
            'y_proba': lr_proba
        }
    except Exception as e:
        print(f"[ERROR] Ошибка Logistic Regression: {e}")

    # 2. Random Forest
    try:
        print("\n" + "-" * 40)
        print("2. Обучаем Random Forest...")
        rf_model, rf_acc, rf_auc, rf_pred, rf_proba = train_random_forest(
            X_train, y_train, X_test, y_test
        )
        results['Random Forest'] = {
            'model': rf_model,
            'accuracy': rf_acc,
            'roc_auc': rf_auc,
            'y_pred': rf_pred,
            'y_true': y_test,
            'y_proba': rf_proba
        }
    except Exception as e:
        print(f"[ERROR] Ошибка Random Forest: {e}")

    # 3. LightGBM
    try:
        print("\n" + "-" * 40)
        print("3. Обучаем LightGBM...")
        lgb_model, lgb_acc, lgb_auc, lgb_pred, lgb_proba = train_lightgbm(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        if lgb_model:
            results['LightGBM'] = {
                'model': lgb_model,
                'accuracy': lgb_acc,
                'roc_auc': lgb_auc,
                'y_pred': lgb_pred,
                'y_true': y_test,
                'y_proba': lgb_proba
            }
    except Exception as e:
        print(f"[ERROR] Ошибка LightGBM: {e}")

    # 4. Neural Network
    try:
        if TORCH_AVAILABLE:
            print("\n" + "-" * 40)
            print("4. Обучаем Neural Network...")
            nn_model, nn_acc, nn_auc, nn_pred, nn_proba = train_neural_network(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            if nn_model:
                results['Neural Network'] = {
                    'model': nn_model,
                    'accuracy': nn_acc,
                    'roc_auc': nn_auc,
                    'y_pred': nn_pred,
                    'y_true': y_test,
                    'y_proba': nn_proba
                }
    except Exception as e:
        print(f"[ERROR] Ошибка Neural Network: {e}")

    # Сравнение моделей
    if results:
        comparison_df = compare_models(results)

        # Сохранение сравнения
        results_dir = Path("models/trained_final")
        results_dir.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(results_dir / "model_comparison.csv", index=False)

        print(f"\n[OK] Все модели обучены и сохранены в: {results_dir}")

        # Создаем confusion matrix для всех моделей
        print(f"\n" + "=" * 60)
        print("СОЗДАНИЕ CONFUSION MATRIX ДЛЯ ВСЕХ МОДЕЛЕЙ")
        print("=" * 60)

        for model_name, metrics in results.items():
            if 'y_pred' in metrics and 'y_true' in metrics:
                create_confusion_matrix(
                    metrics['y_true'],
                    metrics['y_pred'],
                    model_name
                )
                print()

        # Вывод рекомендаций
        print(f"\n[INFO] РЕКОМЕНДАЦИИ ДЛЯ ПРОДАКШЕНА:")
        print("-" * 40)
        print("1. Для максимальной производительности используйте Logistic Regression")
        print("2. Для максимальной точности используйте лучшую модель из таблицы")
        print("3. Для пакетной обработки используйте ONNX конвертацию")
        print("4. Для реального времени используйте LightGBM или NN")

        print(f"\nConfusion matrices сохранены в: models/artifacts/confusion_matrices/")
    else:
        print("[ERROR] Не удалось обучить ни одну модель")


if __name__ == "__main__":
    main()