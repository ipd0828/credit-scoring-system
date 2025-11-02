"""Streamlit frontend application for credit scoring with custom features."""

import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Конфигурация страницы
st.set_page_config(
    page_title="Система Кредитного Скоринга",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Конфигурация API
API_BASE_URL = "http://localhost:8000"


def check_api_health():
    """Проверить доступность API."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def make_prediction(request_data):
    """Выполнить запрос предсказания к API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/predict", json=request_data, timeout=30
        )
        return response.json(), response.status_code
    except Exception as e:
        return {"error": str(e)}, 500


def get_model_info():
    """Получить информацию о модели из API."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/model/info", timeout=10)
        return response.json(), response.status_code
    except:
        return {"error": "API недоступен"}, 500


def main():
    """Основная функция приложения."""

    # Заголовок
    st.markdown(
        '<h1 class="main-header">🏦 Система Кредитного Скоринга</h1>',
        unsafe_allow_html=True,
    )

    # Проверка состояния API
    if not check_api_health():
        st.error("❌ API недоступен. Убедитесь, что backend сервис запущен.")
        st.stop()

    # Получение информации о модели
    model_info, status = get_model_info()
    if status != 200:
        st.error("❌ Не удалось получить информацию о модели")
        st.stop()

    # Боковая панель
    with st.sidebar:
        st.header("📊 Статус Системы")

        # Проверка состояния API
        if check_api_health():
            st.success("✅ API Онлайн")
        else:
            st.error("❌ API Офлайн")

        # Информация о модели
        if model_info.get("model_loaded"):
            st.info(f"🤖 Модель: {model_info.get('model_type', 'ML Модель')}")
            st.info(f"📊 Признаков: {model_info.get('features_count', 6)}")
            st.info(f"🔄 Версия: {model_info.get('model_version', '1.0.0')}")

            # Отображаем метрики если они есть
            if "accuracy" in model_info:
                st.info(f"🎯 Точность: {model_info.get('accuracy', 0.0):.1%}")
            if "roc_auc" in model_info:
                st.info(f"📈 ROC-AUC: {model_info.get('roc_auc', 0.0):.3f}")
        else:
            st.warning("⚠️ Модель не загружена")

    # Основной контент
    tab1, tab2, tab3 = st.tabs(["🎯 Предсказание", "📊 Аналитика", "🤖 Информация"])

    with tab1:
        st.header("🎯 Предсказание Кредитного Скоринга")

        st.info(
            """
        **Используемые признаки (6):**
        - 💰 Кредитный лимит | 👫 Пол | 💍 Семейное положение  
        - 🎂 Возраст | 💳 Статус платежей | 🎓 Образование
        """
        )

        # Основная информация
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👤 Основная Информация")

            limit_bal = st.number_input(
                "💰 Кредитный лимит (TWD)",
                min_value=10000.0,
                max_value=1000000.0,
                value=150000.0,
                step=1000.0,
                help="Установленный кредитный лимит в тайваньских долларах",
            )

            sex = st.selectbox(
                "👫 Пол", ["Мужской", "Женский"], index=0, help="Пол заявителя"
            )

            age = st.number_input(
                "🎂 Возраст",
                min_value=21,
                max_value=79,
                value=35,
                step=1,
                help="Возраст заявителя",
            )

        with col2:
            st.subheader("🏠 Демографическая Информация")

            marriage = st.selectbox(
                "💍 Семейное положение",
                ["Неизвестно", "Женат/Замужем", "Не женат/Не замужем", "Другое"],
                index=1,
                help="Текущее семейное положение",
            )

            education = st.selectbox(
                "🎓 Уровень образования",
                ["Аспирантура", "Университет", "Средняя школа", "Другое"],
                index=1,
                help="Наивысший уровень образования",
            )

        # Статус платежа
        st.subheader("💳 Статус Погашения")

        pay_new = st.selectbox(
            "Статус погашения текущего месяца",
            options=[-1, 0, 1],
            format_func=lambda x: {
                -1: "✅ Погашено в срок",
                0: "⚠️ Не было платежей",
                1: "❌ Задержка платежа",
            }[x],
            index=0,
            help="Статус погашения кредита за текущий месяц",
        )

        # Преобразование значений для API
        sex_mapping = {"Мужской": 1, "Женский": 2}
        marriage_mapping = {
            "Неизвестно": 0,
            "Женат/Замужем": 1,
            "Не женат/Не замужем": 2,
            "Другое": 3,
        }
        education_mapping = {
            "Аспирантура": 1,
            "Университет": 2,
            "Средняя школа": 3,
            "Другое": 4,
        }

        # Prediction button
        if st.button(
            "🔮 Получить Кредитный Скоринг", type="primary", use_container_width=True
        ):
            # Валидация данных
            errors = []
            if limit_bal < 10000 or limit_bal > 1000000:
                errors.append("❌ Кредитный лимит должен быть между 10,000 и 1,000,000")
            if age < 21 or age > 79:
                errors.append("❌ Возраст должен быть между 21 и 79 годами")

            if errors:
                for error in errors:
                    st.error(error)
                return

            # Prepare request data
            request_data = {
                "limit_bal": float(limit_bal),
                "sex": sex_mapping[sex],
                "marriage_new": marriage_mapping[marriage],
                "age": int(age),
                "pay_new": int(pay_new),
                "education_new": education_mapping[education],
            }

            # Отображение отправляемых данных
            with st.expander("📤 Отправляемые данные", expanded=False):
                st.json(request_data)

            # Make prediction
            with st.spinner("🤖 Анализируем данные..."):
                result, status_code = make_prediction(request_data)

            if status_code == 200 and result.get("success"):
                prediction = result["prediction"]
                processing_time = result["processing_time_ms"]

                # Форматирование времени
                time_display = (
                    f"{processing_time:.0f}мс"
                    if processing_time < 1000
                    else f"{processing_time / 1000:.2f}с"
                )

                # Display prediction result
                if prediction["prediction"] == 0:
                    st.success("### ✅ Кредит Одобрен!")
                    st.info(
                        f"""
                    **Детали решения:**
                    - 🤖 Модель: {prediction["model_version"]}
                    - 🎯 Уверенность: {prediction["confidence"].title()}
                    - 📊 Оценка риска: {prediction["risk_score"]:.1f}/100
                    - 📈 Вероятность возврата: {prediction["probability"]:.1%}
                    - ⏱️ Время анализа: {time_display}
                    """
                    )
                else:
                    st.error("### ❌ Высокий риск дефолта")
                    st.warning(
                        f"""
                    **Детали решения:**
                    - 🤖 Модель: {prediction["model_version"]}
                    - 🎯 Уверенность: {prediction["confidence"].title()}
                    - 📊 Оценка риска: {prediction["risk_score"]:.1f}/100
                    - 📉 Вероятность дефолта: {prediction["probability"]:.1%}
                    - ⏱️ Время анализа: {time_display}
                    - 💡 Рекомендация: Требуется дополнительная проверка
                    """
                    )

                # Feature importance chart
                if prediction.get("features_importance"):
                    st.subheader("📊 Влияние Признаков на Решение")
                    importance_df = pd.DataFrame(
                        list(prediction["features_importance"].items()),
                        columns=["Признак", "Влияние"],
                    ).sort_values("Влияние", ascending=True)

                    # Перевод названий признаков на русский
                    feature_translation = {
                        "limit_bal": "Кредитный лимит",
                        "sex": "Пол",
                        "marriage_new": "Семейное положение",
                        "age": "Возраст",
                        "pay_new": "Статус платежей",
                        "education_new": "Образование",
                    }

                    importance_df["Признак"] = importance_df["Признак"].map(
                        feature_translation
                    )

                    fig = px.bar(
                        importance_df,
                        x="Влияние",
                        y="Признак",
                        orientation="h",
                        title="Вклад признаков в решение модели",
                        color="Влияние",
                        color_continuous_scale="RdYlGn",
                        labels={"Влияние": "Относительная важность", "Признак": ""},
                    )
                    fig.update_layout(
                        showlegend=False,
                        yaxis={"categoryorder": "total ascending"},
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            else:
                error_msg = result.get("error", "Неизвестная ошибка")
                st.error(f"❌ Ошибка анализа: {error_msg}")
                if "Проверьте консоль бэкенда" not in error_msg:
                    st.info(
                        "Проверьте консоль бэкенда для подробной информации об ошибке"
                    )

    with tab2:
        st.header("📊 Аналитика")

        if model_info.get("model_loaded"):
            st.success("✅ Модель успешно загружена и готова к работе")

            # Метрики модели
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🤖 Тип модели", model_info.get("model_type", "ML Модель"))
            with col2:
                st.metric("📊 Признаков", model_info.get("features_count", 6))
            with col3:
                st.metric("🔄 Версия", model_info.get("model_version", "1.0.0"))
            with col4:
                model_status = (
                    "Загружена" if model_info.get("model_loaded") else "Не загружена"
                )
                st.metric("📈 Статус", model_status)

            # Дополнительные метрики если есть
            if any(
                key in model_info
                for key in ["accuracy", "roc_auc", "precision", "recall"]
            ):
                st.subheader("📈 Метрики Производительности")
                metrics_cols = st.columns(4)
                metric_index = 0

                if "accuracy" in model_info:
                    with metrics_cols[metric_index % 4]:
                        st.metric("Accuracy", f"{model_info['accuracy']:.3f}")
                    metric_index += 1

                if "roc_auc" in model_info:
                    with metrics_cols[metric_index % 4]:
                        st.metric("ROC-AUC", f"{model_info['roc_auc']:.3f}")
                    metric_index += 1

                if "precision" in model_info:
                    with metrics_cols[metric_index % 4]:
                        st.metric("Precision", f"{model_info['precision']:.3f}")
                    metric_index += 1

                if "recall" in model_info:
                    with metrics_cols[metric_index % 4]:
                        st.metric("Recall", f"{model_info['recall']:.3f}")
        else:
            st.error("❌ Модель не загружена в системе")

        # Статистика по признакам
        st.subheader("📋 Используемые Признаки")
        features_info = {
            "Признак": [
                "Кредитный лимит",
                "Пол",
                "Семейное положение",
                "Возраст",
                "Статус платежей",
                "Образование",
            ],
            "Диапазон": ["10,000-1,000,000", "1-2", "0-3", "21-79", "-1,0,1", "1-4"],
            "Описание": [
                "Сумма кредита в тайваньских долларах",
                "Пол заявителя (1-Мужской, 2-Женский)",
                "Семейное положение (0-Неизвестно, 1-Женат, 2-Не женат, 3-Другое)",
                "Возраст заявителя в годах",
                "Статус погашения (-1: вовремя, 0: не было, 1: задержка)",
                "Уровень образования (1-Аспирантура, 2-Университет, 3-Средняя школа, 4-Другое)",
            ],
            "Тип": [
                "Числовой",
                "Категориальный",
                "Категориальный",
                "Числовой",
                "Категориальный",
                "Категориальный",
            ],
        }

        features_df = pd.DataFrame(features_info)
        st.dataframe(features_df, use_container_width=True, hide_index=True)

    with tab3:
        st.header("🤖 Информация о Модели")

        if model_info.get("model_loaded"):
            st.subheader("🚀 Основные Характеристики")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Архитектура модели:**")
                st.success(f"```{model_info.get('model_type', 'ML Модель')}```")

                st.write("**Версия модели:**")
                st.info(f"```{model_info.get('model_version', '1.0.0')}```")

                st.write("**Количество признаков:**")
                st.info(f"```{model_info.get('features_count', 6)}```")

            with col2:
                st.write("**Статус модели:**")
                status_color = (
                    "✅ Загружена"
                    if model_info.get("model_loaded")
                    else "❌ Не загружена"
                )
                st.info(f"```{status_color}```")

                st.write("**Последнее обновление:**")
                last_updated = model_info.get(
                    "last_updated", datetime.utcnow().isoformat()
                )
                st.info(f"```{last_updated.split('T')[0]}```")

            # Детальная информация о признаках
            st.subheader("🎯 Используемые Признаки")
            if "features" in model_info:
                for i, feature in enumerate(model_info["features"], 1):
                    st.write(f"{i}. {feature}")
            else:
                st.info("Информация о признаках недоступна")

            # Дополнительная информация если есть
            if any(
                key in model_info for key in ["description", "threshold", "model_class"]
            ):
                st.subheader("📝 Дополнительная Информация")

                if "description" in model_info:
                    st.write("**Описание:**")
                    st.info(model_info["description"])

                if "threshold" in model_info:
                    st.write("**Порог классификации:**")
                    st.info(f"```{model_info['threshold']}```")

                if "model_class" in model_info:
                    st.write("**Класс модели:**")
                    st.info(f"```{model_info['model_class']}```")

        else:
            st.error("### ❌ Модель не загружена")
            st.warning(
                """
            **Возможные причины:**
            - Модель не была загружена при запуске бэкенда
            - Файлы модели отсутствуют или повреждены
            - Возникла ошибка при загрузке модели

            **Решение:**
            Проверьте консоль бэкенда для получения подробной информации об ошибке.
            """
            )


if __name__ == "__main__":
    main()
