"""Streamlit frontend application for credit scoring."""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Конфигурация страницы
st.set_page_config(
    page_title="Система Кредитного Скоринга",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Конфигурация API
API_BASE_URL = "http://localhost:8000/api/v1"

# Пользовательский CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .approved {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .rejected {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Проверить доступность API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def make_prediction(request_data):
    """Выполнить запрос предсказания к API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=request_data,
            timeout=30
        )
        return response.json(), response.status_code
    except Exception as e:
        return {"error": str(e)}, 500

def get_model_info():
    """Получить информацию о модели из API."""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=10)
        return response.json(), response.status_code
    except:
        return {"error": "API недоступен"}, 500

def get_prediction_stats():
    """Получить статистику предсказаний из API."""
    try:
        response = requests.get(f"{API_BASE_URL}/predictions/stats", timeout=10)
        return response.json(), response.status_code
    except:
        return {"error": "API недоступен"}, 500

def main():
    """Основная функция приложения."""
    
    # Заголовок
    st.markdown('<h1 class="main-header">Система Кредитного Скоринга</h1>', unsafe_allow_html=True)
    
    # Проверка состояния API
    if not check_api_health():
        st.error("API недоступен. Убедитесь, что backend сервис запущен.")
        st.stop()
    
    # Боковая панель
    with st.sidebar:
        st.header("Статус Системы")
        
        # Проверка состояния API
        if check_api_health():
            st.success("API Онлайн")
        else:
            st.error("API Офлайн")
        
        # Информация о модели
        model_info, status = get_model_info()
        if status == 200:
            st.info(f"Версия модели: {model_info.get('model_version', 'Неизвестно')}")
            st.info(f"Тип модели: {model_info.get('model_type', 'Неизвестно')}")
        
        # Статистика предсказаний
        stats, status = get_prediction_stats()
        if status == 200:
            st.metric("Всего предсказаний", stats.get('total_predictions', 0))
            st.metric("Процент одобрения", f"{stats.get('approval_rate', 0):.1%}")
    
    # Основной контент
    tab1, tab2, tab3, tab4 = st.tabs(["Предсказание", "Аналитика", "Информация о модели", "Настройки"])
    
    with tab1:
        st.header("Credit Score Prediction")
        
        # Create two columns for input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            annual_inc = st.number_input(
                "Annual Income ($)",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
                help="Enter your annual income"
            )
            
            emp_length = st.selectbox(
                "Employment Length",
                ["< 1 year", "1 year", "2 years", "3 years", "4 years", 
                 "5 years", "6 years", "7 years", "8 years", "9 years", 
                 "10+ years", "n/a"],
                index=5
            )
            
            home_ownership = st.selectbox(
                "Home Ownership",
                ["RENT", "OWN", "MORTGAGE", "OTHER"],
                index=2
            )
        
        with col2:
            st.subheader("Loan Information")
            loan_amnt = st.number_input(
                "Loan Amount ($)",
                min_value=1000.0,
                value=10000.0,
                step=1000.0,
                help="Enter the loan amount you're requesting"
            )
            
            term = st.selectbox(
                "Loan Term",
                ["36 months", "60 months"],
                index=0
            )
            
            purpose = st.selectbox(
                "Loan Purpose",
                ["debt_consolidation", "credit_card", "home_improvement", 
                 "major_purchase", "small_business", "other"],
                index=0
            )
        
        # Credit Information
        st.subheader("Credit Information")
        col3, col4 = st.columns(2)
        
        with col3:
            fico_low = st.slider(
                "FICO Score Range (Low)",
                min_value=300,
                max_value=850,
                value=700,
                step=10
            )
            
            fico_high = st.slider(
                "FICO Score Range (High)",
                min_value=300,
                max_value=850,
                value=750,
                step=10
            )
            
            dti = st.slider(
                "Debt-to-Income Ratio (%)",
                min_value=0.0,
                max_value=100.0,
                value=15.5,
                step=0.1
            )
        
        with col4:
            revol_util = st.slider(
                "Revolving Utilization (%)",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                step=1.0
            )
            
            inq_last_6mths = st.number_input(
                "Inquiries in Last 6 Months",
                min_value=0,
                value=2,
                step=1
            )
            
            delinq_2yrs = st.number_input(
                "Delinquencies in Last 2 Years",
                min_value=0,
                value=0,
                step=1
            )
        
        # Prediction button
        if st.button("🔮 Predict Credit Score", type="primary", use_container_width=True):
            # Prepare request data
            request_data = {
                "annual_inc": annual_inc,
                "emp_length": emp_length,
                "home_ownership": home_ownership,
                "loan_amnt": loan_amnt,
                "term": term,
                "purpose": purpose,
                "fico_range_low": fico_low,
                "fico_range_high": fico_high,
                "dti": dti,
                "revol_util": revol_util,
                "inq_last_6mths": inq_last_6mths,
                "delinq_2yrs": delinq_2yrs,
                "pub_rec": 0
            }
            
            # Make prediction
            with st.spinner("Making prediction..."):
                result, status_code = make_prediction(request_data)
            
            if status_code == 200 and result.get("success"):
                prediction = result["prediction"]
                
                # Display prediction result
                if prediction["prediction"] == 0:
                    st.markdown(
                        f'<div class="prediction-result approved">'
                        f'<h2>✅ Loan Approved!</h2>'
                        f'<p><strong>Confidence:</strong> {prediction["confidence"].title()}</p>'
                        f'<p><strong>Risk Score:</strong> {prediction["risk_score"]:.1f}/100</p>'
                        f'<p><strong>Recommended Amount:</strong> ${prediction.get("recommended_amount", loan_amnt):,.2f}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-result rejected">'
                        f'<h2>❌ Loan Rejected</h2>'
                        f'<p><strong>Confidence:</strong> {prediction["confidence"].title()}</p>'
                        f'<p><strong>Risk Score:</strong> {prediction["risk_score"]:.1f}/100</p>'
                        f'<p><strong>Reason:</strong> High risk profile detected</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Display additional metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction Probability", f"{prediction['probability']:.1%}")
                with col2:
                    st.metric("Processing Time", f"{result['processing_time_ms']:.0f}ms")
                with col3:
                    st.metric("Model Version", prediction['model_version'])
                
                # Feature importance chart
                if prediction.get("features_importance"):
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame(
                        list(prediction["features_importance"].items()),
                        columns=["Feature", "Importance"]
                    ).sort_values("Importance", ascending=True)
                    
                    fig = px.bar(
                        importance_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Feature Importance Scores"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
    
    with tab2:
        st.header("Analytics Dashboard")
        
        # Placeholder for analytics
        st.info("📊 Analytics dashboard will be implemented here")
        
        # Sample charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample approval rate chart
            data = pd.DataFrame({
                'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                'Approval Rate': [0.75, 0.78, 0.72, 0.80, 0.76, 0.79]
            })
            
            fig = px.line(data, x='Month', y='Approval Rate', title='Monthly Approval Rate')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sample risk distribution
            risk_data = pd.DataFrame({
                'Risk Level': ['Low', 'Medium', 'High'],
                'Count': [45, 30, 25]
            })
            
            fig = px.pie(risk_data, values='Count', names='Risk Level', title='Risk Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Model Information")
        
        model_info, status = get_model_info()
        if status == 200:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model Version", model_info.get('model_version', 'Unknown'))
                st.metric("Model Type", model_info.get('model_type', 'Unknown'))
                st.metric("Threshold", model_info.get('threshold', 'Unknown'))
            
            with col2:
                st.metric("Features Count", model_info.get('features_count', 'Unknown'))
                st.metric("Last Updated", model_info.get('last_updated', 'Unknown'))
        else:
            st.error("Failed to load model information")
    
    with tab4:
        st.header("Settings")
        
        st.subheader("API Configuration")
        api_url = st.text_input("API Base URL", value=API_BASE_URL)
        
        st.subheader("Model Configuration")
        threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)
        
        if st.button("Save Settings"):
            st.success("Settings saved!")

if __name__ == "__main__":
    main()
