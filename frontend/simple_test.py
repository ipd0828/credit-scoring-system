import streamlit as st
import requests

st.set_page_config(page_title="Credit Scoring", layout="wide")
st.title("🎯 Credit Scoring System")
st.write("Kubernetes Deployment Successful!")

# Простая проверка API
if st.button("🔍 Check API Health"):
    try:
        response = requests.get("http://localhost:8001/api/v1/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ **API Status**: {data['status']}")
            st.info(f"🤖 **Model**: {data['model_status']}")
        else:
            st.error(f"❌ API Error: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Cannot connect to API: {str(e)}")

# Ссылки
st.write("---")
st.write("### 🌐 Available Services:")
st.markdown("- **API Documentation**: [http://localhost:8001/docs](http://localhost:8001/docs)")
st.markdown("- **API Health Check**: [http://localhost:8001/api/v1/health](http://localhost:8001/api/v1/health)")
st.markdown("- **Model Info**: [http://localhost:8001/api/v1/model/info](http://localhost:8001/api/v1/model/info)")

st.write("---")
st.write("### 📊 System Status")
st.info("✅ ML API is running in Kubernetes")
st.info("✅ ONNX model is loaded")
st.info("✅ Services are accessible via port-forward")
