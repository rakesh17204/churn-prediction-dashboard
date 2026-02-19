"""
Streamlit App: Interactive Dashboard for Churn Prediction
"""
import streamlit as st
from src.predict import Predictor
import pandas as pd

# Title
st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
st.title("📞 Customer Churn Prediction System")
st.markdown("Predict whether a customer will churn based on their profile.")

# Sidebar
with st.sidebar:
    st.header("👤 Customer Input")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.checkbox("Senior Citizen")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 120.0, 79.85)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 958.10)

# Prepare input
input_data = {
    "gender": gender,
    "SeniorCitizen": int(senior_citizen),
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

# Submit button
if st.button("🔍 Predict Churn Risk"):
    with st.spinner("Analyzing customer profile..."):
        try:
            predictor = Predictor()
            result = predictor.predict(input_data)

            # Display result
            st.subheader("📊 Prediction Result")
            if result["churn_risk"] == "Yes":
                st.error(f"🚨 HIGH CHURN RISK: {result['probability']:.1%}")
            else:
                st.success(f"✅ LOW CHURN RISK: {result['probability']:.1%}")

            st.write(f"**Recommendation:** {result['recommendation']}")

        except Exception as e:
            st.warning(f"Could not process input: {e}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Python, XGBoost, and Streamlit")
