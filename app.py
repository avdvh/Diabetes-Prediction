import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler only once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("diabetes_nn_model-2.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Streamlit UI
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🏥 AI-Powered Diabetes Prediction")
st.write("**Enter medical details below (with reference value ranges):**")

# User inputs with value ranges
with st.form("user_input_form"):
    age = st.number_input("Age (10 - 100 years)", min_value=10, max_value=100, value=50, help="Typical range: 18-90 years")
    bmi = st.number_input("BMI (10 - 50 kg/m²)", min_value=10.0, max_value=50.0, value=25.0, help="Normal: 18.5 - 24.9, Overweight: 25 - 29.9")
    glucose = st.number_input("Glucose Level (50 - 200 mg/dL)", min_value=50.0, max_value=200.0, value=100.0, help="Normal: <140, Pre-Diabetes: 140-199, Diabetes: >200")
    blood_pressure = st.number_input("Blood Pressure (50 - 200 mmHg)", min_value=50, max_value=200, value=80, help="Normal: <120/80, High: >140/90")
    hba1c = st.number_input("HbA1c (3 - 15%)", min_value=3.0, max_value=15.0, value=5.0, help="Normal: <5.7%, Pre-Diabetes: 5.7-6.4%, Diabetes: >6.5%")
    ldl = st.number_input("LDL Cholesterol (50 - 200 mg/dL)", min_value=50.0, max_value=200.0, value=100.0, help="Optimal: <100, High: >160")
    hdl = st.number_input("HDL Cholesterol (20 - 100 mg/dL)", min_value=20.0, max_value=100.0, value=50.0, help="Healthy: >60, Risk: <40")
    triglycerides = st.number_input("Triglycerides (50 - 500 mg/dL)", min_value=50.0, max_value=500.0, value=150.0, help="Normal: <150, High: >200")

    submitted = st.form_submit_button("🔍 Predict Diabetes Risk")

if submitted:
    # Convert to numpy array
    features = np.array([[age, bmi, glucose, blood_pressure, hba1c, ldl, hdl, triglycerides]])

    # Scale input data
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = model.predict(features_scaled)[0][0]

    st.subheader("📊 Prediction Result:")
    if prediction > 0.5:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    # Show confidence score
    st.write(f"**Confidence Score:** {prediction:.2%}")
