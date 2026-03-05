import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import time
# Load saved files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

def add_bg_video(video_file):
    with open(video_file, "rb") as f:
        data = f.read()
    video_base64 = base64.b64encode(data).decode()

    st.markdown(
    f"""
    <style>
    .stApp {{
        background: transparent;
    }}

    video {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            z-index: -1;}}
    </style>

    <video autoplay muted loop>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
    """,
    unsafe_allow_html=True
)
if "page" not in st.session_state:
    st.session_state.page = "loading"

# -------- Loading Page --------

if st.session_state.page == "loading":
    st.markdown("""
    <style>
    .stApp {
    background-color: black;
    color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center;'>ℋℯ𝒶𝓇𝓉 ℛ𝒾𝓈𝓀 𝒫𝓇ℯ𝒹𝒾𝒸𝓉ℴ𝓇</h1>", unsafe_allow_html=True)

    st.image("logo.gif", width=800)

    st.write("""LOADING...""")
    
    time.sleep(2)
    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.005)
        progress.progress(i + 1)

    st.session_state.page = "welcome"
    st.rerun()

# -------- Welcome Page --------
elif st.session_state.page == "welcome":
    add_bg_video("background.mp4")
    
    st.title("❤️ Welcome to Heart Disease Prediction System")

    st.write("""
    This AI powered system helps predict the **risk of heart disease**
    using patient health parameters.
    """)
    st.session_state.page = "form"
    time.sleep(4)
    st.rerun()


elif st.session_state.page == "form":   
    
    st.title("❤️AI Heart Risk Predictor")
    st.header("Enter Patient Details:")
    add_bg_video("background.mp4")
    age = st.slider("age", 20, 100, 40)
    sex = st.selectbox("sex", ["Male", "Female"])
    chest_pain_type = st.selectbox("chest pain type", 
                               ["Typical angina", "Atypical angina", 
                                "Non-anginal Pain", "Asymptomatic"])
    resting_blood_pressure = st.slider("Resting Blood Pressure", 80, 200, 120)
    cholestoral = st.slider("Cholesterol", 100, 400, 200)
    fasting_blood_sugar = st.selectbox("fasting blood sugar", ["Lower than 120 mg/ml", "Greater than 120 mg/ml"])
    rest_ecg = st.selectbox("Rest ECG", ["ST-T abnormality", "Normal", "Left ventricular hypertrophy"])
    Max_heart_rate = st.slider("Max Heart Rate", 70, 210, 150)
    exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope", ["Downsloping", "Upsloping", "Flat"])
    vessels_colored_by_flourosopy = st.selectbox("Major Vessels (0-3)", ["Two", "Zero", "One", "Three", "Four"])
    thalassemia = st.selectbox("Thalassemia", ["Reversible Defect", "Fixed Defect", "Normal", "No"])

# ----------- PREDICTION ------------

if st.button("Predict Risk"):

    # Create dictionary
    input_dict = {
        "age": age,
        "resting_blood_pressure": resting_blood_pressure,
        "cholestoral": cholestoral,
        "Max_heart_rate": Max_heart_rate,
        "oldpeak": oldpeak,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "fasting_blood_sugar": fasting_blood_sugar,
        "rest_ecg": rest_ecg,
        "exercise_induced_angina": exercise_induced_angina,
        "slope": slope,
        "vessels_colored_by_flourosopy": vessels_colored_by_flourosopy,
        "thalassemia": thalassemia
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Apply same encoding as training
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure same column order
    input_df = input_df[feature_columns]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1] * 100

    st.subheader("🔍 Prediction Result")

    if prediction[0] == 1:
        st.error(f"⚠ High Risk of Heart Disease\n\nRisk Probability: {probability:.2f}%")
    else:
        st.success(f"✅ Low Risk of Heart Disease\n\nRisk Probability: {probability:.2f}%")

    if probability < 30:
        st.info("🟢 Risk Level: Low")
    elif probability < 70:
        st.warning("🟡 Risk Level: Moderate")
    else:
        st.error("🔴 Risk Level: High")