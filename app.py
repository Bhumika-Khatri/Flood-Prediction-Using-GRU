import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

# -------------------------
# 1) CHECK & LOAD MODEL AND SCALER
# -------------------------
if not os.path.exists("model.h5"):
    st.error("❌ Model file 'model.h5' not found!")
    st.stop()

if not os.path.exists("scaler.pkl"):
    st.error("❌ Scaler file 'scaler.pkl' not found!")
    st.stop()

model = load_model("model.h5", compile=False)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------
# 2) APP TITLE
# -------------------------
st.title("🌧️ Flood Prediction Using GRU")

# -------------------------
# 3) USER INPUTS
# -------------------------
Rainfall = st.number_input("Rainfall")
Relative_Humidity = st.number_input("Relative Humidity")
Pressure = st.number_input("Pressure")
Wind_speed = st.number_input("Wind speed")
Wind_direction = st.number_input("Wind direction")
Temperature = st.number_input("Temperature")
Snowfall = st.number_input("Snowfall")
Snow_depth = st.number_input("Snow depth")
Shortwave = st.number_input("Short-wave irradiation")
POONDI = st.number_input("POONDI")
CHOLAVARAM = st.number_input("CHOLAVARAM")
REDHILLS = st.number_input("REDHILLS")
CHEM = st.number_input("CHEMBARAMBAKKAM")

# -------------------------
# 4) PREDICTION
# -------------------------
if st.button("Predict Flood %"):
    x = np.array([[Rainfall, Relative_Humidity, Pressure, Wind_speed,
                   Wind_direction, Temperature, Snowfall, Snow_depth,
                   Shortwave, POONDI, CHOLAVARAM, REDHILLS, CHEM]])
    
    # Scale input features
    x_scaled = scaler.transform(x)
    x_scaled = x_scaled.reshape(1, 1, x_scaled.shape[1])
    
    # Predict
    pred = model.predict(x_scaled)[0][0]
    
    # Clip negative values
    pred = max(0, pred)
    
    st.success(f"🌊 Predicted Flood Percent: {pred:.2f}%")

    st.success(f"🌊 Predicted Flood Percent: {pred_real:.2f}%")
