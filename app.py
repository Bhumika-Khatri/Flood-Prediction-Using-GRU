import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

st.title("🌧️ Flood Percentage Prediction App")
st.write("Enter weather & reservoir parameters to predict flood percentage.")

# ------------------- LOAD MODEL -----------------------
model = load_model("model.h5")  # <-- change name if needed

# Load scaler if your training used MinMaxScaler
try:
    scaler = joblib.load("scaler.pkl")
except:
    scaler = None

# ------------------- USER INPUT SECTION -------------------

st.subheader("Enter Input Values")

col1, col2 = st.columns(2)

with col1:
    Rainfall = st.number_input("Rainfall (mm)", min_value=0.0)
    Relative_Humidity = st.number_input("Relative Humidity (%)", min_value=0.0)
    Pressure = st.number_input("Pressure (hPa)", min_value=0.0)
    Wind_speed = st.number_input("Wind speed (m/s)", min_value=0.0)
    Wind_direction = st.number_input("Wind direction (degrees)", min_value=0.0)

with col2:
    Temperature = st.number_input("Temperature (°C)", min_value=-50.0)
    Snowfall = st.number_input("Snowfall (mm)", min_value=0.0)
    Snow_depth = st.number_input("Snow depth (cm)", min_value=0.0)
    Short_wave = st.number_input("Short-wave irradiation", min_value=0.0)

# Reservoir levels (numeric)
POONDI = st.number_input("POONDI", min_value=0.0)
CHOLAVARAM = st.number_input("CHOLAVARAM", min_value=0.0)
REDHILLS = st.number_input("REDHILLS", min_value=0.0)
CHEMBARAMBAKKAM = st.number_input("CHEMBARAMBAKKAM", min_value=0.0)

# ------------------- PREDICT BUTTON -------------------

if st.button("Predict Flood %"):
    
    input_data = pd.DataFrame([{
        "Rainfall": Rainfall,
        "Relative Humidity": Relative_Humidity,
        "Pressure": Pressure,
        "Wind speed": Wind_speed,
        "Wind direction": Wind_direction,
        "Temperature": Temperature,
        "Snowfall": Snowfall,
        "Snow depth": Snow_depth,
        "Short-wave irradiation\t": Short_wave,
        "POONDI": POONDI,
        "CHOLAVARAM": CHOLAVARAM,
        "REDHILLS": REDHILLS,
        "CHEMBARAMBAKKAM": CHEMBARAMBAKKAM
    }])

    # Apply scaler only if it exists
    if scaler:
        input_scaled = scaler.transform(input_data)
    else:
        input_scaled = input_data

    # Prediction
    pred = model.predict(input_scaled)
    flood_percent = float(pred[0][0])

    st.success(f"🌊 Predicted Flood Percentage: **{flood_percent:.2f}%**")
