import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

# -------------------------
# 1) CHECK & LOAD MODEL AND SCALERS
# -------------------------
if not os.path.exists("model.h5"):
    st.error("❌ Model file 'model.h5' not found!")
    st.stop()

if not os.path.exists("scaler.pkl") or not os.path.exists("scaler_y.pkl"):
    st.error("❌ Scaler files not found!")
    st.stop()

# Load model
model = load_model("model.h5", compile=False)

# Load scalers using joblib
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# -------------------------
# 2) APP TITLE
# -------------------------
st.title("🌧️ Flood Prediction Using GRU")
st.markdown("Enter the values for the following parameters to predict the flood percentage:")

# -------------------------
# 3) USER INPUTS
# -------------------------
Rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=50.0)
Relative_Humidity = st.number_input("Relative Humidity (%)", min_value=0.0, value=80.0)
Pressure = st.number_input("Pressure (hPa)", min_value=900.0, value=1010.0)
Wind_speed = st.number_input("Wind speed (m/s)", min_value=0.0, value=5.0)
Wind_direction = st.number_input("Wind direction (degrees)", min_value=0.0, value=90.0)
Temperature = st.number_input("Temperature (°C)", min_value=-10.0, value=28.0)
Snowfall = st.number_input("Snowfall (mm)", min_value=0.0, value=0.0)
Snow_depth = st.number_input("Snow depth (cm)", min_value=0.0, value=0.0)
Shortwave = st.number_input("Short-wave irradiation", min_value=0.0, value=200.0)
POONDI = st.number_input("POONDI Reservoir level", min_value=0.0, value=50.0)
CHOLAVARAM = st.number_input("CHOLAVARAM Reservoir level", min_value=0.0, value=40.0)
REDHILLS = st.number_input("REDHILLS Reservoir level", min_value=0.0, value=30.0)
CHEM = st.number_input("CHEMBARAMBAKKAM Reservoir level", min_value=0.0, value=35.0)

# -------------------------
# 4) PREDICTION
# -------------------------
if st.button("Predict Flood %"):
    # Prepare input
    x = np.array([[Rainfall, Relative_Humidity, Pressure, Wind_speed,
                   Wind_direction, Temperature, Snowfall, Snow_depth,
                   Shortwave, POONDI, CHOLAVARAM, REDHILLS, CHEM]])
    
    # Scale input features
    x_scaled = scaler_X.transform(x)
    x_scaled = x_scaled.reshape(1, 1, x_scaled.shape[1])
    
    # Predict
    pred_scaled = model.predict(x_scaled)[0][0]
    
    # Inverse transform to original scale
    pred = scaler_y.inverse_transform(np.array([[pred_scaled]]))[0][0]
    
    # Clip negative values
    pred = max(0, pred)
    
    st.success(f"🌊 Predicted Flood Percent: {pred:.2f}%")

