import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import joblib
import os

# -------------------------
# 1) CHECK & LOAD MODEL WEIGHTS AND SCALER
# -------------------------
if not os.path.exists("model.h5"):
    st.error("❌ Model weights file 'model.h5' not found!")
    st.stop()

if not os.path.exists("scaler.pkl"):
    st.error("❌ Scaler file 'scaler.pkl' not found!")
    st.stop()

# -------------------------
# 2) LOAD SCALER
# -------------------------
scaler = joblib.load("scaler.pkl")  # your single scaler.pkl file

# -------------------------
# 3) RECREATE THE MODEL ARCHITECTURE
# -------------------------
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(1, 13)),
    Dropout(0.2),
    GRU(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

# Load the saved weights
model.load_weights("model.h5")

# -------------------------
# 4) APP TITLE
# -------------------------
st.title("🌧️ Flood Prediction Using GRU")
st.markdown("Enter the values for the following parameters to predict the flood percentage:")

# -------------------------
# 5) USER INPUTS
# -------------------------
Rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=50.0)
Relative_Humidity = st.number_input("Relative Humidity (%)", min_value=0.0, value=80.0)
Pressure = st.number_input("Pressure (hPa)", min_value=900.0, value=1010.0)
Wind_speed = st.number_input("Wind speed (m/s)", min_value=0.0, value=5.0)
Wind_direction = st.number_input("Wind direction (degrees)", min_value=0.0, value=90.0)
Temperature = st.number_input("Temperature (K)", min_value=-10.0, value=28.0)
Snowfall = st.number_input("Snowfall (mm)", min_value=0.0, value=0.0)
Snow_depth = st.number_input("Snow depth (cm)", min_value=0.0, value=0.0)
Shortwave = st.number_input("Short-wave irradiation(Wh/m² (watt-hours per square meter)", min_value=0.0, value=200.0)
POONDI = st.number_input("POONDI Reservoir level million cubic feet (MCFT)", min_value=0.0, value=50.0)
CHOLAVARAM = st.number_input("CHOLAVARAM Reservoir level million cubic feet (MCFT)", min_value=0.0, value=40.0)
REDHILLS = st.number_input("REDHILLS Reservoir level million cubic feet (MCFT)", min_value=0.0, value=30.0)
CHEM = st.number_input("CHEMBARAMBAKKAM Reservoir level million cubic feet (MCFT)", min_value=0.0, value=35.0)

# -------------------------
# 6) PREDICTION
# -------------------------
if st.button("Predict Flood %"):
    # Prepare input
    x = np.array([[Rainfall, Relative_Humidity, Pressure, Wind_speed,
                   Wind_direction, Temperature, Snowfall, Snow_depth,
                   Shortwave, POONDI, CHOLAVARAM, REDHILLS, CHEM]])
    
    # Scale input features
    x_scaled = scaler.transform(x)
    x_scaled = x_scaled.reshape(1, 1, x_scaled.shape[1])  # GRU expects 3D input
    
    # Predict
    pred_scaled = model.predict(x_scaled)[0][0]
    
    # If target was scaled, you can inverse transform here
    # pred = scaler.inverse_transform([[pred_scaled]])[0][0]  # only if needed
    
    # Clip negative values
    pred = max(0, pred_scaled)
    
    st.success(f"🌊 Predicted Flood Percent: {pred:.2f}%")

