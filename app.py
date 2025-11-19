from tensorflow.keras.models import load_model
import pickle
import numpy as np
import streamlit as st

# -------------------------
# Load model & scalers
# -------------------------
model = load_model("model.h5", compile=False)

with open("scaler_X.pkl", "rb") as f:
    scaler_X = pickle.load(f)

with open("scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

# Title
st.title("🌧️ Flood Prediction Using GRU")

# User Inputs
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

# Predict button
if st.button("Predict Flood %"):
    x = np.array([[Rainfall, Relative_Humidity, Pressure, Wind_speed,
                   Wind_direction, Temperature, Snowfall, Snow_depth,
                   Shortwave, POONDI, CHOLAVARAM, REDHILLS, CHEM]])

    # Scale X inputs
    x_scaled = scaler_X.transform(x)
    x_scaled = x_scaled.reshape(1, 1, x_scaled.shape[1])

    # Predict (scaled output)
    pred_scaled = model.predict(x_scaled)[0][0]

    # Convert back to REAL flood percent
    pred_real = scaler_y.inverse_transform([[pred_scaled]])[0][0]

    # Avoid negative values
    pred_real = max(0, pred_real)

    st.success(f"🌊 Predicted Flood Percent: {pred_real:.2f}%")
