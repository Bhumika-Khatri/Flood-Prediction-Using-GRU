from tensorflow.keras.models import load_model
import pickle
import numpy as np
import streamlit as st

# -------------------------
# Load model and scaler
# -------------------------
model = load_model("model.h5", compile=False)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------
# Streamlit App
# -------------------------
st.title("🌧️ Flood Prediction Using GRU")

# User Inputs
Rainfall = st.number_input("Rainfall", min_value=0.0, value=0.0)
Relative_Humidity = st.number_input("Relative Humidity", min_value=0.0, value=0.0)
Pressure = st.number_input("Pressure", min_value=0.0, value=0.0)
Wind_speed = st.number_input("Wind speed", min_value=0.0, value=0.0)
Wind_direction = st.number_input("Wind direction", min_value=0.0, value=0.0)
Temperature = st.number_input("Temperature", min_value=0.0, value=0.0)
Snowfall = st.number_input("Snowfall", min_value=0.0, value=0.0)
Snow_depth = st.number_input("Snow depth", min_value=0.0, value=0.0)
Shortwave = st.number_input("Short-wave irradiation", min_value=0.0, value=0.0)
POONDI = st.number_input("POONDI", min_value=0.0, value=0.0)
CHOLAVARAM = st.number_input("CHOLAVARAM", min_value=0.0, value=0.0)
REDHILLS = st.number_input("REDHILLS", min_value=0.0, value=0.0)
CHEM = st.number_input("CHEMBARAMBAKKAM", min_value=0.0, value=0.0)

# Predict button
if st.button("Predict Flood %"):
    # Prepare input
    x = np.array([[Rainfall, Relative_Humidity, Pressure, Wind_speed,
                   Wind_direction, Temperature, Snowfall, Snow_depth,
                   Shortwave, POONDI, CHOLAVARAM, REDHILLS, CHEM]])

    # Scale and reshape
    x_scaled = scaler.transform(x)
    x_scaled = x_scaled.reshape(1, 1, x_scaled.shape[1])

    # Predict
    pred = model.predict(x_scaled)[0][0]

    # Clip negative and NaN predictions
    if np.isnan(pred) or pred < 0:
        pred = 0.0

    st.success(f"🌊 Predicted Flood Percent: {pred:.2f}%")
