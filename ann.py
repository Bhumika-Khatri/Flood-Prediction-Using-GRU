import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import joblib
import streamlit as st

# -----------------------
# 1) LOAD DATA
# -----------------------
df = pd.read_csv("Flood_Prediction.csv")

# Separate features and target
X = df.drop("flood_percent", axis=1).values
y = df["flood_percent"].values

# -----------------------
# 2) SCALE FEATURES
# -----------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------
# 3) TRAIN-TEST SPLIT
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# GRU needs 3D input → (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# -----------------------
# 4) BUILD GRU MODEL
# -----------------------
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(1, X_train.shape[2])),
    Dropout(0.2),
    GRU(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='relu')  # Ensure output is non-negative
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# -----------------------
# 5) TRAIN MODEL
# -----------------------
history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# -----------------------
# 6) EVALUATE TEST SET
# -----------------------
y_pred_test = model.predict(X_test).flatten()
y_pred_test = np.maximum(y_pred_test, 0)  # Clip negative predictions

# Metrics
r2 = r2_score(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)

print("\n===== MODEL METRICS =====")
print("R2 Score:", r2)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)

# -----------------------
# 7) FUTURE PREDICTIONS (CUSTOM YEARS)
# -----------------------
future_years = [2026, 2027, 2028, 2029, 2030]
window = 10
past_data = X[-window:]  # last 10 rows of features

# Trend for every feature
feature_trend = (past_data[-1] - past_data[0]) / window

# Start with last available real data row
future_input = X[-1].copy()
future_predictions = []

for year in future_years:
    future_input = future_input + feature_trend
    scaled = scaler.transform([future_input]).reshape(1, 1, len(future_input))
    pred_future = model.predict(scaled)[0][0]
    pred_future = max(0, pred_future)  # Clip negative predictions
    future_predictions.append(pred_future)

print("\n===== FUTURE FLOOD PREDICTIONS =====")
for yr, p in zip(future_years, future_predictions):
    print(f"{yr}: {p}")

# -----------------------
# 8) PLOT: Actual vs Predicted
# -----------------------
plt.figure(figsize=(10,5))
plt.plot(y_test[:100], label="Actual Flood %", linewidth=3)
plt.plot(y_pred_test[:100], label="Predicted Flood %", linestyle="dashed")
plt.title("Actual vs Predicted Flood %")
plt.xlabel("Sample")
plt.ylabel("Flood %")
plt.legend()
plt.grid()
plt.show()

# -----------------------
# 9) PLOT: Training Loss (Smoothed)
# -----------------------
def smooth_curve(points, factor=0.9):
    smoothed = []
    last = points[0]
    for p in points:
        smoothed_val = last * factor + p * (1 - factor)
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

train_loss_smooth = smooth_curve(history.history['loss'])
val_loss_smooth = smooth_curve(history.history['val_loss'])

plt.figure(figsize=(10,5))
plt.plot(train_loss_smooth, label='Training Loss (Smoothed)', linewidth=3)
plt.plot(val_loss_smooth, label='Validation Loss (Smoothed)', linewidth=3, linestyle='dashed')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------
# 10) SAVE MODEL & SCALER FOR STREAMLIT
# -----------------------
model.save("model.h5", save_format="h5", include_optimizer=False)
joblib.dump(scaler, "scaler.pkl")
print("\nSaved model.h5 and scaler.pkl successfully!")

# -----------------------
# 11) STREAMLIT USER INPUT PREDICTION
# -----------------------
st.title("🌧️ Flood Prediction Using GRU")

# User inputs
Rainfall = st.number_input("Rainfall", 0.0)
Relative_Humidity = st.number_input("Relative Humidity", 0.0)
Pressure = st.number_input("Pressure", 0.0)
Wind_speed = st.number_input("Wind speed", 0.0)
Wind_direction = st.number_input("Wind direction", 0.0)
Temperature = st.number_input("Temperature", 0.0)
Snowfall = st.number_input("Snowfall", 0.0)
Snow_depth = st.number_input("Snow depth", 0.0)
Shortwave = st.number_input("Short-wave irradiation", 0.0)
POONDI = st.number_input("POONDI", 0.0)
CHOLAVARAM = st.number_input("CHOLAVARAM", 0.0)
REDHILLS = st.number_input("REDHILLS", 0.0)
CHEM = st.number_input("CHEMBARAMBAKKAM", 0.0)

# Predict button
if st.button("Predict Flood %"):
    x = np.array([[Rainfall, Relative_Humidity, Pressure, Wind_speed,
                   Wind_direction, Temperature, Snowfall, Snow_depth,
                   Shortwave, POONDI, CHOLAVARAM, REDHILLS, CHEM]])

    x_scaled = scaler.transform(x)
    x_scaled = x_scaled.reshape(1, 1, x_scaled.shape[1])

    pred_input = model.predict(x_scaled)[0][0]
    pred_input = max(0, pred_input)  # Clip negative predictions

    st.success(f"🌊 Predicted Flood Percent: {pred_input:.2f}%")

