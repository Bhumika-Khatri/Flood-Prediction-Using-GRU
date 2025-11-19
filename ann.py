import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# CONFIG
DATA_PATH = "Flood_Prediction.csv"
LOOKBACK = 12
TEST_RATIO = 0.2

# LOAD DATA
df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)
print(df.head())

# TARGET (Flood related automatically)
flood_keywords = ["flood", "rain", "precip", "water", "discharge", "flow", "level"]

candidates = [c for c in df.columns if any(k in c.lower() for k in flood_keywords)]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

target = candidates[0] if candidates else numeric_cols[-1]
print("Selected Target:", target)

# CLEANING
data = df.copy()

# Convert date if present
for col in data.columns:
    if col == target:
        continue
    try:
        tmp = pd.to_datetime(data[col], errors="coerce")
        if tmp.notna().sum() > 0:
            data[col] = tmp
            data = data.sort_values(col)
            break
    except:
        pass

# Remove non-numeric columns
non_numeric = data.select_dtypes(exclude=[np.number]).columns
cleaned = data.drop(columns=non_numeric, errors="ignore")

# Fill missing values
for col in cleaned.columns:
    cleaned[col] = cleaned[col].ffill().fillna(cleaned[col].median())

# Save cleaned dataset
cleaned.to_csv("cleaned_data.csv", index=False)
print("\nSaved cleaned dataset: cleaned_data.csv")
print("\nCleaned shape:", cleaned.shape)

# SCALING FOR GRU/LSTM
feature_cols = [c for c in cleaned.columns if c != target]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_all = scaler_X.fit_transform(cleaned[feature_cols])
y_all = scaler_y.fit_transform(cleaned[[target]])

# MAKE SEQUENCES
def build_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i+lookback])
        ys.append(y[i+lookback])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = build_sequences(X_all, y_all, LOOKBACK)

# TRAIN TEST SPLIT
split = int(len(X_seq) * (1 - TEST_RATIO))

X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

print("\nFinal shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

# -----------------------
# 1) LOAD DATA AGAIN
# -----------------------
df = pd.read_csv("Flood_Prediction.csv")
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

# GRU 3D input
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
    Dense(1)
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
loss, mae = model.evaluate(X_test, y_test, verbose=1)
pred = model.predict(X_test).flatten()

# Metrics
r2 = r2_score(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
mae2 = mean_absolute_error(y_test, pred)

print("\n===== MODEL METRICS =====")
print("R2 Score:", r2)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae2)

# -----------------------
# 7) FUTURE PREDICTIONS
# -----------------------
future_years = [2026, 2027, 2028, 2029, 2030]
window = 10
past_data = X[-window:]
feature_trend = (past_data[-1] - past_data[0]) / window
future_input = X[-1].copy()
future_predictions = []

for year in future_years:
    future_input = future_input + feature_trend
    scaled = scaler.transform([future_input]).reshape(1, 1, len(future_input))
    pred_future = model.predict(scaled)[0][0]
    future_predictions.append(pred_future)

print("\n===== FUTURE FLOOD PREDICTIONS =====")
for yr, p in zip(future_years, future_predictions):
    print(f"{yr}: {p}")

# -----------------------
# 8) PLOT: Actual vs Predicted
# -----------------------
plt.figure(figsize=(10,5))
plt.plot(y_test[:100], label="Actual Flood %", linewidth=3)
plt.plot(pred[:100], label="Predicted Flood %", linestyle="dashed")
plt.title("Actual vs Predicted Flood %")
plt.xlabel("Sample")
plt.ylabel("Flood %")
plt.legend()
plt.grid()
plt.show()

# -----------------------
# 9) PLOT: Training Loss (SMOOTHED)
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


# ============================
# 10) SAVE MODEL & SCALER FOR STREAMLIT
# ============================
import joblib

# Save GRU model
model.save("model.h5", save_format="h5", include_optimizer=False)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

print("\nSaved model.h5 and scaler.pkl successfully!")
