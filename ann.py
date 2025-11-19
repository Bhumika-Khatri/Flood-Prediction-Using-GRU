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


# 1) LOAD DATA
df = pd.read_csv("Flood_Prediction.csv")

# Separate features and target
X = df.drop("flood_percent", axis=1).values
y = df["flood_percent"].values.reshape(-1, 1)   # <-- reshape for scaler


# 2) SCALE FEATURES
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)  # <-- target scaling FIX


# 3) TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# GRU needs 3D input → (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


# 4) BUILD GRU MODEL
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(1, X_train.shape[2])),
    Dropout(0.2),
    GRU(64),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # <-- REMOVE RELU (output in scaled space)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()


# 5) TRAIN MODEL
history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 6) EVALUATE TEST SET
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # <-- FIXED
y_test_real = scaler_y.inverse_transform(y_test)    # <-- FIXED

r2 = r2_score(y_test_real, y_pred)
mse = mean_squared_error(y_test_real, y_pred)
rmse = np.sqrt(mse)
mae2 = mean_absolute_error(y_test_real, y_pred)

print("\n===== MODEL METRICS =====")
print("R2 Score:", r2)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae2)


# 7) SAVE MODEL & SCALERS
model.save("model.h5", include_optimizer=False)
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

print("\nSaved model + scalers successfully!")
