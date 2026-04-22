import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

# ==============================
# 1️⃣ CREATE DATASET (1–20 RANGE)
# ==============================

print("Creating Dataset...")

start_time = datetime(2024, 1, 1, 0, 0, 0)
timestamps = [start_time + timedelta(hours=i) for i in range(2000)]

vehicle_counts = []

for i in range(2000):
    hour = timestamps[i].hour
    day = timestamps[i].weekday()
    
    base = 5
    
    morning_peak = 6 * np.exp(-0.5 * (hour - 9)**2 / 2)
    evening_peak = 8 * np.exp(-0.5 * (hour - 18)**2 / 3)
    
    night_drop = -2 if hour < 5 else 0
    weekend_effect = -3 if day >= 5 else 0
    trend = 0.002 * i
    noise = np.random.normal(0, 1)
    
    traffic = base + morning_peak + evening_peak + night_drop + weekend_effect + trend + noise
    traffic = np.clip(traffic, 1, 20)
    
    vehicle_counts.append(int(round(traffic)))

df = pd.DataFrame({
    "time": timestamps,
    "number_of_vehicles": vehicle_counts
})

df.to_csv("traffic_dataset.csv", index=False)

print("Dataset Created.")
print("Min:", df["number_of_vehicles"].min())
print("Max:", df["number_of_vehicles"].max())

# ==============================
# 2️⃣ PREPARE DATA
# ==============================

data = df.drop("time", axis=1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Save scaler
joblib.dump(scaler, "traffic_scaler.save")
print("Scaler Saved.")

# ==============================
# 3️⃣ CREATE SEQUENCES (10 HOURS)
# ==============================

def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, 10)

# Train Test Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==============================
# 4️⃣ BUILD LSTM MODEL
# ==============================

print("Building Model...")

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(10,1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# ==============================
# 5️⃣ TRAIN MODEL (100 EPOCHS)
# ==============================

print("Training Model...")
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("traffic_rnn_model.h5")
print("Model Saved.")

# ==============================
# 6️⃣ LOAD MODEL & SCALER
# ==============================

print("Loading Model and Scaler...")

loaded_model = load_model("traffic_rnn_model.h5")
loaded_scaler = joblib.load("traffic_scaler.save")

# ==============================
# 7️⃣ PREDICT NEXT HOUR
# ==============================

def predict_next_hour(vehicle_list):
    
    if len(vehicle_list) != 10:
        raise ValueError("Input must contain exactly 10 values.")
    
    arr = np.array(vehicle_list).reshape(-1,1)
    scaled_input = loaded_scaler.transform(arr)
    scaled_input = scaled_input.reshape(1,10,1)
    
    prediction = loaded_model.predict(scaled_input)
    predicted_value = loaded_scaler.inverse_transform(prediction)
    
    return int(round(predicted_value[0][0]))

# Example prediction
last_10 = df["number_of_vehicles"].values[-10:]
print("Last 10 Hours:", last_10)

predicted = predict_next_hour(list(last_10))
print("Predicted Next Hour Vehicles:", predicted)

print("Process Completed Successfully.")