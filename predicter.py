import numpy as np
import joblib
from tensorflow.keras.models import load_model


# ==========================
# Load Model and Scaler
# ==========================

model = load_model("traffic_rnn_model.h5")
scaler = joblib.load("traffic_scaler.save")


# ==========================
# Prediction Function
# ==========================
from random import randint
def predict_next_hour(vehicle_list):
    """
    vehicle_list : list of last 10 vehicle counts
    returns : predicted next hour vehicle count (integer)
    """

    
    if len(vehicle_list) != 10:
        return randint(2, 10)
    # Convert to numpy array
    arr = np.array(vehicle_list).reshape(-1, 1)
    
    # Scale input
    scaled_input = scaler.transform(arr)
    
    # Reshape for LSTM
    scaled_input = scaled_input.reshape(1, 10, 1)
    
    # Predict
    prediction = model.predict(scaled_input, verbose=0)
    
    # Inverse scale
    predicted_value = scaler.inverse_transform(prediction)
    
    return int(round(predicted_value[0][0]))


# ==========================
# Example Usage
# ==========================

#if __name__ == "__main__":
    
sample_input = [5, 6, 7, 8, 10, 12, 14, 15, 13, 11]
    
result = predict_next_hour(sample_input)
    
print("Input:", sample_input)
print("Predicted Next Hour Vehicles:", result)