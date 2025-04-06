import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Import TensorFlow differently to avoid compatibility issues
import tensorflow as tf
from tensorflow import keras

from alpaca.data import CryptoHistoricalDataClient, CryptoBarsRequest, TimeFrame

from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_BASE_URL')

crypto_client = CryptoHistoricalDataClient()

request_params = CryptoBarsRequest(
    symbol_or_symbols='BTC/USD',
    timeframe=TimeFrame.Day,
    start=datetime.datetime(2020, 1, 1),
    end=datetime.datetime(2025, 3, 28)
)

bars = crypto_client.get_crypto_bars(request_params).df
print(bars)

# Create a model to predict the price of BTC

# Import necessary libraries
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Prepare the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(bars['close'].values.reshape(-1, 1))

# Create a function to prepare the data for the model
def prepare_data(data, time_step=100):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Split the data into training and testing sets
train_size = int(0.8 * len(scaled_data))
test_size = len(scaled_data) - train_size

X_train, y_train = prepare_data(scaled_data[:train_size])
X_test, y_test = prepare_data(scaled_data[train_size:])

# Reshape the data for the model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = keras.Sequential()
model.add(keras.layers.LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(32))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model.fit(X_train, y_train, epochs=100, batch_size=64, callbacks=[early_stop])

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform y_train and y_test
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Now compute RMSE correctly
train_score = np.sqrt(np.mean((train_predict - y_train_inv) ** 2))
test_score = np.sqrt(np.mean((test_predict - y_test_inv) ** 2))

print(f'Train Score: {train_score:.2f} RMSE')
print(f'Test Score: {test_score:.2f} RMSE')

print("Train Predictions Range:", train_predict.min(), "to", train_predict.max())
print("Test Predictions Range:", test_predict.min(), "to", test_predict.max())

# Get actual close prices
actual_prices = scaler.inverse_transform(scaled_data)

# Inverse-transform the predictions
train_predict_inv = scaler.inverse_transform(train_predict)
test_predict_inv = scaler.inverse_transform(test_predict)

# Convert timestamp to datetime and handle timezone-aware timestamps
timestamps = pd.to_datetime(pd.Series(bars.index.get_level_values('timestamp')), utc=True)

# Align the prediction dates properly
# Remember: y_train starts at index 100, y_test starts at train_size + 100
train_start = 100
test_start = train_size + 100

train_predict_dates = timestamps[train_start:train_start + len(train_predict)]
test_predict_dates = timestamps[test_start:test_start + len(test_predict)]

# Plot actual vs predictions with correct dates
"""
plt.figure(figsize=(14, 7))
plt.plot(timestamps, actual_prices, label='Actual BTC Price', color='black')

# Plot predictions separately for better visibility
plt.plot(train_predict_dates, train_predict_inv, label='Train Predictions', color='blue', alpha=0.8)
plt.plot(test_predict_dates, test_predict_inv, label='Test Predictions', color='orange', alpha=0.8)

plt.title('BTC Price Prediction with LSTM')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
"""
# Zoom in to show predictions better
#plt.ylim(train_predict.min() - 1000, train_predict.max() + 1000)

#plt.show()

# If you want to visually compare predictions and actual prices only for the parts where predictions exist

plt.figure(figsize=(14, 7))
plt.plot(test_predict_dates, actual_prices[test_start:test_start + len(test_predict_inv)], label='Actual (Test)', color='black')
plt.plot(test_predict_dates, test_predict_inv, label='Predicted (Test)', color='orange')
plt.title('Zoomed-In: BTC Price Prediction on Test Set')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# PREVIOUS PLOT OF ACTUAL PRICE
# Convert timestamp to datetime and handle timezone-aware timestamps
"""
timestamps = pd.to_datetime(pd.Series(bars.index.get_level_values('timestamp')), utc=True)
print(timestamps)

value = bars['close'].values
print(value)

# Plot the price data
plt.figure(figsize=(12, 6))
plt.plot(timestamps, value, label='BTC Price')
plt.title('BTC Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
"""