
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import yfinance as yf
stock_symbol = 'TSLA'  # Tesla
data = yf.download(stock_symbol, start="2023-01-01", end="2023-08-31")
print(data.head())
data_close = data[['Close']]

# Scale data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_close)
def prepare_data(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step, 0])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

#data for LSTM
time_step = 60
X, y = prepare_data(data_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])


model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

#predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Reverse scaling of predictions
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

# Reverse scaling of y_train and y_test
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data_close.index[:len(y_train)], y_train, color='blue', label='Training Data')
plt.plot(data_close.index[len(y_train):(len(y_train) + len(y_test))], y_test, color='green', label='Testing Data')
plt.plot(data_close.index[len(y_train):(len(y_train) + len(test_predictions))], test_predictions, color='red', label='Predicted Price')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
