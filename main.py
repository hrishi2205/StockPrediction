import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Function to load data from Yahoo Finance for Indian stocks
def load_data(ticker):
    ticker = ticker + ".NS"  # For Indian stocks on NSE
    data = yf.download(ticker, start="2015-01-01", end="2023-01-01")
    return data

# Function to plot the stock price history
def plot_data(data, ticker):
    fig, ax = plt.subplots()
    ax.plot(data['Close'], label='Closing Price')
    ax.set_title(f'{ticker} Stock Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (INR)')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Preprocess data for LSTM
def preprocess_data(data):
    # Use 'Close' price for prediction
    data_close = data['Close'].values
    data_close = data_close.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_close_scaled = scaler.fit_transform(data_close)

    # Prepare training data
    train_size = int(len(data_close_scaled) * 0.80)
    train_data = data_close_scaled[:train_size]
    test_data = data_close_scaled[train_size:]

    return train_data, test_data, scaler

# Prepare the LSTM data
def create_lstm_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    X = np.array(X)
    y = np.array(y)
    return X.reshape(X.shape[0], X.shape[1], 1), y

# Build and train LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to make predictions
def predict_future_price(model, test_data, scaler, time_step=60):
    last_60_days = test_data[-time_step:]
    last_60_days_scaled = scaler.transform(last_60_days)

    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][0]

# Streamlit app interface
def main():
    st.title("Indian Stock Price Prediction App")
    
    # User input for ticker 
