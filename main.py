import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import datetime, timedelta

# Function to load stock data till yesterday
def load_data(ticker):
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    data = yf.download(ticker, start="2015-01-01", end=yesterday.strftime('%Y-%m-%d'))
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

# Function to predict today's stock price
def predict_today_price(model, test_data, scaler, time_step=60):
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
    
    # User input for stock ticker symbol (Indian stocks)
    ticker = st.text_input("Enter Indian Stock Ticker (e.g. RELIANCE.NS, TCS.NS, INFY.NS)", "RELIANCE.NS")
    
    if ticker:
        # Load and display the stock data till yesterday
        data = load_data(ticker)
        st.subheader(f"{ticker} Stock Data (Till Yesterday)")
        st.write(data.tail())

        # Plot the stock data
        plot_data(data, ticker)

        # Preprocess data for LSTM
        train_data, test_data, scaler = preprocess_data(data)
        X_train, y_train = create_lstm_dataset(train_data)
        X_test, y_test = create_lstm_dataset(test_data)

        # Build and train the LSTM model
        model = build_model()
        model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)

        # Predict today's stock price
        predicted_price_today = predict_today_price(model, data['Close'].values.reshape(-1, 1), scaler)
        st.subheader(f"Predicted Stock Price for Today ({ticker}): ₹{predicted_price_today:.2f}")

        # Visualize test predictions vs actual values
        st.subheader(f"{ticker} Stock Price Prediction vs Actual")
        predicted_prices = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        fig2, ax2 = plt.subplots()
        ax2.plot(actual_prices, label="Actual Prices")
        ax2.plot(predicted_prices, label="Predicted Prices")
        ax2.set_title(f'{ticker} Price Prediction vs Actual')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price (INR)')
        plt.legend()
        st.pyplot(fig2)

if __name__ == '__main__':
    main()
