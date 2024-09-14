import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta

# Function to load stock data till yesterday
def load_data(ticker):
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    data = yf.download(ticker, start="2010-01-01", end=yesterday.strftime('%Y-%m-%d'))
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

# Compute technical indicators
def compute_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Preprocess data for LSTM
def preprocess_data(data):
    data = compute_technical_indicators(data)
    data = data.dropna()  # Drop rows with NaN values

    # Use 'Close' price and technical indicators for prediction
    features = data[['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)

    # Prepare training and test data
    train_size = int(len(features_scaled) * 0.80)
    train_data = features_scaled[:train_size]
    test_data = features_scaled[train_size:]

    return train_data, test_data, scaler

# Prepare the LSTM data
def create_lstm_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i])
        y.append(data[i, 0])  # Predicting 'Close' price
    X = np.array(X)
    y = np.array(y)
    return X, y

# Build and train LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to predict today's stock price
def predict_today_price(model, test_data, scaler, time_step=60):
    try:
        last_60_days = test_data[-time_step:]
        last_60_days_scaled = scaler.transform(last_60_days)

        X_test = np.array([last_60_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(np.concatenate((predicted_price, np.zeros((predicted_price.shape[0], 4))), axis=1))[:, 0]
        return predicted_price[0]
    except Exception as e:
        st.error(f"An error occurred while predicting today's price: {e}")
        return None

# Streamlit app interface
def main():
    st.title("Indian Stock Price Prediction App")
    
    # User input for stock ticker symbol (Indian stocks)
    ticker = st.text_input("Enter Indian Stock Ticker (e.g. RELIANCE.NS, TCS.NS, INFY.NS)", "RELIANCE.NS")
    
    if ticker:
        # Load and display the stock data till yesterday
        data = load_data(ticker)
        if data.empty:
            st.error(f"No data found for ticker {ticker}. Please check the ticker symbol.")
            return
        
        st.subheader(f"{ticker} Stock Data (Till Yesterday)")
        st.write(data.tail())

        # Plot the stock data
        plot_data(data, ticker)

        # Preprocess data for LSTM
        train_data, test_data, scaler = preprocess_data(data)
        X_train, y_train = create_lstm_dataset(train_data)
        X_test, y_test = create_lstm_dataset(test_data)

        # Build and train the LSTM model
        model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1)

        # Predict today's stock price
        predicted_price_today = predict_today_price(model, data[['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI']].values, scaler)
        if predicted_price_today is not None:
            st.subheader(f"Predicted Stock Price for Today ({ticker}): ₹{predicted_price_today:.2f}")

            # Visualize test predictions vs actual values
            st.subheader(f"{ticker} Stock Price Prediction vs Actual")
            predicted_prices = model.predict(X_test)
            predicted_prices = scaler.inverse_transform(np.concatenate((predicted_prices, np.zeros((predicted_prices.shape[0], 4))), axis=1))[:, 0]
            actual_prices = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4))), axis=1))[:, 0]

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
