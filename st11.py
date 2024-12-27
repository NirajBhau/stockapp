import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Function to preprocess data
def preprocess_data(data):
    if data.empty or len(data) < 200:  # Ensure sufficient data for EMA calculation
        return None
    data['100 EMA'] = data['Close'].ewm(span=100, adjust=False).mean()
    data['200 EMA'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['Prediction'] = data['Close'].shift(-30)
    data = data.dropna(subset=['Prediction'])  # Drop rows with NaN in 'Prediction'
    return data

# Function to resample data
def resample_data(data, interval):
    data = data.resample(interval).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    return data

# Function to train model
def train_model(data):
    X = np.array(data[['Close']])
    y = np.array(data['Prediction'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions, y_test

# Function to evaluate model
def evaluate_model(predictions, actuals):
    if len(predictions) == len(actuals):
        return r2_score(actuals, predictions)
    return None

# Streamlit app
st.title("Stock Analysis Dashboard")

# Stock ticker input
ticker = st.text_input("Enter stock ticker", "AAPL")

# Data loading
data = yf.download(ticker, period="5y")
if data.empty:
    st.error("Failed to load data for the ticker. Please check the ticker symbol.")
else:
    data.index = pd.to_datetime(data.index)
    st.write(f"Data for {ticker}", data.tail())

    # Preprocess data
    preprocessed_data = preprocess_data(data)
    if preprocessed_data is None:
        st.error("Insufficient data for processing. Please select another stock ticker.")
    else:
        st.write("Preprocessed Data", preprocessed_data.tail())

        # Date filtering
        start_date = pd.to_datetime(st.date_input("Select Start Date", preprocessed_data.index.min()))
        end_date = pd.to_datetime(st.date_input("Select End Date", preprocessed_data.index.max()))

        if start_date < end_date:
            filtered_data = preprocessed_data.loc[start_date:end_date]
            st.write("Filtered Data", filtered_data)

            # Resample data
            interval = st.selectbox("Select interval", ["1D", "1W", "1M"])
            resampled_data = resample_data(filtered_data, interval)
            st.write("Resampled Data", resampled_data.tail())

            # Train and evaluate model
            model, predictions, actuals = train_model(preprocessed_data)
            r2 = evaluate_model(predictions, actuals)
            if r2 is not None:
                st.write(f"R² score: {r2:.4f}")
            else:
                st.error("Model evaluation failed due to mismatched prediction and actual values.")

            # Future predictions placeholder
            future_df = filtered_data.iloc[-30:].copy()
            future_df['Open'] = future_df['Close'] + np.random.uniform(-2, 2, size=len(future_df))
            st.write("Future Data Placeholder", future_df)
        else:
            st.error("Start date must be earlier than end date.")
