import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# App title
st.title("Stock Price Prediction App")

# Sidebar options
st.sidebar.header("Settings")
selected_stock = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA):", "AAPL")
time_frame = st.sidebar.selectbox("Select Time Frame:", ["1D", "5D", "1W", "1M", "3M", "6M", "1Y"])
prediction_days = st.sidebar.slider("Prediction Days (1-30):", min_value=1, max_value=30, value=7)

# Helper function to fetch data
def fetch_data(symbol, interval):
    data = yf.download(symbol, interval=interval)
    return data

# Time frame mapping
interval_mapping = {
    "1D": "1d",
    "5D": "5d",
    "1W": "1wk",
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y"
}

# Fetch data
st.header(f"Stock Data for {selected_stock}")
try:
    interval = interval_mapping[time_frame]
    data = fetch_data(selected_stock, interval)

    # Calculate EMAs
    data['50 EMA'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['100 EMA'] = data['Close'].ewm(span=100, adjust=False).mean()

    # Display raw data
    st.write(data.tail())

    # Prepare data for charts
    data.reset_index(inplace=True)

    # Create subplots for candlestick and volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Candlestick chart with EMAs
    fig.add_trace(
        go.Candlestick(x=data['Date'],
                       open=data['Open'],
                       high=data['High'],
                       low=data['Low'],
                       close=data['Close'],
                       name='Candlestick'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=data['Date'], y=data['50 EMA'], mode='lines', name='50 EMA', line=dict(color='blue')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=data['Date'], y=data['100 EMA'], mode='lines', name='100 EMA', line=dict(color='red')),
        row=1, col=1
    )

    # Volume bar chart
    fig.add_trace(
        go.Bar(x=data['Date'], y=data['Volume'], name='Volume'),
        row=2, col=1
    )

    # Layout updates
    fig.update_layout(title=f"{selected_stock} Price, EMAs, and Volume",
                      xaxis_rangeslider_visible=False,
                      template="plotly_white")

    # Show chart
    st.plotly_chart(fig)

    # Linear regression for prediction
    st.subheader("Stock Price Prediction")

    # Feature engineering
    data['Day'] = np.arange(len(data))
    X = data[['Day']]
    y = data['Close']

    # Scale data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict next 30 days
    future_days = np.arange(len(data), len(data) + prediction_days).reshape(-1, 1)
    future_days_scaled = scaler.transform(future_days)
    predictions = model.predict(future_days_scaled)

    # Display predictions
    prediction_df = pd.DataFrame({"Day": future_days.flatten(), "Predicted Close": predictions})
    st.write(prediction_df)

    # Plot predictions
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Actual'))
    future_dates = pd.date_range(data['Date'].iloc[-1], periods=prediction_days + 1)[1:]
    fig_pred.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='Predicted'))
    fig_pred.update_layout(title="Predicted vs Actual", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_pred)

    # Export option
    csv = prediction_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")

except Exception as e:
    st.error(f"Error fetching data: {e}")
