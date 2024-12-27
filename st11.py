import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

# Function to load the data
def load_data(ticker):
    try:
        data = yf.download(ticker, period="5y")
        data = data.dropna()
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to preprocess the data
def preprocess_data(data):
    data['100 EMA'] = data['Close'].ewm(span=100, adjust=False).mean()
    data['200 EMA'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['Prediction'] = data['Close'].shift(-30)  # Predict for 1 month ahead
    return data

# Function to train the model
def train_model(data):
    X = np.array(data[['Close']])[:-30]
    y = np.array(data['Prediction'])[:-30]

    if len(X) == 0 or len(y) == 0:
        return None

    model = LinearRegression()
    model.fit(X, y)

    return model

# Function to make predictions
def make_predictions(model, data):
    if model is None:
        return None

    X = np.array(data[['Close']])[-30:]
    predictions = model.predict(X)
    return predictions

# Function to resample data for different time frames
def resample_data(data, time_frame):
    if time_frame == 'Daily':
        return data
    elif time_frame == 'Weekly':
        return data.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    elif time_frame == 'Monthly':
        return data.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    elif time_frame == 'Quarterly':
        return data.resample('Q').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    elif time_frame == 'Yearly':
        return data.resample('Y').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })

# Streamlit App
st.title("Indian Stock Price Prediction")

ticker = st.text_input("Enter Indian Stock Ticker Symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS)", "RELIANCE.NS")
data = load_data(ticker)

if data is not None:
    st.subheader(f"Stock Price Data for {ticker}")
    st.write(data.tail())

    preprocessed_data = preprocess_data(data)

    model = train_model(preprocessed_data)

    if model is None:
        st.error(f"Not enough data available for {ticker} to make predictions.")
    else:
        predictions = make_predictions(model, preprocessed_data)

        # Visualization for Stock Data with EMAs
        st.subheader("Visualization of Stock Data with 100 EMA and 200 EMA")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=preprocessed_data.index, 
            y=preprocessed_data['Close'], 
            mode='lines', 
            name=f"{ticker} Close Price",
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=preprocessed_data.index, 
            y=preprocessed_data['100 EMA'], 
            mode='lines', 
            name='100 EMA',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=preprocessed_data.index, 
            y=preprocessed_data['200 EMA'], 
            mode='lines', 
            name='200 EMA',
            line=dict(color='yellow')
        ))

        fig.update_layout(
            title=f"{ticker} Stock Price with EMA",
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x'
        )

        st.plotly_chart(fig)

        # Prediction Visualization
        st.subheader("Prediction for Next 30 Days")

        last_date = preprocessed_data.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]

        future_df = pd.DataFrame({
            'Date': future_dates,
            'Close': predictions
        }).set_index('Date')

        st.line_chart(future_df['Close'], height=400, use_container_width=True)

        # Historical vs Predicted Price Comparison
        st.subheader("Historical vs Predicted Price Comparison")

        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=preprocessed_data.index[:-30],
            y=preprocessed_data['Close'][:-30],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue')
        ))

        fig2.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='green', dash='dot')
        ))

        fig2.update_layout(
            title=f"{ticker} Historical vs Predicted Price",
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x'
        )

        st.plotly_chart(fig2)

        # Model Evaluation
        actuals = np.array(preprocessed_data['Prediction'][-30:])
        if len(actuals) == len(predictions):
            r2 = r2_score(actuals, predictions)
            st.write(f"R² score for {ticker} prediction model: {r2:.4f}")
