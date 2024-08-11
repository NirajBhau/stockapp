import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.graph_objects as go

# Function to load the data
def load_data(ticker):
    data = yf.download(ticker, period="5y")
    return data

# Function to preprocess the data
def preprocess_data(data):
    data['100 EMA'] = data['Close'].ewm(span=100, adjust=False).mean()
    data['200 EMA'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['Prediction'] = data['Close'].shift(-30)  # Predict for 1 month ahead
    return data

# Function to train the model
def train_model(data):
    X = np.array(data[['Close']])[:-30]  # Only using 'Close' price for simplicity
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
    
    X = np.array(data[['Close']])[-30:]  # Predicting for the last 30 days
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
    
    # Plotting the Close price
    fig.add_trace(go.Scatter(
        x=preprocessed_data.index, 
        y=preprocessed_data['Close'], 
        mode='lines', 
        name=f"{ticker} Close Price",
        line=dict(color='blue')
    ))
    
    # Plotting the 100 EMA
    fig.add_trace(go.Scatter(
        x=preprocessed_data.index, 
        y=preprocessed_data['100 EMA'], 
        mode='lines', 
        name='100 EMA',
        line=dict(color='red')
    ))
    
    # Plotting the 200 EMA
    fig.add_trace(go.Scatter(
        x=preprocessed_data.index, 
        y=preprocessed_data['200 EMA'], 
        mode='lines', 
        name='200 EMA',
        line=dict(color='yellow')
    ))

    # Setting the title and axis labels
    fig.update_layout(
        title=f"{ticker} Stock Price with EMA",
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x'
    )

    st.plotly_chart(fig)

    # Separate Visualization for Prediction
    st.subheader("Prediction for Next 30 Days")
    
    fig2 = go.Figure()
    
    # Plotting the prediction line
    fig2.add_trace(go.Scatter(
        x=preprocessed_data.index[-30:], 
        y=predictions, 
        mode='lines+markers', 
        name='Prediction for next 30 days',
        line=dict(color='green', dash='dot')
    ))
    
    fig2.update_layout(
        title=f"{ticker} Stock Price Prediction (Next 30 Days)",
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x'
    )

    st.plotly_chart(fig2)

    st.subheader("Stock Price Trend Over Time")
    
    # Visualization of the overall trend with zoom and hover features
    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        x=preprocessed_data.index, 
        y=preprocessed_data['Close'], 
        mode='lines', 
        name=f"{ticker} Close Price",
        line=dict(color='blue')
    ))

    fig3.update_layout(
        title="Stock Price Trend Over Time",
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x',
        xaxis_rangeslider_visible=True  # Enables zoom-in and zoom-out features
    )

    st.plotly_chart(fig3)

    # New Visualizations

    # 1. Volume Visualization
    st.subheader("Volume Traded Over Time")
    
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=preprocessed_data.index,
        y=preprocessed_data['Volume'],
        name='Volume Traded',
        marker=dict(color='orange')
    ))

    fig4.update_layout(
        title="Volume Traded Over Time",
        xaxis_title='Date',
        yaxis_title='Volume',
        hovermode='x'
    )

    st.plotly_chart(fig4)

    # 2. Candlestick Chart with Drawing Tools
    st.subheader("Candlestick Chart with Drawing Tools")

    # User selects the time frame
    time_frame = st.selectbox("Select Time Frame", options=["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])

    # Resample data based on selected time frame
    resampled_data = resample_data(preprocessed_data, time_frame)

    fig5 = go.Figure(data=[go.Candlestick(
        x=resampled_data.index,
        open=resampled_data['Open'],
        high=resampled_data['High'],
        low=resampled_data['Low'],
        close=resampled_data['Close'],
        name=f'{time_frame} Candlestick'
    )])

    # Adding a trend line
    st.subheader("Add a Trend Line")
    start_trend = st.date_input("Select Start Date for Trend Line", resampled_data.index.min())
    end_trend = st.date_input("Select End Date for Trend Line", resampled_data.index.max())

    trend_data = resampled_data.loc[start_trend:end_trend]

    if not trend_data.empty:
        fig5.add_trace(go.Scatter(
            x=trend_data.index,
            y=trend_data['Close'],
            mode='lines',
            name='Trend Line',
            line=dict(color='green', dash='dash')
        ))

    # Adding Fibonacci retracement
    st.subheader("Add Fibonacci Retracement")
    max_price = trend_data['High'].max()
    min_price = trend_data['Low'].min()

    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    for level in fib_levels:
        fig5.add_shape(
            type="line",
            x0=trend_data.index[0],
            y0=min_price + level * (max_price - min_price),
            x1=trend_data.index[-1],
            y1=min_price + level * (max_price - min_price),
            line=dict(color="purple", dash="dash"),
            name=f"Fib {level}"
        )

    # Adding a rectangle
    st.subheader("Add a Rectangle")
    rect_start = st.date_input("Select Start Date for Rectangle", resampled_data.index.min())
    rect_end = st.date_input("Select End Date for Rectangle", resampled_data.index.max())
    rect_y0 = st.number_input("Select Lower Price for Rectangle", value=min_price, step=10.0)
    rect_y1 = st.number_input("Select Upper Price for Rectangle", value=max_price, step=10.0)

    fig5.add_shape(
        type="rect",
        x0=rect_start,
        y0=rect_y0,
        x1=rect_end,
        y1=rect_y1,
        line=dict(color="blue", width=2),
        fillcolor="lightblue",
        opacity=0.2
    )

    # Adding other shapes, like ellipses, could follow a similar pattern.

    fig5.update_layout(
        title=f"{ticker} {time_frame} Candlestick with Custom Tools",
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x'
    )

    st.plotly_chart(fig5)

    # You can further customize these tools by allowing users to input specific start and end points for trend lines, rectangles, and more.
    # Additionally, integrating interactive drawing tools is possible with more advanced features in libraries like Dash or custom JavaScript integrations.