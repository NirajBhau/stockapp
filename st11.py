import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta

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
    
    # Generate dates for the next 30 days
    last_date = preprocessed_data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]

    # Adding Candlestick Chart Selection
    st.subheader("Candlestick Chart for Predictions")
    time_frame = st.selectbox("Select Time Frame for Predictions", options=["Daily", "Weekly", "Monthly"])

    # Resample future dates based on selected time frame
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Open': predictions,  # Using predictions as a placeholder for Open prices
        'High': predictions + (np.random.random(len(predictions)) * 5),  # Adding some variance for High prices
        'Low': predictions - (np.random.random(len(predictions)) * 5),  # Adding some variance for Low prices
        'Close': predictions,
        'Volume': np.random.randint(1e6, 1e7, size=len(predictions))  # Random volume data
    }).set_index('Date')

    resampled_predictions = resample_data(future_df, time_frame)

    fig2 = go.Figure(data=[go.Candlestick(
        x=resampled_predictions.index,
        open=resampled_predictions['Open'],
        high=resampled_predictions['High'],
        low=resampled_predictions['Low'],
        close=resampled_predictions['Close'],
        name=f'{time_frame} Candlestick'
    )])

    fig2.update_layout(
        title=f"{ticker} Stock Price Prediction (Next 30 Days) - {time_frame} Candlestick",
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x',
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig2)

    # Volume Visualization
    st.subheader("Volume Traded Over Time")
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=preprocessed_data.index,
        y=preprocessed_data['Volume'],
        name='Volume Traded',
        marker=dict(color='orange')
    ))

    fig3.update_layout(
        title="Volume Traded Over Time",
        xaxis_title='Date',
        yaxis_title='Volume',
        hovermode='x'
    )

    st.plotly_chart(fig3)

    # Candlestick Chart with Multiple Time Frames
    st.subheader("Candlestick Chart")

    # User selects the time frame
    time_frame = st.selectbox("Select Time Frame", options=["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])

    # Resample data based on selected time frame
    resampled_data = resample_data(preprocessed_data, time_frame)

    fig4 = go.Figure(data=[go.Candlestick(
        x=resampled_data.index,
        open=resampled_data['Open'],
        high=resampled_data['High'],
        low=resampled_data['Low'],
        close=resampled_data['Close'],
        name=f'{time_frame} Candlestick'
    )])

    fig4.update_layout(
        title=f"{ticker} {time_frame} Candlestick Chart",
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x',
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig4)

    # Historical vs Predicted Price Comparison
    st.subheader("Historical vs Predicted Price Comparison")
    
    fig5 = go.Figure()

    fig5.add_trace(go.Scatter(
        x=preprocessed_data.index[:-30],
        y=preprocessed_data['Close'][:-30],
        mode='lines',
        name='Historical Price',
        line=dict(color='blue')
    ))

    fig5.add_trace(go.Scatter(
        x=preprocessed_data.index[-30:],
        y=predictions,
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='green', dash='dot')
    ))

    fig5.update_layout(
        title=f"{ticker} Historical vs Predicted Price",
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x'
    )

    st.plotly_chart(fig5)

    # User customization
    st.subheader("Customize Visualization")

    # Date Range Selection
    start_date = st.date_input("Select Start Date", preprocessed_data.index.min())
    end_date = st.date_input("Select End Date", preprocessed_data.index.max())

    # Filtering data based on date selection
    filtered_data = preprocessed_data.loc[start_date:end_date]

    # User-selected EMA (e.g., 50, 100, 200)
    ema_span = st.slider("Select EMA Span", 10, 300, 100)
    filtered_data[f'{ema_span} EMA'] = filtered_data['Close'].ewm(span=ema_span, adjust=False).mean()

    # Custom EMA Visualization
    st.subheader(f"Custom {ema_span} EMA Visualization")

    fig6 = go.Figure()

    fig6.add_trace(go.Scatter(
        x=filtered_data.index,
        y=filtered_data['Close'],
        mode='lines',
        name=f"{ticker} Close Price",
        line=dict(color='blue')
    ))

    fig6.add_trace(go.Scatter(
        x=filtered_data.index,
        y=filtered_data[f'{ema_span} EMA'],
        mode='lines',
        name=f'{ema_span} EMA',
        line=dict(color='purple')
    ))

    fig6.update_layout(
        title=f"{ticker} Stock Price with {ema_span} EMA",
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x'
    )

    st.plotly_chart(fig6)


from sklearn.metrics import r2_score

# Function to calculate and print the R² score
def evaluate_model(predictions, actuals):
    r2 = r2_score(actuals, predictions)
    return r2

# Assuming you have already trained the model and made predictions:
if model is not None:
    actuals = np.array(preprocessed_data['Prediction'][-30:])  # Last 30 actuals
    r2 = evaluate_model(predictions, actuals)
    st.write(f"R² score for {ticker} prediction model: {r2:.4f}")


