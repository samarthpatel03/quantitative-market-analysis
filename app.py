import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import datetime

# Load the saved model
model = joblib.load('best_model.joblib')

# App title
st.title("Stock Movement Predictor")
st.subheader("Predict whether a stock will go UP or DOWN tomorrow")

# User input
ticker = st.text_input("Enter Stock Ticker", value="RELIANCE.NS")

# Feature engineering function
def calculate_features(df):
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    # RSI
    daily_change = df['Close'].diff()
    gain = daily_change.clip(lower=0)
    loss = daily_change.clip(upper=0).abs()
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    RS = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + RS))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands
    std_20 = df['Close'].rolling(20).std()
    df['bb_Width'] = (4 * std_20) / df['SMA_20']
    
    # Momentum and Volatility
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    
    df.dropna(inplace=True)
    return df

# Features list
features = ['Close', 'High', 'Low', 'Open', 'Volume',
            'SMA_20', 'SMA_50', 'RSI', 'MACD',
            'Signal_Line', 'MACD_Histogram', 'bb_Width',
            'Momentum', 'Volatility']

# Sidebar
with st.sidebar:
    st.title("About")
    st.write("""
    **Quantitative Market Analysis**
    
    This app predicts the next day's price direction 
    for any NSE listed stock using machine learning
    and technical indicators.
    
    **Model:** Random Forest Classifier  
    **Accuracy:** ~53%  
    **Indicators Used:**
    - SMA 20 & SMA 50
    - RSI
    - MACD & Signal Line
    - Bollinger Band Width
    - Momentum
    - Volatility
    """)
    
    st.divider()
    st.write("**How to use:**")
    st.write("1. Enter NSE ticker (e.g. RELIANCE.NS)")
    st.write("2. Click Predict")
    st.write("3. Run after 3:30 PM IST for best results")
    
    st.divider()
    st.caption("Built by Samarth Patel")
    st.caption("github.com/samarthpatel03")

# Button to trigger prediction
if st.button("Predict"):
    with st.spinner("Downloading data and calculating indicators..."):
        
        # Download data
        yesterday = (datetime.datetime.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        df = yf.download(ticker, start="2022-01-01", end=yesterday)
        df.columns = df.columns.droplevel(1)
        
        # Calculate features
        df = calculate_features(df)
        
        # Get latest row for prediction
        latest_features = df[features].iloc[-1:]
        
        # Predict
        prediction = model.predict(latest_features)
        confidence = model.predict_proba(latest_features)[0]
        
        # Show prediction
        st.subheader("Tomorrow's Prediction")
        if prediction[0] == 1:
            st.success(f"📈 BUY — Model predicts {ticker} will go UP tomorrow")
            st.metric("Confidence", f"{confidence[1]*100:.0f}%")
        else:
            st.error(f"📉 SELL/HOLD — Model predicts {ticker} will go DOWN tomorrow")
            st.metric("Confidence", f"{confidence[0]*100:.0f}%")
        
        # Show latest indicator values
        st.subheader("Current Technical Indicators")
        col1, col2, col3 = st.columns(3)
        col1.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
        col2.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
        col3.metric("BB Width", f"{df['bb_Width'].iloc[-1]:.4f}")

        # Price chart with moving averages
        st.subheader("Price Chart with Moving Averages")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df['Close'], label='Close Price', alpha=0.7)
        ax.plot(df['SMA_20'], label='SMA 20', alpha=0.8)
        ax.plot(df['SMA_50'], label='SMA 50', alpha=0.8)
        ax.set_title(f'{ticker} — Price with Moving Averages')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)

        # RSI chart
        st.subheader("RSI Indicator")
        fig2, ax2 = plt.subplots(figsize=(14, 3))
        ax2.plot(df['RSI'], color='purple', label='RSI')
        ax2.axhline(70, color='red', linestyle='--', label='Overbought')
        ax2.axhline(30, color='green', linestyle='--', label='Oversold')
        ax2.set_title(f'{ticker} — RSI')
        ax2.legend()
        st.pyplot(fig2)

        # Show last close price
        st.subheader("Latest Data")
        st.write(f"Last Close Price: ₹{df['Close'].iloc[-1]:.2f}")
        st.write(f"Data as of: {df.index[-1].date()}")
        st.caption("⚠️ Run after 3:30 PM IST for accurate next-day prediction. This is not financial advice.")