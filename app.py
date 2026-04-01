import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Quantitative Market Analysis", 
                   page_icon="📈", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("About")
    st.write("""
    **Quantitative Market Analysis**
    
    Predicts next day's price direction for any stock
    using machine learning and technical indicators.
    
    **Model:** Random Forest Classifier  
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
    st.write("1. Enter any NSE ticker (e.g. RELIANCE.NS)")
    st.write("2. Click Predict")
    st.write("3. Run after 3:30 PM IST for best results")
    st.divider()
    st.caption("Built by Samarth Patel")
    st.caption("github.com/samarthpatel03")

# ── Main ──────────────────────────────────────────────────────
st.title("📈 Stock Movement Predictor")
st.subheader("Predict whether a stock will go UP or DOWN tomorrow")

ticker = st.text_input("Enter NSE Stock Ticker", value="RELIANCE.NS")

# ── Feature Engineering Function ─────────────────────────────
def calculate_features(df):
    df = df.copy()
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

# ── Target Variable Function ──────────────────────────────────
def create_target(df):
    df = df.copy()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    return df

# ── Features List ─────────────────────────────────────────────
features = ['Close', 'High', 'Low', 'Open', 'Volume',
            'SMA_20', 'SMA_50', 'RSI', 'MACD',
            'Signal_Line', 'MACD_Histogram', 'bb_Width',
            'Momentum', 'Volatility']

# ── Predict Button ────────────────────────────────────────────
if st.button("Predict"):
    with st.spinner("Downloading data and training model..."):

        try:
            # Download data
            yesterday = (datetime.datetime.today() - 
                        datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            start_date = (datetime.datetime.today() - 
                         datetime.timedelta(days=1095)).strftime('%Y-%m-%d')

            df_raw = yf.download(ticker, start=start_date, end=yesterday)
            
            if df_raw.empty:
                st.error("No data found. Please check the ticker symbol.")
                st.stop()

            df_raw.columns = df_raw.columns.droplevel(1)

            # Feature engineering
            df = calculate_features(df_raw)
            df = create_target(df)

            if len(df) < 100:
                st.error("Not enough data to train. Try a different ticker.")
                st.stop()

            # Train/test split
            split = int(len(df) * 0.80)
            X_train = df[features][:split]
            X_test = df[features][split:]
            y_train = df['Target'][:split]
            y_test = df['Target'][split:]

            # Train fresh model on this stock
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Accuracy on test set
            test_pred = model.predict(X_test)
            acc = accuracy_score(y_test, test_pred)

            # Predict tomorrow using latest row
            latest_features = df[features].iloc[-1:]
            prediction = model.predict(latest_features)
            confidence = model.predict_proba(latest_features)[0]

            # ── Results ──────────────────────────────────────
            st.divider()
            st.subheader(f"Tomorrow's Prediction for {ticker}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Last Close", f"₹{df['Close'].iloc[-1]:.2f}")
            col2.metric("Model Accuracy", f"{acc*100:.1f}%")
            col3.metric("Data Points", f"{len(df)}")

            st.divider()

            if prediction[0] == 1:
                st.success(f"📈 BUY — Model predicts {ticker} will go UP tomorrow")
                st.metric("Confidence", f"{confidence[1]*100:.0f}%")
            else:
                st.error(f"📉 SELL/HOLD — Model predicts {ticker} will go DOWN tomorrow")
                st.metric("Confidence", f"{confidence[0]*100:.0f}%")

            # ── Technical Indicators ─────────────────────────
            st.divider()
            st.subheader("Current Technical Indicators")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
            c2.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
            c3.metric("BB Width", f"{df['bb_Width'].iloc[-1]:.4f}")
            c4.metric("Momentum", f"{df['Momentum'].iloc[-1]:.2f}")

            # ── Price Chart ───────────────────────────────────
            st.divider()
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

            # ── RSI Chart ─────────────────────────────────────
            st.subheader("RSI Indicator")
            fig2, ax2 = plt.subplots(figsize=(14, 3))
            ax2.plot(df['RSI'], color='purple', label='RSI')
            ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
            ax2.axhline(30, color='green', linestyle='--', label='Oversold (30)')
            ax2.set_title(f'{ticker} — RSI')
            ax2.legend()
            st.pyplot(fig2)

            # ── Latest Data ───────────────────────────────────
            st.divider()
            st.subheader("Recent Data")
            st.dataframe(df[['Close', 'SMA_20', 'SMA_50', 
                             'RSI', 'MACD', 'bb_Width']].tail(10))

            st.caption(f"Data as of: {df.index[-1].date()}")
            st.caption("⚠️ Run after 3:30 PM IST for accurate next-day prediction.")
            st.caption("⚠️ This is not financial advice.")

        except Exception as e:
            st.error(f"Error: {str(e)}")