import streamlit as st
import datetime
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from src.data_loader import download_stock_data
from src.features import calculate_features, FEATURES
from src.model import (train_test_split_timeseries, train_random_forest,
                       evaluate_model, calculate_strategy_returns, predict_next_day)
from src.utils import plot_price_with_sma, plot_rsi

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

# ── Predict Button ────────────────────────────────────────────
if st.button("Predict"):
    with st.spinner("Downloading data and training model..."):
        try:
            # Download data
            yesterday = (datetime.datetime.today() -
                        datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            start_date = (datetime.datetime.today() -
                         datetime.timedelta(days=1095)).strftime('%Y-%m-%d')

            df_raw = download_stock_data(ticker, start_date, yesterday)

            if df_raw.empty:
                st.error("No data found. Please check the ticker symbol.")
                st.stop()

            # Feature engineering
            df = calculate_features(df_raw)

            if len(df) < 100:
                st.error("Not enough data to train. Try a different ticker.")
                st.stop()

            # Train/test split
            X_train, X_test, y_train, y_test, split = train_test_split_timeseries(df, FEATURES)

            # Train model
            model = train_random_forest(X_train, y_train)

            # Evaluate
            predictions, accuracy, report = evaluate_model(model, X_test, y_test)

            # Strategy returns
            buy_hold, strategy = calculate_strategy_returns(df, predictions, split)

            # Predict tomorrow
            prediction, confidence = predict_next_day(model, df, FEATURES)

            # ── Results ──────────────────────────────────────
            st.divider()
            st.subheader(f"Tomorrow's Prediction for {ticker}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Last Close", f"₹{df['Close'].iloc[-1]:.2f}")
            col2.metric("Model Accuracy", f"{accuracy*100:.1f}%")
            col3.metric("Data Points", f"{len(df)}")

            st.divider()

            if prediction == 1:
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

            # ── Strategy Returns ──────────────────────────────
            st.divider()
            st.subheader("Strategy Performance")
            s1, s2, s3 = st.columns(3)
            s1.metric("Buy & Hold Return", f"{buy_hold:.2f}%")
            s2.metric("ML Strategy Return", f"{strategy:.2f}%")
            s3.metric("Outperformance", f"{strategy - buy_hold:.2f}%",
                     delta=f"{strategy - buy_hold:.2f}%")

            # ── Charts ────────────────────────────────────────
            st.divider()
            st.subheader("Price Chart with Moving Averages")
            fig1 = plot_price_with_sma(df, ticker)
            st.pyplot(fig1)

            st.subheader("RSI Indicator")
            fig2 = plot_rsi(df, ticker)
            st.pyplot(fig2)
            

            # ── Sentiment Analysis ────────────────────────────────────────
            st.divider()
            st.subheader("News Sentiment Analysis")

            from src.sentiment import get_news_sentiment, interpret_sentiment

            with st.spinner("Fetching latest news..."):
                sentiment_score, headlines = get_news_sentiment(ticker)
                sentiment_label = interpret_sentiment(sentiment_score)

            col_s1, col_s2 = st.columns(2)
            col_s1.metric("Sentiment Score", f"{sentiment_score:.4f}")
            col_s2.metric("Market Sentiment", sentiment_label)

            if headlines:
                st.write("**Latest Headlines:**")
                for h in headlines[:5]:
                    color = "🟢" if h['sentiment'] > 0.05 else "🔴" if h['sentiment'] < -0.05 else "⚪"
                    st.write(f"{color} {h['headline']}")
            else:
                st.write("No recent news found.")

            # ── Recent Data ───────────────────────────────────
            st.divider()
            st.subheader("Recent Data")
            st.dataframe(df[['Close', 'SMA_20', 'SMA_50',
                             'RSI', 'MACD', 'bb_Width']].tail(10))

            st.caption(f"Data as of: {df.index[-1].date()}")
            st.caption("⚠️ Run after 3:30 PM IST for accurate next-day prediction.")
            st.caption("⚠️ This is not financial advice.")

        except Exception as e:
            st.error(f"Error: {str(e)}")