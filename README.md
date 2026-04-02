# Quantitative Market Analysis

A machine learning project that predicts stock market movement direction 
using technical indicators and quantitative analysis techniques.
Includes a live web application for real-time trading signals.

## Live Demo
Run the Streamlit app locally for live BUY/SELL predictions on any NSE stock.

## Current Project — Stock Movement Prediction
Predicting whether Reliance Industries (RELIANCE.NS) stock will go
up or down the next trading day using historical price data and 
technical indicators.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- yfinance
- Matplotlib, Seaborn
- Streamlit
- Joblib

## Features Engineered
| Indicator | Type | Purpose |
|-----------|------|---------|
| SMA 20, SMA 50 | Trend | Identifies Golden/Death Cross |
| RSI | Momentum | Overbought/Oversold signals |
| MACD & Histogram | Trend + Momentum | Momentum phase detection |
| Signal Line | Trend | MACD trigger line |
| Bollinger Band Width | Volatility | Market squeeze detection |
| Momentum | Momentum | 10-day price change |
| Volatility | Risk | Rolling standard deviation of returns |

## Exploratory Data Analysis
- Price trend with Golden Cross & Death Cross analysis
- RSI overbought/oversold signals (21 overbought, 15 oversold instances)
- MACD Histogram momentum phases
- Daily return distribution with fat tail analysis
- Feature correlation heatmap with multicollinearity analysis
- Bollinger Bands with support/resistance zones

## Model Comparison
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 47.83% |
| XGBoost | 49.46% |
| **Random Forest** | **52.72%** ✅ |

## Strategy Results
| Metric | Result |
|--------|--------|
| Test Period | 2024-2025 |
| Buy & Hold Return | -3.05% |
| ML Strategy Return | **-0.11%** |
| Outperformance | **+2.94%** |

> During a declining market period, the ML strategy preserved capital 
> better than passive holding — demonstrating downside protection 
> capability of the model.

## Key Findings
- Volume is the most important predictive feature (9.09% importance)
- Volatility and Bollinger Band Width outperform price-based features
- Model shows asymmetric performance — better at identifying DOWN days (61% accuracy) than UP days
- Logistic Regression underperformed below random chance — confirming stock direction is a non-linear problem
- Efficient Market Hypothesis limits accuracy of technical indicators alone
- Model adds most value in choppy/sideways markets as a defensive strategy

## Confusion Matrix Insights
- True Negatives (DOWN correctly predicted): 58/95 = 61%
- True Positives (UP correctly predicted): 39/89 = 44%
- Model has defensive bias — better at avoiding losses than catching gains

## Experiments Conducted
| Experiment | Result |
|------------|--------|
| 5-day prediction horizon | Lower strategy returns |
| 10-year data period | Regime change hurt performance |
| Threshold filtering (55%) | Reduced trades, lower returns |
| Ensemble (all 3 models) | Underperformed standalone RF |
| Nifty 50 index | Buy & Hold outperformed in bull market |

## Web Application
Built with Streamlit — enter any NSE ticker for live predictions.

**Features:**
- Real-time data download via yfinance
- Automatic technical indicator calculation
- BUY/SELL signal with confidence score
- Price chart with moving averages
- RSI indicator chart
- Current indicator values

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Note:** Run after 3:30 PM IST for accurate next-day predictions.

## Project Structure
```
quantitative-market-analysis/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Data download
│   ├── features.py        # Feature engineering
│   ├── model.py           # Training and prediction
│   └── utils.py           # Visualization helpers
│
├── stock_prediction.ipynb # Research notebook
├── app.py                 # Streamlit web app
├── requirements.txt       # Dependencies
└── README.md
```

## Limitations
- Based on technical indicators only — no fundamental or sentiment data
- Model performance varies with market regime changes
- Not suitable for live trading without proper risk management
- Transaction costs not accounted for in strategy returns

## Future Work
- Add news sentiment analysis
- Include fundamental indicators (P/E ratio, EPS)
- Implement proper backtesting with transaction costs
- Expand to multiple stocks
- Add Sharpe ratio and maximum drawdown metrics

## Author
Samarth Patel  
github.com/samarthpatel03