# Quantitative Market Analysis

A machine learning project that predicts stock market movement direction 
using technical indicators, news sentiment analysis, and quantitative techniques.
Includes a live web application for real-time trading signals on any NSE stock.

## Current Project — Stock Movement Prediction
Predicting whether a stock will go UP or DOWN the next trading day using 
historical price data, technical indicators, and real-time news sentiment.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- yfinance
- Matplotlib, Seaborn
- Streamlit
- VADER Sentiment Analyzer
- Joblib

## Project Structure
```
quantitative-market-analysis/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Data download
│   ├── features.py        # Feature engineering
│   ├── model.py           # Training and prediction
    ├── sentiment.py       # News sentiment analysis
│   └── utils.py           # Visualization helpers
│
├── stock_prediction.ipynb # Research notebook
├── app.py                 # Streamlit web app
├── requirements.txt       # Dependencies
└── README.md
```

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
| News Sentiment | Sentiment | VADER analysis of Yahoo Finance headlines |

## Sentiment Analysis
Fetches latest stock specific news from Yahoo Finance and analyzes sentiment 
using VADER. Combines ML prediction with sentiment for stronger signals:
- **STRONG BUY** — Model predicts UP + positive/neutral sentiment
- **WEAK BUY** — Model predicts UP but sentiment is negative
- **STRONG SELL** — Model predicts DOWN + negative/neutral sentiment
- **WEAK SELL** — Model predicts DOWN but sentiment is positive

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
| **Random Forest** | **52-56%** ✅ |

## Strategy Results
| Metric | Result |
|--------|--------|
| Test Period | 2024-2025 |
| Buy & Hold Return | -3.05% |
| ML Strategy Return | **-0.11%** |
| Outperformance | **+2.94%** |

> During a declining market period the ML strategy preserved capital 
> better than passive holding — demonstrating downside protection.

## Key Findings
- Volume is the most important predictive feature (9.09% importance)
- Volatility and Bollinger Band Width outperform price-based features
- Model shows asymmetric performance — better at identifying DOWN days (61%) than UP days (44%)
- Logistic Regression underperformed below random chance — confirming stock direction is a non-linear problem
- Efficient Market Hypothesis limits accuracy of technical indicators alone
- Model accuracy varies by stock — 50-56% depending on market efficiency
- Sentiment analysis strengthens signals when aligned with model prediction

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
- Dynamic model training per stock — no single pre-trained model
- STRONG/WEAK BUY or SELL signal with adjusted confidence
- Real-time news sentiment from Yahoo Finance via yfinance
- Price chart with moving averages
- RSI indicator chart
- Current technical indicator values
- Strategy performance vs buy and hold

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Note:** Run after 3:30 PM IST for accurate next-day predictions.

## Limitations
- Technical indicators based on historical patterns only
- Sentiment analysis limited to English language news
- Model performance varies with market regime changes
- Transaction costs not accounted for in strategy returns
- Not suitable for live trading without proper risk management

## Future Work
- ~~Add news sentiment analysis~~ ✅ Completed
- Include fundamental indicators (P/E ratio, EPS)
- Implement walk-forward validation
- Add Sharpe ratio and maximum drawdown metrics
- Expand to Hindi financial news sources
- Deploy to Streamlit Cloud for public access

## Author
Samarth Patel
github.com/samarthpatel03