"""
Quantitative Market Analysis
"""

import datetime
import warnings
import sys

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

sys.path.append(".")
warnings.filterwarnings("ignore")

from src.data_loader import download_stock_data
from src.features import calculate_features
from src.config import FEATURES
from src.model import (
    train_test_split_timeseries,
    train_random_forest,
    evaluate_model,
    calculate_strategy_returns,
    predict_with_sentiment,
    walk_forward_validation,
    calculate_risk_metrics,
)
from src.sentiment import get_news_sentiment, interpret_sentiment

st.set_page_config(
    page_title="QMA · Market Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700;800&display=swap');

:root {
    --bg:       #f7f5f0;
    --surface:  #ffffff;
    --border:   #e8e4dc;
    --border2:  #d4cfc4;
    --text-1:   #1a1714;
    --text-2:   #5a5550;
    --text-3:   #9a948c;
    --accent:   #e85d3a;
    --green:    #1a9e6e;
    --green-bg: #eaf7f1;
    --red:      #d93025;
    --red-bg:   #fdecea;
    --blue:     #2d5be3;
    --amber:    #d97706;
    --amber-bg: #fef3e0;
    --shadow:   0 1px 3px rgba(0,0,0,.05), 0 4px 14px rgba(0,0,0,.04);
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-1) !important;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--bg) !important; }
.block-container { padding: 2rem 2.6rem 4rem !important; max-width: 1320px; }

.nav-bar {
    display:flex; align-items:center; justify-content:space-between;
    padding-bottom:1.2rem; margin-bottom:1.8rem;
    border-bottom:2px solid var(--text-1);
}
.nav-logo { font-family:'Playfair Display',serif; font-size:1.3rem; font-weight:800; color:var(--text-1); letter-spacing:-0.02em; }
.nav-logo span { color:var(--accent); }
.nav-meta { font-family:'DM Mono',monospace; font-size:0.65rem; color:var(--text-3); }

.sec-head {
    font-family:'DM Mono',monospace; font-size:0.57rem; font-weight:500;
    letter-spacing:0.22em; text-transform:uppercase; color:var(--text-3);
    display:flex; align-items:center; gap:0.5rem; margin:0 0 1rem;
}
.sec-head::after { content:''; flex:1; height:1px; background:var(--border); }

.card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:1.3rem 1.5rem; box-shadow:var(--shadow); }

/* Signal */
.sig-hero { border-radius:14px; padding:1.8rem 2rem; }
.sig-bull  { background:var(--green-bg); border:1.5px solid #9adec0; }
.sig-bear  { background:var(--red-bg);   border:1.5px solid #f4aaa4; }
.sig-wbull { background:var(--amber-bg); border:1.5px solid #f0d080; }
.sig-wbear { background:var(--amber-bg); border:1.5px solid #f0d080; }
.sig-dir { font-family:'Playfair Display',serif; font-size:2.6rem; font-weight:800; letter-spacing:-0.03em; line-height:1; margin-bottom:0.3rem; }
.sig-lbl { font-family:'DM Mono',monospace; font-size:0.66rem; font-weight:500; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:1rem; }
.c-bull  { color:var(--green); }
.c-bear  { color:var(--red); }
.c-mixed { color:var(--amber); }
.conf-track { height:5px; background:rgba(0,0,0,.09); border-radius:99px; overflow:hidden; margin-bottom:0.3rem; }
.cf-bull  { height:100%; background:var(--green); border-radius:99px; }
.cf-bear  { height:100%; background:var(--red);   border-radius:99px; }
.cf-mixed { height:100%; background:var(--amber); border-radius:99px; }
.conf-txt { font-family:'DM Mono',monospace; font-size:0.61rem; color:var(--text-3); }

/* Key-value rows */
.kv { display:flex; justify-content:space-between; align-items:center; padding:0.44rem 0; border-bottom:1px solid var(--border); }
.kv:last-child { border-bottom:none; }
.kv-k { font-size:0.72rem; color:var(--text-2); }
.kv-v { font-family:'DM Mono',monospace; font-size:0.75rem; font-weight:500; color:var(--text-1); }
.kv-pos { color:var(--green) !important; }
.kv-neg { color:var(--red)   !important; }

/* Metric strip */
.mstrip { display:grid; grid-template-columns:repeat(3,1fr); gap:0.5rem; }
.ms { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:0.85rem 1rem; box-shadow:var(--shadow); }
.ms-lbl { font-family:'DM Mono',monospace; font-size:0.54rem; color:var(--text-3); letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.22rem; }
.ms-val { font-size:1.15rem; font-weight:600; color:var(--text-1); line-height:1.1; }
.ms-d   { font-family:'DM Mono',monospace; font-size:0.58rem; margin-top:0.1rem; }
.d-pos  { color:var(--green); }
.d-neg  { color:var(--red); }
.d-neu  { color:var(--text-3); }

/* Headlines */
.hl { display:flex; justify-content:space-between; align-items:flex-start; gap:0.8rem; padding:0.55rem 0; border-bottom:1px solid var(--border); }
.hl:last-child { border-bottom:none; }
.hl-txt  { font-size:0.75rem; color:var(--text-2); flex:1; line-height:1.45; }
.hl-chip { font-family:'DM Mono',monospace; font-size:0.63rem; font-weight:500; white-space:nowrap; padding:0.12rem 0.42rem; border-radius:4px; }
.chip-pos { background:var(--green-bg); color:var(--green); }
.chip-neg { background:var(--red-bg);   color:var(--red); }
.chip-neu { background:var(--bg); color:var(--text-3); border:1px solid var(--border); }

/* Input / Button */
[data-testid="stTextInput"] input {
    background:var(--surface) !important; border:1.5px solid var(--border2) !important;
    border-radius:8px !important; color:var(--text-1) !important;
    font-family:'DM Sans',sans-serif !important; font-size:0.9rem !important;
    padding:0.52rem 0.85rem !important;
}
[data-testid="stTextInput"] input:focus { border-color:var(--accent) !important; box-shadow:0 0 0 3px rgba(232,93,58,.1) !important; }
[data-testid="stTextInput"] label { font-family:'DM Mono',monospace !important; font-size:0.62rem !important; letter-spacing:0.12em !important; text-transform:uppercase !important; color:var(--text-3) !important; }
.stButton>button { background:var(--text-1) !important; color:#fff !important; border:none !important; border-radius:8px !important; font-family:'DM Sans',sans-serif !important; font-size:0.86rem !important; font-weight:500 !important; padding:0.55rem 1.4rem !important; transition:opacity .15s !important; }
.stButton>button:hover { opacity:0.82 !important; }

hr { border:none !important; border-top:1px solid var(--border) !important; margin:1.6rem 0 !important; }
[data-testid="stAlert"] { background:var(--amber-bg) !important; border:1px solid #f0d080 !important; border-radius:8px !important; color:var(--amber) !important; font-size:0.76rem !important; }
.stCaption,[data-testid="stCaptionContainer"] { color:var(--text-3) !important; font-size:0.64rem !important; }
[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:10px !important; font-family:'DM Mono',monospace !important; font-size:0.73rem !important; }
[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:0.9rem 1.1rem; box-shadow:var(--shadow); }
[data-testid="stMetricLabel"] { font-family:'DM Mono',monospace !important; font-size:0.54rem !important; color:var(--text-3) !important; letter-spacing:0.12em !important; text-transform:uppercase !important; }
[data-testid="stMetricValue"] { font-size:1.2rem !important; font-weight:600 !important; color:var(--text-1) !important; }
[data-testid="stMetricDelta"] { font-family:'DM Mono',monospace !important; font-size:0.62rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Chart theme
# ─────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#ffffff",
    "axes.edgecolor":    "#e8e4dc",
    "axes.labelcolor":   "#9a948c",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       "#9a948c",
    "ytick.color":       "#9a948c",
    "text.color":        "#5a5550",
    "grid.color":        "#f0ece4",
    "grid.linestyle":    "-",
    "grid.linewidth":    0.6,
    "axes.grid":         True,
    "legend.facecolor":  "#ffffff",
    "legend.edgecolor":  "#e8e4dc",
    "legend.framealpha": 1,
    "font.family":       "sans-serif",
    "font.size":         8.5,
    "lines.linewidth":   1.6,
})

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def kv(label, val, color=""):
    cls = f"kv-{color}" if color else ""
    return f'<div class="kv"><span class="kv-k">{label}</span><span class="kv-v {cls}">{val}</span></div>'

def ms(label, val, delta="", dc="d-neu"):
    d = f'<div class="ms-d {dc}">{delta}</div>' if delta else ""
    return f'<div class="ms"><div class="ms-lbl">{label}</div><div class="ms-val">{val}</div>{d}</div>'

def sig_cfg(sig):
    if "STRONG BUY"  in sig: return "sig-bull",  "c-bull",  "cf-bull",  "📈 LONG"
    if "STRONG SELL" in sig: return "sig-bear",  "c-bear",  "cf-bear",  "📉 SHORT"
    if "WEAK BUY"    in sig: return "sig-wbull", "c-mixed", "cf-mixed", "↗ WEAK LONG"
    return                          "sig-wbear", "c-mixed", "cf-mixed", "↘ WEAK SHORT"

@st.cache_data(ttl=3600, show_spinner=False)
def load_and_prepare(ticker, start, end):
    return calculate_features(download_stock_data(ticker, start, end))

@st.cache_data(ttl=3600, show_spinner=False)
def train_cached(ticker, _df):
    X_train, X_test, y_train, y_test, split = train_test_split_timeseries(_df, FEATURES)
    return train_random_forest(X_train, y_train), X_train, X_test, y_train, y_test, split

# ─────────────────────────────────────────────────────────────
# Nav
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
  <div class="nav-logo">Quant<span>.</span>Market</div>
  <span class="nav-meta">NSE / BSE &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; VADER Sentiment</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────────────────────
ic, bc, _ = st.columns([2, 0.65, 3.5])
with ic:
    raw    = st.text_input("Ticker", value="RELIANCE.NS", placeholder="e.g. INFY.NS · TCS.NS · HDFCBANK.NS")
    ticker = raw.strip().upper()
with bc:
    st.markdown("<div style='margin-top:1.68rem'>", unsafe_allow_html=True)
    run = st.button("Analyse →", type="primary", width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)

if ticker and not ticker.endswith(".NS") and not ticker.endswith(".BO"):
    st.warning(f"NSE tickers end with `.NS` — did you mean **{ticker}.NS**?")

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────
if run:
    today = datetime.datetime.today()
    end   = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    start = (today - datetime.timedelta(days=1095)).strftime("%Y-%m-%d")

    prog = st.progress(0, text="Downloading market data…")
    try:
        df = load_and_prepare(ticker, start, end)
    except Exception as e:
        st.error(str(e)); st.stop()

    if df.empty or len(df) < 100:
        st.error("Not enough data for this ticker."); st.stop()

    prog.progress(30, text="Training model…")
    model, X_train, X_test, y_train, y_test, split = train_cached(ticker, df)
    predictions, accuracy, _ = evaluate_model(model, X_test, y_test)

    prog.progress(60, text="Fetching sentiment…")
    try:
        sent_score, headlines = get_news_sentiment(ticker)
    except Exception:
        sent_score, headlines = 0.0, []

    prog.progress(80, text="Computing metrics…")
    wf_accs, wf_avg    = walk_forward_validation(df, FEATURES)
    buy_hold, strategy = calculate_strategy_returns(df, predictions, split)
    risk               = calculate_risk_metrics(df, predictions, split)
    pred, conf, signal = predict_with_sentiment(model, df, FEATURES, sent_score)
    prog.progress(100, text="Done ✓"); prog.empty()

    # Derived
    outperf    = strategy - buy_hold
    last_close = df["Close"].iloc[-1]
    rsi_val    = df["RSI"].iloc[-1]
    conf_pct   = int(conf * 100)
    macd_val   = df["MACD"].iloc[-1]
    sma_cross  = "Golden Cross ↑" if df["SMA_20"].iloc[-1] > df["SMA_50"].iloc[-1] else "Death Cross ↓"
    sc, cc, fc, dir_lbl = sig_cfg(signal)

    # Cumulative returns
    test_ret  = df["Close"].pct_change().iloc[split + 1:].values
    pred_aln  = predictions[:len(test_ret)]
    strat_ret = test_ret * pred_aln
    cum_mkt   = np.cumprod(1 + test_ret)
    cum_str   = np.cumprod(1 + strat_ret)
    test_dts  = df.index[split + 1: split + 1 + len(test_ret)]

    # Trade log
    rows, in_t, ep, ed = [], False, None, None
    for date, p in zip(test_dts, pred_aln):
        price = df.loc[date, "Close"]
        if p == 1 and not in_t:
            in_t, ep, ed = True, price, date
        elif p == 0 and in_t:
            r = (price - ep) / ep * 100
            rows.append({"Entry": ed.date(), "Exit": date.date(),
                          "Entry ₹": round(ep, 2), "Exit ₹": round(price, 2), "Return %": round(r, 2)})
            in_t = False
    if in_t:
        lp = df["Close"].iloc[-1]
        rows.append({"Entry": ed.date(), "Exit": "Open",
                      "Entry ₹": round(ep, 2), "Exit ₹": round(lp, 2),
                      "Return %": round((lp - ep) / ep * 100, 2)})
    trade_df = pd.DataFrame(rows)

    # ══════════════════════════════════════════════════════════
    # A · Signal + Snapshot
    # ══════════════════════════════════════════════════════════
    st.markdown('<p class="sec-head">Signal · Next Session</p>', unsafe_allow_html=True)

    col_sig, col_snap = st.columns([1, 1])

    with col_sig:
        st.markdown(f"""
        <div class="sig-hero {sc}">
          <div class="sig-dir {cc}">{dir_lbl}</div>
          <div class="sig-lbl {cc}">{signal}</div>
          <div class="conf-track"><div class="{fc}" style="width:{conf_pct}%"></div></div>
          <div class="conf-txt">Confidence · {conf_pct}%</div>
        </div>""", unsafe_allow_html=True)

    with col_snap:
        rsi_c  = "neg" if rsi_val > 70 else "pos" if rsi_val < 30 else ""
        macd_c = "pos" if macd_val > 0 else "neg"
        cross_c = "pos" if "Golden" in sma_cross else "neg"
        op_c   = "pos" if outperf > 0 else "neg"
        sh_c   = "pos" if risk["sharpe_strategy"] > risk["sharpe_market"] else "neg"
        st.markdown(f"""
        <div class="card">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:0 1.2rem">
            <div>
              {kv("Last Close",  f"₹{last_close:,.2f}")}
              {kv("Session",     str(df.index[-1].date()))}
              {kv("RSI 14",      f"{rsi_val:.1f}", rsi_c)}
              {kv("MACD",        f"{macd_val:+.3f}", macd_c)}
              {kv("MA Cross",    sma_cross, cross_c)}
            </div>
            <div>
              {kv("Accuracy",    f"{accuracy*100:.1f}%")}
              {kv("WF Avg",      f"{wf_avg*100:.1f}%")}
              {kv("ML Return",   f"{strategy:+.2f}%", op_c)}
              {kv("Outperform",  f"{outperf:+.2f}%", op_c)}
              {kv("Sharpe ML",   f"{risk['sharpe_strategy']:.3f}", sh_c)}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # B · Price chart — full width
    # ══════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="sec-head">Price · Moving Averages</p>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.plot(df.index, df["Close"],  color="#1a1714", linewidth=1.4, label="Close",  zorder=3)
    ax.plot(df.index, df["SMA_20"], color="#e85d3a", linewidth=1.1, label="SMA 20", linestyle="--", zorder=2)
    ax.plot(df.index, df["SMA_50"], color="#2d5be3", linewidth=1.1, label="SMA 50", linestyle="--", zorder=2)
    ax.fill_between(df.index, df["Close"], df["SMA_50"],
                    where=(df["Close"] >= df["SMA_50"]), color="#1a9e6e", alpha=0.05)
    ax.fill_between(df.index, df["Close"], df["SMA_50"],
                    where=(df["Close"] <  df["SMA_50"]), color="#d93025", alpha=0.05)
    ax.axvline(df.index[split], color="#9a948c", linewidth=0.9, linestyle=":", label="Train / Test")
    ax.set_ylabel("Price (₹)"); ax.legend(fontsize=8)
    ax.set_title(f"{ticker} — Close · SMA 20 · SMA 50", fontsize=9, color="#5a5550", pad=8)
    fig.tight_layout()
    st.pyplot(fig, width='stretch')

    # ══════════════════════════════════════════════════════════
    # C · Strategy returns chart + metric strip
    # ══════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="sec-head">Strategy vs Buy & Hold</p>', unsafe_allow_html=True)

    ch_col, mt_col = st.columns([2.2, 1])

    with ch_col:
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(test_dts, cum_mkt, color="#1a1714", linewidth=1.6, label="Buy & Hold")
        ax.plot(test_dts, cum_str, color="#e85d3a", linewidth=1.6, label="ML Strategy")
        ax.fill_between(test_dts, cum_mkt, cum_str,
                        where=(cum_str >= cum_mkt), color="#1a9e6e", alpha=0.13, label="ML ahead")
        ax.fill_between(test_dts, cum_mkt, cum_str,
                        where=(cum_str <  cum_mkt), color="#d93025", alpha=0.10, label="ML behind")
        ax.axhline(1.0, color="#d4cfc4", linewidth=0.8, linestyle="--")
        ax.annotate(f"B&H {buy_hold:+.1f}%",
                    xy=(test_dts[-1], cum_mkt[-1]), xytext=(8, 0),
                    textcoords="offset points", fontsize=7.5, color="#1a1714", va="center")
        ax.annotate(f"ML {strategy:+.1f}%",
                    xy=(test_dts[-1], cum_str[-1]), xytext=(8, 0),
                    textcoords="offset points", fontsize=7.5, color="#e85d3a", va="center")
        ax.set_ylabel("Growth of ₹1"); ax.legend(fontsize=8)
        ax.set_title("Cumulative Returns — Test Period", fontsize=9, color="#5a5550", pad=8)
        fig.tight_layout()
        st.pyplot(fig, width='stretch')

    with mt_col:
        dd_red = abs(risk["mdd_market"]) - abs(risk["mdd_strategy"])
        op_c  = "d-pos" if outperf > 0      else "d-neg"
        sh_c  = "d-pos" if risk["sharpe_strategy"] > risk["sharpe_market"] else "d-neg"
        dd_c  = "d-pos" if dd_red > 0       else "d-neg"
        st.markdown(f"""
        <div class="mstrip">
          {ms("Buy & Hold",   f"{buy_hold:+.2f}%")}
          {ms("ML Strategy",  f"{strategy:+.2f}%",  f"{outperf:+.2f}% α", op_c)}
          {ms("Sharpe · B&H", f"{risk['sharpe_market']:.3f}")}
          {ms("Sharpe · ML",  f"{risk['sharpe_strategy']:.3f}", f"{risk['sharpe_strategy']-risk['sharpe_market']:+.3f}", sh_c)}
          {ms("Max DD · B&H", f"{risk['mdd_market']:.2f}%")}
          {ms("Max DD · ML",  f"{risk['mdd_strategy']:.2f}%",  f"{dd_red:.2f}pp saved", dd_c)}
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # D · Sentiment + Headlines
    # ══════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="sec-head">News Sentiment</p>', unsafe_allow_html=True)

    s_col, h_col = st.columns([1, 2])

    with s_col:
        sb  = "#eaf7f1" if sent_score > 0.05 else "#fdecea" if sent_score < -0.05 else "#f7f5f0"
        sbd = "#9adec0" if sent_score > 0.05 else "#f4aaa4" if sent_score < -0.05 else "#e8e4dc"
        sc2 = "#1a9e6e" if sent_score > 0.05 else "#d93025" if sent_score < -0.05 else "#9a948c"
        st.markdown(f"""
        <div style="background:{sb};border:1.5px solid {sbd};border-radius:14px;padding:1.6rem 1.8rem">
          <div style="font-family:'DM Mono',monospace;font-size:0.56rem;color:{sc2};letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.35rem">Sentiment</div>
          <div style="font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:800;color:{sc2};line-height:1;margin-bottom:0.25rem">{interpret_sentiment(sent_score)}</div>
          <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:{sc2}">{sent_score:+.4f}</div>
        </div>""", unsafe_allow_html=True)

    with h_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if headlines:
            for h in headlines[:7]:
                s    = h["sentiment"]
                chip = "chip-pos" if s > 0.05 else "chip-neg" if s < -0.05 else "chip-neu"
                lbl  = f"▲ {s:+.3f}" if s > 0.05 else f"▼ {s:+.3f}" if s < -0.05 else f"● {s:+.3f}"
                st.markdown(f'<div class="hl"><span class="hl-txt">{h["headline"]}</span><span class="hl-chip {chip}">{lbl}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:0.75rem;color:var(--text-3)">No recent headlines.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # E · Trade Log
    # ══════════════════════════════════════════════════════════
    if len(trade_df) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="sec-head">Simulated Trade Log</p>', unsafe_allow_html=True)

        wins    = (trade_df["Return %"] > 0).sum()
        losses  = (trade_df["Return %"] <= 0).sum()
        avg_ret = trade_df["Return %"].mean()

        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Trades",     len(trade_df))
        t2.metric("Win Rate",   f"{wins/len(trade_df)*100:.0f}%", delta=f"{wins}W / {losses}L")
        t3.metric("Avg Return", f"{avg_ret:+.2f}%")
        t4.metric("Best",       f"{trade_df['Return %'].max():+.2f}%")
        t5.metric("Worst",      f"{trade_df['Return %'].min():+.2f}%")

        st.markdown("<br style='margin:0.2rem'>", unsafe_allow_html=True)

        def colour_ret(val):
            if isinstance(val, float):
                return f"color:{'#1a9e6e' if val > 0 else '#d93025'};font-weight:600"
            return ""

        st.dataframe(
            trade_df.style
                .map(colour_ret, subset=["Return %"])
                .format({"Entry ₹": "₹{:.2f}", "Exit ₹": "₹{:.2f}", "Return %": "{:+.2f}%"}),
            width='stretch',
            height=min(400, 58 + len(trade_df) * 35),
        )
        st.caption("Long only · cash on predicted DOWN days · no short selling")

    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;font-size:0.62rem;color:#9a948c">
      <span>Quant.Market · Random Forest · Walk-Forward · VADER</span>
      <span>Not financial advice · {df.index[-1].date()} · Samarth Patel</span>
    </div>""", unsafe_allow_html=True)