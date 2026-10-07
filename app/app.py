"""
Quantum Stock Trend Predictor — Streamlit Dashboard
Beginner-friendly interface for the 4-Qubit Variational Quantum Classifier (VQC).
The Quantum VQC (PennyLane) is the central and ONLY prediction model presented.
All controls are on the main dashboard with NO sidebar.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Path setup
# ──────────────────────────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection import SUPPORTED_TICKERS
from src.prediction import get_latest_stock_features, predict_trend

# ──────────────────────────────────────────────────────────────────────────────
# Page config — Sidebar completely collapsed and hidden
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Stock Trend Predictor",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────────────────────────────
# Global Styling
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Completely hide sidebar and collapse toggle button */
[data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}

.hero-banner {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 2.2rem 2rem 2rem 2rem;
    margin-bottom: 1.6rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25);
}
.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin-bottom: 0.4rem;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #a5b4fc;
    font-weight: 400;
}
.quantum-badge {
    display: inline-block;
    background: rgba(99,102,241,0.25);
    border: 1px solid #6366f1;
    color: #a5b4fc;
    font-size: 0.78rem;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    margin-top: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}

.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1e1b4b;
    margin-top: 0.8rem;
    margin-bottom: 0.8rem;
}

.pred-card {
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin: 1rem auto;
    max-width: 580px;
}
.pred-card-up {
    background: linear-gradient(145deg, #ecfdf5, #d1fae5);
    border: 2px solid #10b981;
}
.pred-card-down {
    background: linear-gradient(145deg, #fff1f2, #ffe4e6);
    border: 2px solid #f43f5e;
}
.pred-label {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.4rem;
    color: #4b5563;
}
.pred-direction-up {
    font-size: 4.2rem;
    font-weight: 900;
    color: #059669;
    line-height: 1;
    margin: 0.3rem 0;
    letter-spacing: 2px;
}
.pred-direction-down {
    font-size: 4.2rem;
    font-weight: 900;
    color: #e11d48;
    line-height: 1;
    margin: 0.3rem 0;
    letter-spacing: 2px;
}
.pred-emoji {
    font-size: 2.5rem;
    margin-bottom: 0.2rem;
}
.pred-confidence {
    font-size: 1.05rem;
    color: #374151;
    margin-top: 0.6rem;
    font-weight: 600;
}
.pred-explanation {
    font-size: 0.92rem;
    color: #4b5563;
    margin-top: 0.8rem;
    line-height: 1.5;
}

.conf-bar-outer {
    background: #e2e8f0;
    border-radius: 999px;
    height: 14px;
    width: 82%;
    margin: 0.8rem auto 0.4rem auto;
    overflow: hidden;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
}
.conf-bar-fill-up {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #10b981, #34d399);
}
.conf-bar-fill-down {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #f43f5e, #fb7185);
}

.flow-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
    margin: 0.5rem 0;
}
.flow-step {
    background: linear-gradient(135deg, #eef2ff, #e0e7ff);
    border: 1.5px solid #a5b4fc;
    border-radius: 10px;
    padding: 0.55rem 1.4rem;
    text-align: center;
    font-size: 0.85rem;
    font-weight: 600;
    color: #3730a3;
    min-width: 210px;
    box-shadow: 0 2px 6px rgba(99,102,241,0.08);
}
.flow-step-highlight {
    background: linear-gradient(135deg, #312e81, #4f46e5);
    color: white;
    border: 1.5px solid #4f46e5;
}
.flow-arrow {
    font-size: 1.25rem;
    color: #818cf8;
    margin: 0.05rem 0;
    line-height: 1;
}

.disclaimer {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #78350f;
    line-height: 1.5;
    margin-top: 1rem;
}

.badge-live {
    display: inline-block;
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #6ee7b7;
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-archive {
    display: inline-block;
    background: #fffbeb;
    color: #92400e;
    border: 1px solid #fcd34d;
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 600;
}

.fancy-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #c7d2fe, transparent);
    border: none;
    margin: 1.8rem 0;
    border-radius: 999px;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Caching
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_features(ticker: str):
    return get_latest_stock_features(ticker)


# ──────────────────────────────────────────────────────────────────────────────
# TOP HERO BANNER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">⚛️ QUANTUM STOCK PREDICTOR</div>
    <div class="hero-subtitle">Quantum Machine Learning for Next-Day Stock Trend Prediction</div>
    <span class="quantum-badge">🔬 Variational Quantum Classifier · PennyLane · 4 Qubits · Simulated Circuit</span>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. STOCK SELECTION (MAIN DASHBOARD)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📌 Stock Selection</div>', unsafe_allow_html=True)

sel_col1, sel_col2 = st.columns([1, 1])
with sel_col1:
    selected_ticker = st.selectbox(
        "Select a Stock:",
        options=SUPPORTED_TICKERS,
        index=0,
        help="Choose one of the supported stocks (NSE or US)"
    )

with sel_col2:
    custom_ticker = st.text_input(
        "Or enter a custom ticker (e.g. AAPL, MSFT, RELIANCE.NS):",
        value="",
        placeholder="e.g. AAPL, MSFT, RELIANCE.NS, TATAMOTORS.NS"
    ).strip().upper()

active_ticker = custom_ticker if custom_ticker else selected_ticker

analyze_btn = st.button("⚡ Analyze & Predict", type="primary", use_container_width=True)
if analyze_btn:
    fetch_features.clear()
    st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
try:
    with st.spinner(f"Loading market data for **{active_ticker}**…"):
        featured_df, overview = fetch_features(active_ticker)
except Exception as err:
    st.error(f"⚠️ Could not load data for **{active_ticker}**: {err}")
    st.info("Please verify the ticker symbol or check your internet connection.")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# 2. STOCK INFORMATION
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📊 Stock Information</div>', unsafe_allow_html=True)

is_live = overview.get('is_live', False)
source_badge = (
    '<span class="badge-live">🟢 Live Data</span>'
    if is_live else
    '<span class="badge-archive">🟡 Historical Archive</span>'
)

st.markdown(f"**Selected Stock:** `{active_ticker}` &nbsp;&nbsp; {source_badge}", unsafe_allow_html=True)

if not is_live:
    st.warning(
        f"⚠️ **Archive Mode:** Live market data could not be fetched. "
        f"The prediction below is based on historical data from `{overview['latest_date']}`. "
        "This is not a current-market prediction."
    )

is_us_ticker = not (active_ticker.endswith(".NS") or active_ticker.endswith(".BO"))
currency_sym = "$" if is_us_ticker else "₹"

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Latest Close", f"{currency_sym}{overview['latest_close']:.2f}", f"{overview['daily_change']:+.2f}")
col_b.metric("Trading Date", overview['latest_date'])
col_c.metric("Daily Change", f"{currency_sym}{overview['daily_change']:+.2f}")
col_d.metric("Daily % Change", f"{overview['daily_pct']:+.2f}%")


# ──────────────────────────────────────────────────────────────────────────────
# 3. HISTORICAL STOCK CHART
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📈 Historical Stock Chart</div>', unsafe_allow_html=True)
st.caption(f"Historical candlestick price chart and daily trading volume for **{active_ticker}** with 20-day and 50-day moving averages.")

fig_price = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(f"{active_ticker} — Candlestick Price History", "Daily Trading Volume"),
    row_heights=[0.75, 0.25]
)

fig_price.add_trace(
    go.Candlestick(
        x=featured_df['Date'],
        open=featured_df['Open'],
        high=featured_df['High'],
        low=featured_df['Low'],
        close=featured_df['Close'],
        name="Price",
        increasing_line_color="#10b981",
        decreasing_line_color="#f43f5e"
    ),
    row=1, col=1
)
fig_price.add_trace(
    go.Scatter(
        x=featured_df['Date'], y=featured_df['SMA_20'],
        line=dict(color='#f59e0b', width=1.5), name="20-day Average"
    ),
    row=1, col=1
)
fig_price.add_trace(
    go.Scatter(
        x=featured_df['Date'], y=featured_df['SMA_50'],
        line=dict(color='#8b5cf6', width=1.5), name="50-day Average"
    ),
    row=1, col=1
)

vol_colors = ['#10b981' if c >= o else '#f43f5e'
              for c, o in zip(featured_df['Close'], featured_df['Open'])]
fig_price.add_trace(
    go.Bar(x=featured_df['Date'], y=featured_df['Volume'],
           marker_color=vol_colors, name="Volume", opacity=0.7),
    row=2, col=1
)

fig_price.update_layout(
    height=500,
    xaxis_rangeslider_visible=False,
    template="plotly_white",
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    font=dict(family="Inter, sans-serif")
)
st.plotly_chart(fig_price, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# 4. QUANTUM PREDICTION (MOST IMPORTANT SECTION)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-header">🎯 Quantum Prediction</div>', unsafe_allow_html=True)

try:
    with st.spinner("Running Quantum Circuit… ⚛️"):
        pred_result = predict_trend(
            ticker=active_ticker,
            model_name="Quantum VQC",
            featured_df=featured_df
        )
except Exception as pe:
    st.error(f"Quantum prediction failed: {pe}")
    st.info(
        "The Quantum VQC model needs to be trained first. "
        "Run `python run.py --quantum` from the project root to train the model."
    )
    st.stop()

is_up = pred_result['prediction'] == 1
direction = "UP" if is_up else "DOWN"
direction_emoji = "📈" if is_up else "📉"
card_class = "pred-card-up" if is_up else "pred-card-down"
dir_class = "pred-direction-up" if is_up else "pred-direction-down"
conf_class = "conf-bar-fill-up" if is_up else "conf-bar-fill-down"

prob = pred_result.get('predicted_probability')
confidence_pct = round(prob * 100, 1) if prob is not None else None

# Plain text dynamic explanation
if is_up:
    explanation = "Based on the latest available market data, the quantum model predicts an upward price trend."
else:
    explanation = "Based on the latest available market data, the quantum model predicts a downward price trend."

# Single-line HTML snippets without any leading spaces or newlines to ensure perfect rendering
conf_bar_html = f'<div class="conf-bar-outer"><div class="{conf_class}" style="width:{confidence_pct}%;"></div></div>' if confidence_pct is not None else ""
conf_text_html = f'<div class="pred-confidence">Confidence: <strong>{confidence_pct:.1f}%</strong></div>' if confidence_pct is not None else ""

# Guaranteed clean HTML string with zero leading indentation on lines
card_html = (
    f'<div class="pred-card {card_class}">'
    f'<div class="pred-emoji">{direction_emoji}</div>'
    f'<div class="pred-label">NEXT TRADING DAY PREDICTION</div>'
    f'<div class="{dir_class}">{direction}</div>'
    f'{conf_bar_html}'
    f'{conf_text_html}'
    f'<div class="pred-explanation">{explanation}</div>'
    f'</div>'
)

left_sp, pred_col, right_sp = st.columns([1, 2, 1])
with pred_col:
    st.markdown(card_html, unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚠️ <b>Academic Disclaimer:</b> This prediction is generated by a research quantum machine learning algorithm
    evaluating historical technical indicators. It does <b>NOT</b> constitute financial advice or guarantee future
    market performance. Stock markets are complex and influenced by many macroeconomic and news factors beyond technical indicators.
</div>
""", unsafe_allow_html=True)



# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2.5rem; padding:1.2rem; background:linear-gradient(135deg,#0f0c29,#302b63);
            border-radius:12px; text-align:center;">
    <div style="color:#a5b4fc; font-size:0.82rem; line-height:1.8;">
        ⚛️ <strong style="color:#c7d2fe;">Quantum Stock Trend Predictor</strong> &nbsp;|&nbsp;
        PennyLane VQC · 4 Qubits · Simulated Quantum Circuit &nbsp;|&nbsp;
        Academic Research Prototype &nbsp;|&nbsp;
        <span style="color:#f87171;">Not financial advice</span>
    </div>
</div>
""", unsafe_allow_html=True)
