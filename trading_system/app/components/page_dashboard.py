"""
components/page_dashboard.py
Dashboard overview page
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import os
from datetime import datetime
from utils.charts import equity_curve, backtest_metrics_bar
from utils.table import dark_table

from pathlib import Path
BASE      = Path(__file__).resolve().parent.parent.parent
OUTPUTS   = BASE / "outputs"
PROCESSED = BASE / "data" / "processed"

LIVE_TICKERS = ["AAPL","MSFT","GOOGL","AMZN","META",
                "TSLA","NVDA","JPM","JNJ","SPY"]


@st.cache_data
def load():
    bt  = pd.read_csv(os.path.join(OUTPUTS,   "backtest_results.csv"))
    pv  = pd.read_csv(os.path.join(OUTPUTS,   "portfolio_values.csv"))
    xgb = pd.read_csv(os.path.join(PROCESSED, "xgb_predictions.csv"))
    wsb = pd.read_csv(os.path.join(PROCESSED, "wsb_sentiment.csv"))
    pv["Date"]  = pd.to_datetime(pv["Date"])
    xgb["Date"] = pd.to_datetime(xgb["Date"])
    return bt, pv, xgb, wsb


def fetch_live_prices():
    prices = []
    for ticker in LIVE_TICKERS:
        try:
            hist = yf.Ticker(ticker).history(period="2d")
            if len(hist) >= 2:
                curr  = hist["Close"].iloc[-1]
                prev  = hist["Close"].iloc[-2]
                chg   = curr - prev
                chg_p = (chg / prev) * 100
                vol   = hist["Volume"].iloc[-1]
                prices.append({
                    "ticker" : ticker,
                    "price"  : curr,
                    "change" : chg,
                    "chg_pct": chg_p,
                    "volume" : vol,
                })
        except Exception:
            pass
    return prices


def render():
    st.markdown("## 📊 Dashboard")
    st.markdown("<p style='color:#94a3b8;margin-top:-12px'>"
                "Real-time snapshot of model performance and signals</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    bt, pv, xgb, wsb = load()

    xgb_row = bt[bt["Strategy"] == "XGBoost"].iloc[0]
    spy_row = bt[bt["Strategy"] == "SPY Buy&Hold"].iloc[0]

    # ── KPI row ────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📈 XGBoost Return",
                  xgb_row["Total Return"],
                  f"vs SPY {spy_row['Total Return']}")
    with col2:
        st.metric("⚡ Sharpe Ratio",
                  xgb_row["Sharpe Ratio"],
                  "Industry benchmark >1.0")
    with col3:
        st.metric("🎯 Win Rate", xgb_row["Win Rate"])
    with col4:
        st.metric("📉 Max Drawdown", xgb_row["Max Drawdown"])
    with col5:
        latest     = xgb["Date"].max()
        latest_xgb = xgb[xgb["Date"] == latest]
        avg_conf   = latest_xgb["xgb_prob"].mean()
        st.metric("🤖 Avg Confidence",
                  f"{avg_conf*100:.1f}%", "Latest signals")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ─────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Portfolio Growth")
        fig = equity_curve(pv)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Strategy Returns")
        fig2 = backtest_metrics_bar(bt)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Signals + Model summary ────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Latest Signals")
        latest_date = xgb["Date"].max()
        latest_pred = (xgb[xgb["Date"] == latest_date]
                       .sort_values("xgb_prob", ascending=False))
        st.caption(f"As of {latest_date.strftime('%Y-%m-%d')}")

        for _, row in latest_pred.iterrows():
            prob   = row["xgb_prob"]
            ticker = row["Ticker"]
            signal = "BUY" if prob > 0.5 else "SELL"
            color  = "#10b981" if signal == "BUY" else "#ef4444"
            pct    = int(prob * 100)
            col_a, col_b, col_c, col_d = st.columns([1, 3, 1, 1])
            with col_a:
                st.markdown(f"**{ticker}**")
            with col_b:
                st.progress(pct)
            with col_c:
                st.markdown(f"{prob:.2f}")
            with col_d:
                st.markdown(
                    f"<span style='color:{color};font-weight:700'>"
                    f"{signal}</span>",
                    unsafe_allow_html=True)

    with col2:
        st.subheader("📋 Model Summary")
        display_cols = ["Strategy","Total Return","Sharpe Ratio",
                        "Win Rate","Max Drawdown"]
        dark_table(bt[display_cols].reset_index(drop=True))

    st.markdown("---")

    # ── Dataset stats ──────────────────────────────────────
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stocks Covered",      "44", "S&P 500 large caps")
    col2.metric("Trading Days",        "106,232", "2015–2024")
    col3.metric("WSB Posts",           f"{len(wsb):,}", "FinBERT scored")
    col4.metric("Sentiment Coverage",  "15.1%", "of all trading days")

    st.markdown("---")

    # ── Live price feed ────────────────────────────────────
    st.subheader("🔴 Live Market Prices")

    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.caption(
            f"Data from Yahoo Finance · "
            f"Last fetched: {datetime.now().strftime('%H:%M:%S')}")
    with col_hdr2:
        refresh = st.button("🔄 Refresh Prices",
                            key="live_refresh")

    with st.spinner("Fetching live prices..."):
        prices = fetch_live_prices()

    if not prices:
        st.warning("Could not fetch live prices. "
                   "Check your internet connection.")
    else:
        cols = st.columns(5)
        for i, p in enumerate(prices):
            with cols[i % 5]:
                color = "#10b981" if p["chg_pct"] >= 0 else "#ef4444"
                arrow = "▲" if p["chg_pct"] >= 0 else "▼"
                bg    = ("rgba(16,185,129,0.08)"
                         if p["chg_pct"] >= 0
                         else "rgba(239,68,68,0.08)")
                st.markdown(f"""
                <div style='background:#1e293b;
                            border:1px solid #334155;
                            border-left:3px solid {color};
                            border-radius:10px;
                            padding:14px 16px;
                            margin-bottom:10px;'>
                    <div style='color:#64748b;font-size:0.68rem;
                                font-weight:700;letter-spacing:0.08em;
                                text-transform:uppercase;
                                margin-bottom:6px;'>
                        {p["ticker"]}</div>
                    <div style='color:#f1f5f9;font-size:1.35rem;
                                font-weight:700;margin-bottom:4px;'>
                        ${p["price"]:.2f}</div>
                    <div style='color:{color};font-size:0.82rem;
                                font-weight:600;margin-bottom:6px;'>
                        {arrow} {p["chg_pct"]:+.2f}%
                        <span style='color:#64748b;font-weight:400;
                                     font-size:0.78rem;'>
                          ({p["change"]:+.2f})</span>
                    </div>
                    <div style='color:#475569;font-size:0.7rem;'>
                        Vol {p["volume"]/1e6:.1f}M</div>
                </div>
                """, unsafe_allow_html=True)
