"""
components/page_stock_analysis.py
Per-ticker price chart with technical indicators and sentiment
"""

import streamlit as st
import pandas as pd
import os
from utils.charts import stock_chart, sentiment_chart, prediction_chart
from utils.table import dark_table

from pathlib import Path
BASE      = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE / "data" / "processed"


@st.cache_data
def load():
    df  = pd.read_csv(os.path.join(PROCESSED, "final_dataset.csv"))
    wsb = pd.read_csv(os.path.join(PROCESSED, "wsb_sentiment.csv"))
    xgb = pd.read_csv(os.path.join(PROCESSED, "xgb_predictions.csv"))
    lst = pd.read_csv(os.path.join(PROCESSED, "lstm_predictions.csv"))
    df["Date"]  = pd.to_datetime(df["Date"])
    xgb["Date"] = pd.to_datetime(xgb["Date"])
    lst["Date"] = pd.to_datetime(lst["Date"])
    return df, wsb, xgb, lst


def render():
    st.markdown("## 📈 Stock Analysis")
    st.markdown("<p style='color:#94a3b8;margin-top:-12px'>"
                "Deep dive into price, indicators, sentiment "
                "and model signals per stock</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    df, wsb, xgb, lst = load()

    # ── Controls ───────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 2])

    SENTIMENT_TICKERS = sorted([
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "NVDA", "JPM", "JNJ", "SPY"
    ])
    ALL_TICKERS = sorted(df["Ticker"].unique().tolist())

    with col1:
        ticker = st.selectbox("Select Stock", ALL_TICKERS,
                              index=ALL_TICKERS.index("AAPL"))
    with col2:
        period = st.selectbox("Time Period",
                              ["1Y", "2Y", "5Y", "All"],
                              index=0)
    with col3:
        show_bb   = st.checkbox("Bollinger Bands", value=True)
        show_sent = st.checkbox("Show Sentiment",  value=True)

    # Filter by period
    stock_df = df[df["Ticker"] == ticker].copy()
    stock_df = stock_df.sort_values("Date")

    if period == "1Y":
        cutoff = stock_df["Date"].max() - pd.DateOffset(years=1)
    elif period == "2Y":
        cutoff = stock_df["Date"].max() - pd.DateOffset(years=2)
    elif period == "5Y":
        cutoff = stock_df["Date"].max() - pd.DateOffset(years=5)
    else:
        cutoff = stock_df["Date"].min()

    stock_df = stock_df[stock_df["Date"] >= cutoff]

    if stock_df.empty:
        st.warning("No data for this ticker.")
        return

    # ── Snapshot metrics ───────────────────────────────────
    latest = stock_df.iloc[-1]
    prev   = stock_df.iloc[-2] if len(stock_df) > 1 else latest
    pct    = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Last Price",   f"${latest['Close']:.2f}",
                f"{pct:+.2f}%")
    col2.metric("RSI",          f"{latest['RSI']:.1f}")
    col3.metric("MACD",         f"{latest['MACD']:.4f}")
    col4.metric("Volatility",   f"{latest['Volatility_20']:.2f}")

    # Sentiment for this ticker
    wsb_ticker = wsb[wsb["ticker"] == ticker]
    if len(wsb_ticker) > 0:
        avg_sent = wsb_ticker["sentiment_score"].mean()
        col5.metric("Avg WSB Sentiment", f"{avg_sent:+.3f}")
    else:
        col5.metric("WSB Posts", "No data")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Price chart ────────────────────────────────────────
    st.subheader(f"{ticker} — Price & Technical Indicators")
    fig = stock_chart(stock_df, ticker)
    st.plotly_chart(fig, use_container_width=True)

    # ── Sentiment chart ────────────────────────────────────
    if show_sent and ticker in SENTIMENT_TICKERS:
        st.subheader(f"{ticker} — WSB Sentiment")
        if len(wsb_ticker) > 0:
            fig_sent = sentiment_chart(wsb, ticker)
            st.plotly_chart(fig_sent, use_container_width=True)
        else:
            st.info("No WSB sentiment data for this ticker.")
    elif show_sent:
        st.info(f"Sentiment data available for: "
                f"{', '.join(SENTIMENT_TICKERS)}")

    # ── Model predictions ──────────────────────────────────
    st.subheader(f"{ticker} — Model Predictions")

    xgb_t = xgb[xgb["Ticker"] == ticker].copy()
    lst_t = lst[lst["Ticker"] == ticker].copy()

    if len(xgb_t) > 0 or len(lst_t) > 0:
        # Merge both predictions
        if len(xgb_t) > 0 and len(lst_t) > 0:
            pred_merged = pd.merge(
                xgb_t[["Date","Ticker","Close",
                        "Target","xgb_prob","xgb_pred"]],
                lst_t[["Date","Ticker",
                        "lstm_prob","lstm_pred"]],
                on=["Date","Ticker"], how="outer"
            )
        elif len(xgb_t) > 0:
            pred_merged = xgb_t.rename(
                columns={"xgb_prob": "xgb_prob"})
        else:
            pred_merged = lst_t

        fig_pred = prediction_chart(pred_merged, ticker)
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("No predictions available for this ticker.")

    # ── Raw data table ─────────────────────────────────────
    with st.expander("View Raw Data", expanded=False):
        display_cols = ["Date", "Open", "High", "Low",
                        "Close", "Volume", "RSI", "MACD",
                        "BB_upper", "BB_lower",
                        "combined_sentiment", "Target"]
        available = [c for c in display_cols
                     if c in stock_df.columns]
        df_show = stock_df[available].tail(20).reset_index(drop=True)
        # Round numeric columns for cleaner display
        num_cols = df_show.select_dtypes(include="number").columns
        df_show[num_cols] = df_show[num_cols].round(4)
        dark_table(df_show)