"""
components/page_portfolio.py
User portfolio tracker
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from utils.table import dark_table

from pathlib import Path
BASE      = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE / "data" / "processed"

DARK_BG = "#0f172a"
CARD_BG = "#1e293b"
BORDER  = "#334155"
BLUE    = "#3b82f6"
GREEN   = "#10b981"
RED     = "#ef4444"
AMBER   = "#f59e0b"
SLATE   = "#64748b"


def dark_layout(**kwargs):
    base = dict(
        paper_bgcolor = DARK_BG,
        plot_bgcolor  = DARK_BG,
        font          = dict(color="#e2e8f0", size=12),
        margin        = dict(l=50, r=20, t=50, b=40),
        hoverlabel    = dict(bgcolor=CARD_BG, bordercolor=BORDER),
    )
    base.update(kwargs)
    return base


@st.cache_data
def load_market():
    df  = pd.read_csv(os.path.join(PROCESSED, "final_dataset.csv"))
    xgb = pd.read_csv(os.path.join(PROCESSED, "xgb_predictions.csv"))
    lst = pd.read_csv(os.path.join(PROCESSED, "lstm_predictions.csv"))
    df["Date"]  = pd.to_datetime(df["Date"])
    xgb["Date"] = pd.to_datetime(xgb["Date"])
    lst["Date"] = pd.to_datetime(lst["Date"])
    return df, xgb, lst


def render():
    st.markdown("## 👤 My Portfolio")
    st.markdown("<p style='color:#94a3b8;margin-top:-12px'>"
                "Track your personal holdings and get model signals</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    df, xgb, lst = load_market()

    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []

    # ── Add stock form ─────────────────────────────────────
    st.subheader("➕ Add a Stock")
    ALL_TICKERS = sorted(df["Ticker"].unique().tolist())

    with st.form("add_stock_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ticker = st.selectbox("Ticker", ALL_TICKERS)
        with col2:
            shares = st.number_input("Shares", min_value=0.01,
                                     value=10.0, step=1.0)
        with col3:
            buy_price = st.number_input("Buy Price ($)",
                                        min_value=0.01,
                                        value=100.0, step=1.0)
        with col4:
            buy_date = st.date_input("Buy Date")

        submitted = st.form_submit_button("Add to Portfolio",
                                          type="primary")
        if submitted:
            st.session_state.portfolio.append({
                "Ticker"    : ticker,
                "Shares"    : shares,
                "Buy Price" : buy_price,
                "Buy Date"  : str(buy_date),
            })
            st.success(f"Added {shares} shares of "
                       f"{ticker} at ${buy_price:.2f}")

    st.markdown("---")

    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Add stocks above!")
        st.markdown("""
        ### 💡 What you can do here
        - See your current P&L (profit/loss)
        - Get buy/sell signals from our models
        - View portfolio allocation chart
        - Compare against SPY benchmark
        """)
        return

    # ── Build portfolio dataframe ──────────────────────────
    portfolio_df = pd.DataFrame(st.session_state.portfolio)

    latest_prices = (df.sort_values("Date")
                     .groupby("Ticker")["Close"]
                     .last().reset_index()
                     .rename(columns={"Close": "Current Price"}))

    portfolio_df = portfolio_df.merge(latest_prices,
                                      on="Ticker", how="left")
    portfolio_df["Current Value"] = (portfolio_df["Shares"] *
                                     portfolio_df["Current Price"])
    portfolio_df["Cost Basis"]    = (portfolio_df["Shares"] *
                                     portfolio_df["Buy Price"])
    portfolio_df["PnL_dollars"]   = (portfolio_df["Current Value"] -
                                     portfolio_df["Cost Basis"])
    portfolio_df["PnL_pct"]       = ((portfolio_df["PnL_dollars"] /
                                      portfolio_df["Cost Basis"]) * 100)

    # ── Summary metrics ────────────────────────────────────
    total_value   = portfolio_df["Current Value"].sum()
    total_cost    = portfolio_df["Cost Basis"].sum()
    total_pnl     = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Value",  f"${total_value:,.2f}")
    col2.metric("Total Cost",   f"${total_cost:,.2f}")
    col3.metric("Total P&L",    f"${total_pnl:+,.2f}",
                                f"{total_pnl_pct:+.2f}%")
    col4.metric("Holdings",     f"{len(portfolio_df)}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Get model signals ──────────────────────────────────
    latest_date = xgb["Date"].max()
    latest_xgb  = xgb[xgb["Date"] == latest_date][
        ["Ticker","xgb_prob"]]
    portfolio_df = portfolio_df.merge(latest_xgb,
                                      on="Ticker", how="left")
    portfolio_df["Signal"] = portfolio_df["xgb_prob"].apply(
        lambda x: "🟢 BUY"  if pd.notna(x) and x > 0.5
             else "🔴 SELL" if pd.notna(x) and x <= 0.5
             else "⚪ N/A")

    # ── Holdings table ─────────────────────────────────────
    st.subheader("📋 Holdings")
    display = portfolio_df[[
        "Ticker","Shares","Buy Price","Current Price",
        "Current Value","PnL_dollars","PnL_pct","Signal"
    ]].copy()
    display["PnL_pct"]       = display["PnL_pct"].apply(
        lambda x: f"{x:+.2f}%")
    display["PnL_dollars"]   = display["PnL_dollars"].apply(
        lambda x: f"${x:+,.2f}")
    display["Current Value"] = display["Current Value"].apply(
        lambda x: f"${x:,.2f}")
    display["Current Price"] = display["Current Price"].apply(
        lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
    display = display.rename(columns={
        "PnL_dollars": "P&L ($)",
        "PnL_pct":     "P&L (%)"
    })
    dark_table(display.reset_index(drop=True))

    # ── Charts ─────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🥧 Portfolio Allocation")
        pie_colors = [BLUE, GREEN, AMBER, "#8b5cf6",
                      "#06b6d4", "#f43f5e", "#84cc16",
                      "#0ea5e9"]
        fig_pie = go.Figure(go.Pie(
            labels  = portfolio_df["Ticker"],
            values  = portfolio_df["Current Value"],
            hole    = 0.4,
            marker  = dict(colors=pie_colors[:len(portfolio_df)]),
            textinfo= "label+percent"
        ))
        fig_pie.update_layout(**dark_layout(
            height=320, showlegend=False))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("📊 P&L by Stock")
        pnl_vals = portfolio_df["PnL_dollars"].values
        colors   = [GREEN if v >= 0 else RED for v in pnl_vals]
        fig_pnl  = go.Figure(go.Bar(
            x=portfolio_df["Ticker"],
            y=pnl_vals,
            marker_color=colors,
            opacity=0.85,
            text=[f"${v:+,.0f}" for v in pnl_vals],
            textposition="outside"
        ))
        fig_pnl.add_hline(y=0, line_color=SLATE,
                          line_dash="dash")
        fig_pnl.update_layout(**dark_layout(
            title="Unrealised P&L ($)", height=320))
        fig_pnl.update_yaxes(title_text="P&L ($)",
                              tickprefix="$",
                              showgrid=True, gridcolor=BORDER)
        fig_pnl.update_xaxes(showgrid=False)
        st.plotly_chart(fig_pnl, use_container_width=True)

    # ── Model signals chart ────────────────────────────────
    st.markdown("---")
    st.subheader("🤖 Model Signals for Your Holdings")
    my_tickers = portfolio_df["Ticker"].tolist()
    xgb_my = xgb[
        (xgb["Date"] == latest_date) &
        (xgb["Ticker"].isin(my_tickers))
    ].sort_values("xgb_prob", ascending=False)

    if len(xgb_my) > 0:
        fig_sig = go.Figure(go.Bar(
            x=xgb_my["Ticker"],
            y=xgb_my["xgb_prob"],
            marker_color=[GREEN if p > 0.5 else RED
                          for p in xgb_my["xgb_prob"]],
            text=[f"{p:.1%}" for p in xgb_my["xgb_prob"]],
            textposition="outside",
            opacity=0.85
        ))
        fig_sig.add_hline(y=0.5, line_color=SLATE,
                          line_dash="dash",
                          annotation_text="BUY threshold")
        fig_sig.update_layout(**dark_layout(
            title="XGBoost Buy Probability — Your Holdings",
            height=360))
        fig_sig.update_yaxes(title_text="Buy Probability",
                              tickformat=".0%", range=[0, 1],
                              showgrid=True, gridcolor=BORDER)
        fig_sig.update_xaxes(showgrid=False)
        st.plotly_chart(fig_sig, use_container_width=True)
    else:
        st.info("No model signals available for your holdings.")

    # ── Clear button ───────────────────────────────────────
    st.markdown("---")
    if st.button("🗑️ Clear Portfolio"):
        st.session_state.portfolio = []
        st.rerun()