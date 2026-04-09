"""
components/page_backtesting.py
Interactive backtesting simulator
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from utils.charts import equity_curve, BLUE, GREEN, AMBER, SLATE
from utils.table import dark_table

from pathlib import Path
BASE      = Path(__file__).resolve().parent.parent.parent
OUTPUTS   = BASE / "outputs"
PROCESSED = BASE / "data" / "processed"

DARK_BG = "#0f172a"
CARD_BG = "#1e293b"
BORDER  = "#334155"
RED     = "#ef4444"


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
def load():
    bt  = pd.read_csv(os.path.join(OUTPUTS,   "backtest_results.csv"))
    pv  = pd.read_csv(os.path.join(OUTPUTS,   "portfolio_values.csv"))
    xgb = pd.read_csv(os.path.join(PROCESSED, "xgb_predictions.csv"))
    lst = pd.read_csv(os.path.join(PROCESSED, "lstm_predictions.csv"))
    pv["Date"]  = pd.to_datetime(pv["Date"])
    xgb["Date"] = pd.to_datetime(xgb["Date"])
    lst["Date"] = pd.to_datetime(lst["Date"])
    return bt, pv, xgb, lst


def simulate(pred_df, prob_col, top_n, txn_cost, initial):
    pred_df        = pred_df.copy()
    pred_df["Week"] = pred_df["Date"].dt.to_period("W")
    weeks          = sorted(pred_df["Week"].unique())
    returns        = []
    dates          = []
    holdings       = []

    for week in weeks:
        wd = pred_df[pred_df["Week"] == week]
        if wd.empty:
            continue
        top = (wd.sort_values(prob_col, ascending=False)
                 .drop_duplicates("Ticker")
                 .head(top_n))
        if top.empty:
            continue

        actual_rets = []
        for _, row in top.iterrows():
            ticker = row["Ticker"]
            wdata  = wd[wd["Ticker"] == ticker].sort_values("Date")
            if len(wdata) >= 2:
                ret = ((wdata.iloc[-1]["Close"] -
                        wdata.iloc[0]["Close"]) /
                       wdata.iloc[0]["Close"])
                actual_rets.append(ret)

        if actual_rets:
            avg_ret = np.mean(actual_rets) - txn_cost
            returns.append(avg_ret)
            dates.append(wd["Date"].min())
            holdings.append(top["Ticker"].tolist())

    series = pd.Series(returns, index=dates)
    value  = initial * (1 + series).cumprod()
    return series, value, holdings


def render():
    st.markdown("## 💰 Backtesting Simulator")
    st.markdown("<p style='color:#94a3b8;margin-top:-12px'>"
                "Replay predictions with custom portfolio parameters</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    bt, pv, xgb, lst = load()

    tab1, tab2 = st.tabs(["📊 Pre-computed Results",
                           "⚙️ Custom Simulation"])

    # ── Tab 1 ──────────────────────────────────────────────
    with tab1:
        st.subheader("Portfolio Growth — All Strategies (2024)")
        fig = equity_curve(pv)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 Full Metrics Table")
        dark_table(bt.reset_index(drop=True))

        st.subheader("📉 Drawdown Analysis")
        fig_dd = go.Figure()

        for col, name, color in [
            ("XGBoost_Value",  "XGBoost",  BLUE),
            ("LSTM_Value",     "LSTM",      GREEN),
            ("Ensemble_Value", "Ensemble",  AMBER),
            ("SPY_Value",      "SPY",       SLATE),
        ]:
            if col not in pv.columns:
                continue
            val = pv[col]
            dd  = (val - val.cummax()) / val.cummax() * 100
            fig_dd.add_trace(go.Scatter(
                x=pv["Date"], y=dd,
                name=name, mode="lines",
                line=dict(color=color, width=1.5),
                fill="tozeroy" if name == "XGBoost" else None,
                fillcolor="rgba(59,130,246,0.05)"
                if name == "XGBoost" else None
            ))

        fig_dd.update_layout(**dark_layout(
            title="Strategy Drawdowns (%)", height=350,
            legend=dict(bgcolor="rgba(0,0,0,0)")))
        fig_dd.update_yaxes(title_text="Drawdown (%)",
                            ticksuffix="%",
                            showgrid=True, gridcolor=BORDER)
        fig_dd.update_xaxes(showgrid=True, gridcolor=BORDER)
        st.plotly_chart(fig_dd, use_container_width=True)

    # ── Tab 2 ──────────────────────────────────────────────
    with tab2:
        st.subheader("⚙️ Configure Your Simulation")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            model = st.selectbox("Model", ["XGBoost", "LSTM"],
                                 key="sim_model")
        with col2:
            top_n = st.slider("Top-N Holdings",
                              min_value=1, max_value=8,
                              value=5, key="sim_topn")
        with col3:
            txn = st.slider("Transaction Cost (%)",
                            min_value=0.0, max_value=1.0,
                            value=0.1, step=0.05,
                            key="sim_txn") / 100
        with col4:
            capital = st.number_input(
                "Initial Capital ($)",
                min_value=10_000, max_value=10_000_000,
                value=100_000, step=10_000, key="sim_cap")

        run = st.button("▶  Run Simulation", type="primary")

        if run:
            with st.spinner("Running simulation..."):
                if model == "XGBoost":
                    rets, val, holdings = simulate(
                        xgb, "xgb_prob", top_n, txn, capital)
                else:
                    rets, val, holdings = simulate(
                        lst, "lstm_prob", top_n, txn, capital)

            # Metrics
            total_ret  = (1 + rets).prod() - 1
            n          = len(rets)
            annual_ret = (1 + total_ret) ** (252/n) - 1 if n > 0 else 0
            annual_vol = rets.std() * np.sqrt(252)
            sharpe     = annual_ret / annual_vol if annual_vol > 0 else 0
            drawdown   = ((val - val.cummax()) / val.cummax()).min()
            win_rate   = (rets > 0).mean()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return",  f"{total_ret*100:.2f}%")
            col2.metric("Sharpe Ratio",  f"{sharpe:.3f}")
            col3.metric("Max Drawdown",  f"{drawdown*100:.2f}%")
            col4.metric("Win Rate",      f"{win_rate*100:.1f}%")

            # SPY benchmark
            spy_weekly = (pv.set_index("Date")["SPY_Value"]
                          .resample("W").last()
                          .pct_change().dropna())
            spy_val = capital * (1 + spy_weekly).cumprod()

            fig_sim = go.Figure()
            fig_sim.add_trace(go.Scatter(
                x=val.index, y=val.values,
                name=f"{model} Strategy",
                line=dict(color=BLUE, width=2.5)
            ))
            fig_sim.add_trace(go.Scatter(
                x=spy_val.index, y=spy_val.values,
                name="SPY Benchmark",
                line=dict(color=SLATE, width=1.5, dash="dot")
            ))
            fig_sim.add_hline(
                y=capital, line_dash="dot",
                line_color=SLATE,
                annotation_text=f"Start ${capital:,}"
            )
            fig_sim.update_layout(**dark_layout(
                title="Simulated Portfolio vs SPY",
                height=420,
                legend=dict(bgcolor="rgba(0,0,0,0)")))
            fig_sim.update_yaxes(tickprefix="$", tickformat=",",
                                 showgrid=True, gridcolor=BORDER)
            fig_sim.update_xaxes(showgrid=True, gridcolor=BORDER)
            st.plotly_chart(fig_sim, use_container_width=True)

            with st.expander("📋 View Weekly Holdings"):
                holdings_df = pd.DataFrame({
                    "Week"    : range(1, len(holdings)+1),
                    "Holdings": [", ".join(h) for h in holdings],
                })
                dark_table(holdings_df)
        else:
            st.info("Configure parameters above and click "
                    "**▶ Run Simulation** to start.")
