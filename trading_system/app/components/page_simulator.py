"""
components/page_simulator.py — Paper Trading Simulator
─────────────────────────────────────────────────────────────────────────────
WHAT THIS PAGE DOES:
  A paper trading simulator lets users practise trading with fake money using
  REAL historical data from our dataset (2015–2024). No real money involved —
  it's purely for learning and demonstrating the system.

HOW IT WORKS:
  1. User picks a start date and starting capital (e.g. $10,000)
  2. They "walk forward" day-by-day through history
  3. Each day they see: stock price, RSI/MACD indicators, sentiment score,
     AND the model's predicted signal (BUY / SELL)
  4. They click BUY or SELL for any stock, or skip to next day
  5. The system tracks their portfolio value, cash, and all trades
  6. At any point they can see their performance vs the SPY benchmark

WHY THIS IS IMPRESSIVE FOR YOUR FYP:
  - It proves your system has PRACTICAL value, not just academic accuracy
  - Users can see in real-time that sentiment + technical signals help
  - The AI Suggestion panel shows your model's recommendation before the user decides

DATA USED:
  - Individual clean CSV files (AAPL_clean.csv, MSFT_clean.csv etc.)
    for price + technical indicators (computed on-the-fly)
  - xgb_predictions.csv for the model's buy probability on each day
  - wsb_sentiment.csv for the sentiment context
─────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import date as dt_date
from utils.table import dark_table
from utils.copilot_engine import _load_xgb, _load_wsb, _load_stock

BASE      = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE / "data" / "processed"

DARK_BG = "#0f172a"
CARD_BG = "#1e293b"
BORDER  = "#334155"
GREEN   = "#10b981"
RED     = "#ef4444"
BLUE    = "#3b82f6"
AMBER   = "#f59e0b"
PURPLE  = "#8b5cf6"
SLATE   = "#64748b"

# Tickers available in our processed dataset
TICKERS = ["AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","JPM","JNJ","SPY"]


def dark_layout(**kw):
    b = dict(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
             font=dict(color="#e2e8f0", size=12),
             margin=dict(l=45, r=20, t=40, b=40),
             hoverlabel=dict(bgcolor=CARD_BG, bordercolor=BORDER))
    b.update(kw)
    return b


# ── Helper: get price for a ticker on a specific date ─────────────────────────
def get_price(ticker: str, date: pd.Timestamp) -> float | None:
    """
    Look up the closing price for a ticker on a given date.
    Returns None if the date is not a trading day for that ticker.
    """
    df = _load_stock(ticker)
    if df is None:
        return None
    row = df[df["Date"] == date]
    return float(row["Close"].iloc[0]) if not row.empty else None


# ── Helper: get SPY price for a date (used to compute benchmark return) ───────
def get_spy_at(date: pd.Timestamp) -> float | None:
    return get_price("SPY", date)


# ── Helper: get all trading dates in our dataset ──────────────────────────────
@st.cache_data
def get_all_trading_dates():
    """
    Returns sorted list of all dates that appear in xgb_predictions.csv.
    These are the days we have full model data for.
    """
    xgb = _load_xgb()
    dates = sorted(xgb["Date"].unique())
    return [pd.Timestamp(d) for d in dates]


# ── Helper: get model signal for a ticker on a date ──────────────────────────
def get_model_signal(ticker: str, date: pd.Timestamp) -> dict:
    """
    Returns the XGBoost prediction for a specific ticker and date.
    Format: {"signal": "BUY"/"SELL", "probability": 0.73}
    """
    xgb = _load_xgb()
    row = xgb[(xgb["Ticker"] == ticker) & (xgb["Date"] == date)]
    if row.empty:
        return {"signal": "HOLD", "probability": 0.5}
    prob   = float(row.iloc[0]["xgb_prob"])
    signal = "BUY" if prob > 0.5 else "SELL"
    return {"signal": signal, "probability": prob}


# ── Helper: get sentiment for a ticker on a date ──────────────────────────────
def get_sentiment(ticker: str, date: pd.Timestamp) -> dict:
    """
    Returns average WSB sentiment score within 7 days before the given date.
    """
    wsb = _load_wsb()
    window = wsb[
        (wsb["ticker"] == ticker) &
        (wsb["date"] >= date - pd.Timedelta(days=7)) &
        (wsb["date"] <= date)
    ]
    if len(window) == 0:
        return {"score": 0.0, "posts": 0}
    return {"score": float(window["sentiment_score"].mean()), "posts": len(window)}


# ── Helper: get indicator snapshot for a ticker on a date ─────────────────────
def get_indicators(ticker: str, date: pd.Timestamp) -> dict:
    """
    Returns technical indicator values for display on the simulator page.
    All indicators are computed in copilot_engine._load_stock().
    """
    df = _load_stock(ticker)
    if df is None:
        return {}
    row = df[df["Date"] == date]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "RSI":        round(float(r.get("RSI", 50)),    1),
        "MACD_hist":  round(float(r.get("MACD_hist", 0)), 4),
        "BB_pos":     round(float(r.get("BB_pos", 0.5)), 2),
        "Momentum5d": round(float(r.get("Momentum_5d", 0)), 2),
        "Close":      round(float(r.get("Close", 0)), 2),
    }


# ── Session state initialisation ──────────────────────────────────────────────
def init_simulator(start_date: pd.Timestamp, capital: float):
    """
    Initialise (or reset) the simulator state in Streamlit session_state.

    st.session_state is Streamlit's way of preserving data between page
    re-renders (normally Streamlit re-runs the whole script on every click).

    We store:
      sim_capital   : starting cash
      sim_cash      : current cash remaining
      sim_holdings  : dict of {ticker: {shares, avg_price}}
      sim_trades    : list of trade records
      sim_history   : list of {date, total_value} for the chart
      sim_date_idx  : index into the trading dates list
      sim_dates     : full list of trading dates
      sim_spy_start : SPY price at start (for benchmark comparison)
      sim_active    : whether a simulation is running
    """
    all_dates = get_all_trading_dates()
    # Find the first trading date >= selected start date
    start_idx = next(
        (i for i, d in enumerate(all_dates) if d >= start_date),
        0
    )

    spy_start = get_spy_at(all_dates[start_idx]) or 1.0

    st.session_state.sim_capital  = capital
    st.session_state.sim_cash     = capital
    st.session_state.sim_holdings = {}   # {ticker: {"shares": N, "avg_price": P}}
    st.session_state.sim_trades   = []
    st.session_state.sim_history  = [{"date": all_dates[start_idx], "value": capital}]
    st.session_state.sim_date_idx = start_idx
    st.session_state.sim_dates    = all_dates
    st.session_state.sim_spy_start = spy_start
    st.session_state.sim_active   = True
    st.session_state.sim_msg      = None  # last trade result message


def portfolio_value(date: pd.Timestamp) -> float:
    """
    Compute total portfolio value = cash + (shares × current price) for all holdings.
    """
    total = st.session_state.sim_cash
    for ticker, pos in st.session_state.sim_holdings.items():
        price = get_price(ticker, date)
        if price:
            total += pos["shares"] * price
    return total


def execute_trade(ticker: str, action: str, shares: float, date: pd.Timestamp):
    """
    Execute a BUY or SELL trade.

    BUY:  deduct cash, add shares to holdings
    SELL: add cash, reduce/remove shares from holdings
    A 0.1% transaction cost is applied (same as in your backtesting).

    Returns (success: bool, message: str)
    """
    price = get_price(ticker, date)
    if price is None:
        return False, f"No price data for {ticker} on this date"

    txn_cost = 0.001   # 0.1% commission

    if action == "BUY":
        cost = shares * price * (1 + txn_cost)
        if cost > st.session_state.sim_cash:
            return False, f"Insufficient cash. Need ${cost:,.2f}, have ${st.session_state.sim_cash:,.2f}"
        st.session_state.sim_cash -= cost
        h = st.session_state.sim_holdings
        if ticker in h:
            # Weighted average buy price
            total_shares = h[ticker]["shares"] + shares
            total_cost   = h[ticker]["shares"] * h[ticker]["avg_price"] + shares * price
            h[ticker]    = {"shares": total_shares, "avg_price": total_cost / total_shares}
        else:
            h[ticker] = {"shares": shares, "avg_price": price}
        msg = f"✅ Bought {shares:.1f} shares of {ticker} at ${price:.2f} (cost: ${cost:,.2f})"

    elif action == "SELL":
        h = st.session_state.sim_holdings
        if ticker not in h or h[ticker]["shares"] < shares:
            avail = h.get(ticker, {}).get("shares", 0)
            return False, f"Only {avail:.1f} shares available to sell"
        proceeds = shares * price * (1 - txn_cost)
        st.session_state.sim_cash += proceeds
        h[ticker]["shares"] -= shares
        if h[ticker]["shares"] < 0.001:
            del h[ticker]
        msg = f"✅ Sold {shares:.1f} shares of {ticker} at ${price:.2f} (proceeds: ${proceeds:,.2f})"

    else:
        return False, "Unknown action"

    # Record trade
    st.session_state.sim_trades.append({
        "date":   date.strftime("%Y-%m-%d"),
        "ticker": ticker,
        "action": action,
        "price":  round(price, 2),
        "shares": round(shares, 2),
        "value":  round(shares * price, 2),
    })
    return True, msg


# ── Render main page ──────────────────────────────────────────────────────────

def render():
    st.markdown("## 🎮 Paper Trading Simulator")
    st.markdown(
        "<p style='color:#94a3b8;margin-top:-12px;'>"
        "Practise trading with real historical data — no real money involved</p>",
        unsafe_allow_html=True)
    st.markdown("---")

    # ── Setup screen (shown before simulation starts) ─────────────────────────
    if not st.session_state.get("sim_active", False):
        st.markdown("### ⚙️ Set Up Your Simulation")
        st.markdown(
            "<p style='color:#94a3b8;'>Choose a historical start date and "
            "starting capital. You'll trade day-by-day using our real dataset "
            "(2024 data) with the model's predictions as guidance.</p>",
            unsafe_allow_html=True)

        all_dates   = get_all_trading_dates()
        min_date    = all_dates[0].date()
        max_date    = all_dates[-50].date()   # leave room for forward-trading

        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=dt_date(2024, 1, 2),
                min_value=min_date,
                max_value=max_date,
                key="sim_start_date"
            )
        with col2:
            capital = st.number_input(
                "Starting Capital ($)",
                min_value=1000, max_value=1_000_000,
                value=10_000, step=1000,
                key="sim_capital_input"
            )
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Start Simulation", type="primary",
                         use_container_width=True):
                init_simulator(pd.Timestamp(start_date), float(capital))
                st.rerun()

        # Info boxes
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div style='background:#1e293b;border-radius:10px;padding:16px;
                        border-left:3px solid #3b82f6;'>
              <div style='color:#3b82f6;font-weight:700;font-size:0.85rem;
                          margin-bottom:6px;'>📊 Real Data</div>
              <div style='color:#94a3b8;font-size:0.8rem;'>
                Uses actual historical prices from Yahoo Finance (2015–2024)
                and real WSB Reddit posts scored by FinBERT
              </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div style='background:#1e293b;border-radius:10px;padding:16px;
                        border-left:3px solid #10b981;'>
              <div style='color:#10b981;font-weight:700;font-size:0.85rem;
                          margin-bottom:6px;'>🤖 AI Guidance</div>
              <div style='color:#94a3b8;font-size:0.8rem;'>
                Our XGBoost model gives you a BUY/SELL suggestion each day.
                Learn to trade with — or against — the AI
              </div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div style='background:#1e293b;border-radius:10px;padding:16px;
                        border-left:3px solid #8b5cf6;'>
              <div style='color:#8b5cf6;font-weight:700;font-size:0.85rem;
                          margin-bottom:6px;'>📈 vs Benchmark</div>
              <div style='color:#94a3b8;font-size:0.8rem;'>
                Your results are compared to a simple SPY buy-and-hold.
                Can you beat the market?
              </div>
            </div>""", unsafe_allow_html=True)
        return

    # ── Active simulation ─────────────────────────────────────────────────────
    dates     = st.session_state.sim_dates
    date_idx  = st.session_state.sim_date_idx
    today     = dates[date_idx]
    cash      = st.session_state.sim_cash
    holdings  = st.session_state.sim_holdings
    capital   = st.session_state.sim_capital
    history   = st.session_state.sim_history

    # Compute current portfolio value
    total_value = portfolio_value(today)
    pnl         = total_value - capital
    pnl_pct     = (pnl / capital) * 100

    # SPY benchmark performance
    spy_now    = get_spy_at(today) or 1.0
    spy_start  = st.session_state.sim_spy_start
    spy_ret    = ((spy_now - spy_start) / spy_start) * 100

    # ── Top KPI bar ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:#1e293b;border-radius:12px;padding:16px 20px;
                border:1px solid #334155;margin-bottom:16px;'>
      <div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px;'>
        <div style='text-align:center;'>
          <div style='color:#64748b;font-size:0.7rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.08em;'>📅 Day</div>
          <div style='color:#f1f5f9;font-size:1.1rem;font-weight:700;'>
            {today.strftime("%b %d, %Y")}</div>
          <div style='color:#64748b;font-size:0.72rem;'>
            Step {date_idx - (st.session_state.sim_dates.index(history[0]["date"]) if len(history) else 0) + 1}</div>
        </div>
        <div style='text-align:center;'>
          <div style='color:#64748b;font-size:0.7rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.08em;'>💵 Cash</div>
          <div style='color:#f1f5f9;font-size:1.1rem;font-weight:700;'>${cash:,.2f}</div>
        </div>
        <div style='text-align:center;'>
          <div style='color:#64748b;font-size:0.7rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.08em;'>💼 Invested</div>
          <div style='color:#f1f5f9;font-size:1.1rem;font-weight:700;'>${total_value - cash:,.2f}</div>
        </div>
        <div style='text-align:center;'>
          <div style='color:#64748b;font-size:0.7rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.08em;'>📊 Total Value</div>
          <div style='color:#f1f5f9;font-size:1.2rem;font-weight:800;'>${total_value:,.2f}</div>
        </div>
        <div style='text-align:center;'>
          <div style='color:#64748b;font-size:0.7rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.08em;'>📈 P&L</div>
          <div style='color:{"#10b981" if pnl >= 0 else "#ef4444"};font-size:1.1rem;font-weight:700;'>
            {pnl:+,.2f} ({pnl_pct:+.1f}%)</div>
        </div>
        <div style='text-align:center;'>
          <div style='color:#64748b;font-size:0.7rem;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.08em;'>🏦 vs SPY</div>
          <div style='color:{"#10b981" if pnl_pct >= spy_ret else "#ef4444"};
                      font-size:1.1rem;font-weight:700;'>
            SPY {spy_ret:+.1f}% &nbsp;{'✅' if pnl_pct >= spy_ret else '❌'}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Show last trade message
    if st.session_state.sim_msg:
        is_err = "❌" in st.session_state.sim_msg or "Insufficient" in st.session_state.sim_msg
        if is_err:
            st.error(st.session_state.sim_msg)
        else:
            st.success(st.session_state.sim_msg)
        st.session_state.sim_msg = None

    # ── Main content: trading panel + chart ───────────────────────────────────
    tab_trade, tab_chart, tab_history, tab_end = st.tabs(
        ["📊 Trade", "📈 Performance", "📋 Trade History", "🏁 End Simulation"])

    # ── TAB 1: TRADE ──────────────────────────────────────────────────────────
    with tab_trade:
        st.markdown(f"### Market Data for {today.strftime('%A, %B %d %Y')}")

        # Stock selector
        col_sel, col_next = st.columns([2, 1])
        with col_sel:
            selected_ticker = st.selectbox("Select Stock", TICKERS, key="sim_ticker")
        with col_next:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⏭️ Next Day →", use_container_width=True, type="primary"):
                # Advance to next trading day
                if date_idx + 1 < len(dates):
                    new_date = dates[date_idx + 1]
                    new_val  = portfolio_value(new_date)
                    st.session_state.sim_history.append({"date": new_date, "value": new_val})
                    st.session_state.sim_date_idx += 1
                    st.rerun()
                else:
                    st.warning("You've reached the end of the available data!")

        ticker = selected_ticker
        price  = get_price(ticker, today)
        ind    = get_indicators(ticker, today)
        sent   = get_sentiment(ticker, today)
        model  = get_model_signal(ticker, today)

        if price is None:
            st.warning(f"No trading data for {ticker} on {today.strftime('%Y-%m-%d')}")
        else:
            # ── AI Suggestion banner ───────────────────────────────────────────
            sig_color = "#10b981" if model["signal"] == "BUY" else "#ef4444"
            sig_emoji = "🟢" if model["signal"] == "BUY" else "🔴"

            sent_color = "#10b981" if sent["score"] > 0 else (
                         "#ef4444" if sent["score"] < 0 else "#64748b")

            st.markdown(f"""
            <div style='background:#1e293b;border:1px solid {sig_color};
                        border-radius:10px;padding:14px 18px;margin:12px 0;'>
              <div style='font-size:0.7rem;color:#64748b;font-weight:700;
                          text-transform:uppercase;letter-spacing:0.08em;
                          margin-bottom:6px;'>🤖 AI Model Suggestion</div>
              <div style='display:flex;align-items:center;justify-content:space-between;
                          flex-wrap:wrap;gap:10px;'>
                <div>
                  <span style='color:{sig_color};font-size:1.4rem;font-weight:800;'>
                    {sig_emoji} {model["signal"]}</span>
                  <span style='color:#94a3b8;font-size:0.85rem;margin-left:10px;'>
                    {model["probability"]:.1%} confidence</span>
                </div>
                <div style='color:#94a3b8;font-size:0.82rem;'>
                  Sentiment: <span style='color:{sent_color};font-weight:600;'>
                  {sent["score"]:+.3f}</span>
                  ({sent["posts"]} posts)
                </div>
                <div style='color:#64748b;font-size:0.8rem;font-style:italic;'>
                  Use as guidance only — final decision is yours
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Price + indicators row ─────────────────────────────────────────
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Close Price", f"${price:.2f}")

            rsi = ind.get("RSI", 50)
            rsi_label = "Oversold" if rsi < 30 else ("Overbought" if rsi > 70 else "Normal")
            mc2.metric("RSI", f"{rsi:.1f}", rsi_label)

            mh = ind.get("MACD_hist", 0)
            mc3.metric("MACD Histogram", f"{mh:+.4f}",
                       "Bullish" if mh > 0 else "Bearish")

            bp = ind.get("BB_pos", 0.5)
            mc4.metric("Bollinger Position", f"{bp:.0%}",
                       "Near Lower Band" if bp < 0.3 else ("Near Upper Band" if bp > 0.7 else "Mid Band"))

            mom = ind.get("Momentum5d", 0)
            mc5.metric("5-Day Momentum", f"{mom:+.1f}%")

            # ── Holdings for this ticker ───────────────────────────────────────
            holding = holdings.get(ticker)
            if holding:
                current_val = holding["shares"] * price
                pos_pnl     = current_val - holding["shares"] * holding["avg_price"]
                st.markdown(f"""
                <div style='background:rgba(59,130,246,0.08);border:1px solid #3b82f6;
                            border-radius:8px;padding:10px 14px;margin:8px 0;
                            font-size:0.82rem;color:#94a3b8;'>
                  📦 You hold <b style='color:#f1f5f9;'>{holding["shares"]:.1f} shares</b>
                  of {ticker} · avg price ${holding["avg_price"]:.2f} ·
                  current value <b style='color:#f1f5f9;'>${current_val:,.2f}</b> ·
                  P&L <span style='color:{"#10b981" if pos_pnl >= 0 else "#ef4444"};font-weight:700;'>
                  {pos_pnl:+,.2f}</span>
                </div>""", unsafe_allow_html=True)

            # ── Trade form ─────────────────────────────────────────────────────
            st.markdown("#### Execute Trade")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                shares_input = st.number_input(
                    "Number of Shares",
                    min_value=0.1, max_value=10000.0,
                    value=10.0, step=1.0,
                    key="sim_shares"
                )
            with col_b:
                est_cost = shares_input * price
                st.markdown(f"""
                <div style='background:#1e293b;border-radius:8px;padding:12px;margin-top:22px;'>
                  <div style='color:#64748b;font-size:0.72rem;font-weight:700;
                              text-transform:uppercase;'>Estimated Cost</div>
                  <div style='color:#f1f5f9;font-size:1rem;font-weight:700;'>
                    ${est_cost:,.2f}</div>
                  <div style='color:#64748b;font-size:0.7rem;'>
                    + 0.1% commission</div>
                </div>""", unsafe_allow_html=True)
            with col_c:
                st.markdown("<br>", unsafe_allow_html=True)
                bc1, bc2 = st.columns(2)
                with bc1:
                    if st.button("🟢 BUY", use_container_width=True, key="sim_buy"):
                        ok, msg = execute_trade(ticker, "BUY", shares_input, today)
                        st.session_state.sim_msg = msg
                        st.rerun()
                with bc2:
                    if st.button("🔴 SELL", use_container_width=True, key="sim_sell"):
                        ok, msg = execute_trade(ticker, "SELL", shares_input, today)
                        st.session_state.sim_msg = msg
                        st.rerun()

            # ── Current holdings summary ───────────────────────────────────────
            if holdings:
                st.markdown("#### 💼 All Holdings")
                rows = []
                for t, pos in holdings.items():
                    p = get_price(t, today) or pos["avg_price"]
                    val   = pos["shares"] * p
                    pnl_h = val - pos["shares"] * pos["avg_price"]
                    rows.append({
                        "Ticker":     t,
                        "Shares":     round(pos["shares"], 2),
                        "Avg Price":  f"${pos['avg_price']:.2f}",
                        "Cur Price":  f"${p:.2f}",
                        "Value":      f"${val:,.2f}",
                        "P&L":        f"{'+'if pnl_h>=0 else ''}{pnl_h:,.2f}",
                    })
                dark_table(pd.DataFrame(rows))

    # ── TAB 2: PERFORMANCE CHART ──────────────────────────────────────────────
    with tab_chart:
        st.markdown("### 📈 Portfolio Performance vs SPY")
        if len(history) < 2:
            st.info("Advance at least one day to see your performance chart.")
        else:
            hist_df   = pd.DataFrame(history)
            hist_df["date"] = pd.to_datetime(hist_df["date"])

            # Build SPY benchmark line
            spy_vals = []
            for d in hist_df["date"]:
                spy_px = get_spy_at(d)
                if spy_px:
                    spy_vals.append((spy_px / spy_start) * capital)
                else:
                    spy_vals.append(None)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_df["date"], y=hist_df["value"],
                name="Your Portfolio",
                line=dict(color=BLUE, width=2.5),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.06)",
            ))
            fig.add_trace(go.Scatter(
                x=hist_df["date"], y=spy_vals,
                name="SPY Benchmark",
                line=dict(color=SLATE, width=1.5, dash="dash"),
            ))
            fig.add_hline(y=capital, line_color="#334155",
                          line_dash="dot", annotation_text="Starting capital")
            fig.update_layout(**dark_layout(),
                height=400, hovermode="x unified",
                legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1))
            fig.update_yaxes(title_text="Portfolio Value ($)",
                             tickformat="$,.0f", showgrid=True, gridcolor="#1e293b")
            st.plotly_chart(fig, use_container_width=True)

            # Stats row
            days_traded = len(history) - 1
            best_day    = max(history, key=lambda x: x["value"])
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Days Traded",   days_traded)
            s2.metric("Your Return",   f"{pnl_pct:+.1f}%")
            s3.metric("SPY Return",    f"{spy_ret:+.1f}%")
            s4.metric("Alpha",         f"{pnl_pct - spy_ret:+.1f}%",
                      "vs benchmark")

    # ── TAB 3: TRADE HISTORY ──────────────────────────────────────────────────
    with tab_history:
        st.markdown("### 📋 Your Trade History")
        trades = st.session_state.sim_trades
        if not trades:
            st.info("No trades yet. Go to the Trade tab to buy or sell stocks.")
        else:
            df_trades = pd.DataFrame(trades)
            df_trades["action"] = df_trades["action"].apply(
                lambda x: "🟢 BUY" if x == "BUY" else "🔴 SELL")
            dark_table(df_trades.rename(columns={
                "date": "Date", "ticker": "Ticker",
                "action": "Action", "price": "Price ($)",
                "shares": "Shares", "value": "Total Value ($)"
            }))

    # ── TAB 4: END SIMULATION ─────────────────────────────────────────────────
    with tab_end:
        st.markdown("### 🏁 End Your Simulation")
        st.markdown("Review your final results and compare to the benchmark before ending.")

        # Final results
        days_run = len(history) - 1
        c1, c2, c3 = st.columns(3)
        c1.metric("Starting Capital",  f"${capital:,.2f}")
        c2.metric("Final Value",        f"${total_value:,.2f}")
        c3.metric("Total Return",       f"{pnl_pct:+.1f}%",
                  f"vs SPY {spy_ret:+.1f}%")

        result_color = "#10b981" if pnl_pct > spy_ret else "#ef4444"
        result_text  = "You BEAT the market! 🏆" if pnl_pct > spy_ret else "SPY beat you this time."
        st.markdown(f"""
        <div style='background:#1e293b;border-radius:10px;padding:16px 20px;
                    margin:12px 0;border-left:4px solid {result_color};'>
          <div style='color:{result_color};font-size:1rem;font-weight:700;'>{result_text}</div>
          <div style='color:#94a3b8;font-size:0.82rem;margin-top:6px;'>
            You made {len(trades)} trades over {days_run} days · 
            Final alpha: {pnl_pct - spy_ret:+.1f}% vs SPY
          </div>
        </div>""", unsafe_allow_html=True)

        col_reset, _ = st.columns([1, 3])
        with col_reset:
            if st.button("🔄 Start New Simulation", type="primary",
                         use_container_width=True):
                for k in ["sim_active","sim_cash","sim_holdings","sim_trades",
                          "sim_history","sim_date_idx","sim_dates",
                          "sim_capital","sim_spy_start","sim_msg"]:
                    st.session_state.pop(k, None)
                st.rerun()
