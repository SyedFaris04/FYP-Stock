"""
components/page_alerts.py — Smart Alerts System
─────────────────────────────────────────────────────────────────────────────
WHAT THIS PAGE DOES:
  Instead of the user having to check each stock manually, the Smart Alerts
  system scans ALL stocks on a selected date and automatically surfaces:

    1. Strong model signals (XGBoost probability > 72% or < 28%)
    2. RSI oversold / overbought conditions
    3. WSB sentiment spikes (sudden community activity)
    4. Combo alerts: multiple signals agree simultaneously (rarest & most powerful)

  This turns your system from PASSIVE (user looks things up)
  into PROACTIVE (system notifies user of opportunities).

HOW IT WORKS:
  1. User selects a date (defaults to latest available)
  2. We call copilot_engine.scan_alerts() which runs explain_prediction()
     on every ticker and filters for strong signals
  3. Results are shown as colour-coded alert cards, sorted by priority:
       Priority 0: Combo alerts (rarest signal)
       Priority 1: Strong model BUY/SELL (>72% confidence)
       Priority 2: RSI extremes
       Priority 3: Sentiment spikes
  4. User can filter by alert type
  5. Clicking "Explain" on an alert opens the full copilot panel

WHY THIS IS IMPRESSIVE:
  - Real fintech products (Bloomberg Terminal, Robinhood) have alert systems
  - Shows your ML system has PRACTICAL, real-time utility
  - Combo alerts are academically justified: multi-indicator confluence
    is a well-documented pattern in technical analysis literature
─────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from utils.copilot_engine import scan_alerts, explain_prediction, _load_xgb

BASE      = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE / "data" / "processed"

DARK_BG = "#0f172a"
CARD_BG = "#1e293b"
BORDER  = "#334155"

# All tickers in our dataset
ALL_TICKERS = ["AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","JPM","JNJ","SPY"]

# Human-readable labels for each alert type (used in filter buttons)
TYPE_LABELS = {
    "all":              "🔔 All",
    "model_buy":        "🟢 Strong BUY",
    "model_sell":       "🔴 Strong SELL",
    "rsi_oversold":     "📉 RSI Oversold",
    "rsi_overbought":   "📈 RSI Overbought",
    "sentiment_spike":  "💬 Sentiment Spike",
    "combo":            "🔥 Combo Signal",
}

# Map alert type → card accent colour
TYPE_COLORS = {
    "model_buy":       "#10b981",
    "model_sell":      "#ef4444",
    "rsi_oversold":    "#f59e0b",
    "rsi_overbought":  "#f59e0b",
    "sentiment_spike": "#8b5cf6",
    "combo":           "#06b6d4",
}

# Badge background colours (lighter versions for readability)
BADGE_COLORS = {
    "BUY":        ("#10b981", "rgba(16,185,129,0.12)"),
    "SELL":       ("#ef4444", "rgba(239,68,68,0.12)"),
    "OVERSOLD":   ("#f59e0b", "rgba(245,158,11,0.12)"),
    "OVERBOUGHT": ("#f59e0b", "rgba(245,158,11,0.12)"),
    "SENTIMENT":  ("#8b5cf6", "rgba(139,92,246,0.12)"),
    "COMBO":      ("#06b6d4", "rgba(6,182,212,0.12)"),
}


@st.cache_data(ttl=300)   # cache for 5 minutes so we don't re-scan on every click
def cached_scan_alerts(date_str: str, tickers: tuple) -> list:
    """
    Wrapper around scan_alerts() with caching.
    date_str is a string (e.g. "2024-01-05") because st.cache_data requires
    hashable arguments — pd.Timestamp is not hashable, but str is.
    """
    date = pd.Timestamp(date_str)
    return scan_alerts(date, list(tickers))


def render_alert_card(alert: dict, date: pd.Timestamp, expanded_key: str):
    """
    Render a single alert card.

    Each card shows:
      - Ticker name + badge (BUY / SELL / OVERSOLD / COMBO etc.)
      - Confidence percentage
      - Up to 3 reason bullets
      - An "Explain" button that opens the full copilot panel
    """
    ticker     = alert["ticker"]
    badge      = alert["badge"]
    label      = alert["label"]
    confidence = alert["confidence"]
    reasons    = alert["reasons"]
    combos     = alert["combo"]
    card_color = TYPE_COLORS.get(alert["type"], "#334155")

    badge_text_color, badge_bg = BADGE_COLORS.get(badge, ("#94a3b8", "#1e293b"))

    # Unique key for this card's expand button
    expand_key = f"alert_expand_{ticker}_{alert['type']}"
    is_open    = st.session_state.get(expanded_key) == expand_key

    # Card container
    st.markdown(f"""
    <div style='background:#1e293b;border:1px solid {card_color};
                border-radius:12px;padding:14px 16px;margin-bottom:10px;'>
      <div style='display:flex;align-items:flex-start;
                  justify-content:space-between;gap:10px;'>
        <div style='flex:1;'>
          <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>
            <span style='font-size:1.1rem;font-weight:800;color:#f1f5f9;'>{ticker}</span>
            <span style='background:{badge_bg};color:{badge_text_color};
                         font-size:0.65rem;font-weight:700;padding:2px 8px;
                         border-radius:4px;letter-spacing:0.06em;'>{badge}</span>
            <span style='color:#64748b;font-size:0.72rem;'>
              {confidence:.0%} confidence</span>
          </div>
          <div style='color:#94a3b8;font-size:0.8rem;font-weight:500;
                      margin-bottom:8px;'>{label}</div>
          {"".join([
              f"<div style='color:#64748b;font-size:0.75rem;margin-bottom:3px;'>• {r}</div>"
              for r in (reasons or [])[:3]
          ])}
          {"".join([
              f"<div style='color:#06b6d4;font-size:0.75rem;margin-top:4px;'>🔥 {c}</div>"
              for c in (combos or [])
          ])}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Explain button
    btn_label = "🔽 Close" if is_open else "🧠 Explain"
    if st.button(btn_label, key=f"btn_{expand_key}", use_container_width=False):
        if is_open:
            st.session_state[expanded_key] = None
        else:
            st.session_state[expanded_key] = expand_key
        st.rerun()

    # Expanded copilot panel
    if is_open:
        with st.container():
            st.markdown("""
            <div style='background:#0a1628;border:1px solid #3b82f6;
                        border-radius:10px;padding:18px;margin:6px 0 12px;'>
            """, unsafe_allow_html=True)

            # Import here to avoid circular import issues
            from components.page_predictions import render_copilot_panel
            render_copilot_panel(ticker, date)
            st.markdown("</div>", unsafe_allow_html=True)


def render():
    st.markdown("## 🚨 Smart Alerts")
    st.markdown(
        "<p style='color:#94a3b8;margin-top:-12px;'>"
        "Automated signal scanner — the system finds opportunities for you</p>",
        unsafe_allow_html=True)
    st.markdown("---")

    # ── Date selector ─────────────────────────────────────────────────────────
    xgb       = _load_xgb()
    all_dates = sorted(xgb["Date"].unique(), reverse=True)
    all_dates = [pd.Timestamp(d) for d in all_dates]

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_date = st.selectbox(
            "Scan Date",
            all_dates,
            format_func=lambda x: x.strftime("%Y-%m-%d"),
            key="alerts_date"
        )

    selected_date = pd.Timestamp(selected_date)

    # ── Alert type filter ──────────────────────────────────────────────────────
    with col2:
        filter_type = st.selectbox(
            "Filter by type",
            list(TYPE_LABELS.keys()),
            format_func=lambda k: TYPE_LABELS[k],
            key="alerts_filter"
        )

    # ── Scan all tickers ───────────────────────────────────────────────────────
    with st.spinner(f"Scanning {len(ALL_TICKERS)} stocks for signals on "
                    f"{selected_date.strftime('%Y-%m-%d')}..."):
        alerts = cached_scan_alerts(
            selected_date.strftime("%Y-%m-%d"),
            tuple(ALL_TICKERS)
        )

    # Apply filter
    if filter_type != "all":
        filtered = [a for a in alerts if a["type"] == filter_type]
    else:
        filtered = alerts

    # ── Summary metrics ────────────────────────────────────────────────────────
    all_buys    = sum(1 for a in alerts if a["type"] == "model_buy")
    all_sells   = sum(1 for a in alerts if a["type"] == "model_sell")
    all_combos  = sum(1 for a in alerts if a["type"] == "combo")
    all_rsi     = sum(1 for a in alerts if a["type"] in ("rsi_oversold","rsi_overbought"))
    all_sent    = sum(1 for a in alerts if a["type"] == "sentiment_spike")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔔 Total Alerts",      len(alerts))
    c2.metric("🟢 Strong BUY",        all_buys)
    c3.metric("🔴 Strong SELL",       all_sells)
    c4.metric("🔥 Combo Signals",     all_combos)
    c5.metric("💬 Sentiment Spikes",  all_sent)

    st.markdown("---")

    # ── Alert explanation header ───────────────────────────────────────────────
    st.markdown("""
    <div style='display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;'>
      <div style='background:rgba(6,182,212,0.08);border:1px solid #06b6d4;
                  border-radius:8px;padding:8px 14px;font-size:0.78rem;color:#06b6d4;'>
        🔥 <b>Combo</b> — multiple indicators agree simultaneously (rarest, most significant)
      </div>
      <div style='background:rgba(16,185,129,0.08);border:1px solid #10b981;
                  border-radius:8px;padding:8px 14px;font-size:0.78rem;color:#10b981;'>
        🟢 <b>Strong BUY</b> — model confidence > 72%
      </div>
      <div style='background:rgba(239,68,68,0.08);border:1px solid #ef4444;
                  border-radius:8px;padding:8px 14px;font-size:0.78rem;color:#ef4444;'>
        🔴 <b>Strong SELL</b> — model confidence < 28%
      </div>
      <div style='background:rgba(245,158,11,0.08);border:1px solid #f59e0b;
                  border-radius:8px;padding:8px 14px;font-size:0.78rem;color:#f59e0b;'>
        📉 <b>RSI</b> — oversold (&lt;30) or overbought (&gt;70)
      </div>
      <div style='background:rgba(139,92,246,0.08);border:1px solid #8b5cf6;
                  border-radius:8px;padding:8px 14px;font-size:0.78rem;color:#8b5cf6;'>
        💬 <b>Sentiment Spike</b> — unusual WSB Reddit activity
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Alert cards ────────────────────────────────────────────────────────────
    if not filtered:
        st.info(f"No alerts of type '{TYPE_LABELS[filter_type]}' found for "
                f"{selected_date.strftime('%Y-%m-%d')}. "
                f"Try a different date or filter.")
    else:
        # Track which card has its copilot panel open
        # We use a single "alerts_open" key in session_state
        if "alerts_open" not in st.session_state:
            st.session_state.alerts_open = None

        st.markdown(f"**{len(filtered)} alerts found** — sorted by priority:")
        st.markdown("<br>", unsafe_allow_html=True)

        # Render in 2-column grid for better layout
        col_a, col_b = st.columns(2)

        for idx, alert in enumerate(filtered):
            target_col = col_a if idx % 2 == 0 else col_b
            with target_col:
                render_alert_card(alert, selected_date, "alerts_open")

    st.markdown("---")

    # ── Historical alert frequency chart ──────────────────────────────────────
    st.subheader("📊 Signal Frequency Overview")
    st.markdown(
        "<p style='color:#94a3b8;font-size:0.85rem;'>How many stocks had "
        "strong signals on the selected date vs model predictions</p>",
        unsafe_allow_html=True)

    # Build a mini summary table for the selected date
    summary_data = {
        "Alert Type":  ["Strong BUY (>72%)", "Strong SELL (<28%)",
                        "RSI Oversold", "RSI Overbought", "Sentiment Spike", "Combo"],
        "Count":       [all_buys, all_sells,
                        sum(1 for a in alerts if a["type"]=="rsi_oversold"),
                        sum(1 for a in alerts if a["type"]=="rsi_overbought"),
                        all_sent, all_combos],
        "Color":       ["#10b981","#ef4444","#f59e0b","#f97316","#8b5cf6","#06b6d4"],
    }
    df_summary = pd.DataFrame(summary_data)

    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=df_summary["Alert Type"],
        y=df_summary["Count"],
        marker_color=df_summary["Color"].tolist(),
        text=df_summary["Count"],
        textposition="outside",
        opacity=0.85,
    ))
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color="#e2e8f0", size=12),
        margin=dict(l=20, r=20, t=20, b=60),
        height=300,
        showlegend=False,
        xaxis=dict(showgrid=False, tickangle=-20),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", title="Number of stocks"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── How to use section ────────────────────────────────────────────────────
    with st.expander("📖 How to use Smart Alerts"):
        st.markdown("""
        **Smart Alerts scans all 10 tracked stocks and flags notable signals.**

        **Alert priority (highest → lowest):**
        1. 🔥 **Combo** — Multiple indicators align simultaneously. E.g. RSI oversold +
           positive sentiment + bullish MACD. Rarest signal but highest conviction.
        2. 🟢/🔴 **Strong Model Signal** — XGBoost confidence above 72% (BUY) or
           below 28% (SELL). Model is highly certain.
        3. 📉📈 **RSI Extreme** — RSI below 30 (historically oversold, may bounce)
           or above 70 (overbought, may correct).
        4. 💬 **Sentiment Spike** — Unusually high WSB Reddit activity. May precede
           price movement (as shown in our research: wsb_avg_score = #1 XGBoost feature).

        **Tips:**
        - Use the **filter** to focus on one alert type at a time
        - Click **🧠 Explain** to see a full AI Copilot breakdown for any alert
        - Alerts are based on the same FinBERT + XGBoost pipeline used in backtesting
        - Always cross-reference multiple signals before making any decision
        """)
