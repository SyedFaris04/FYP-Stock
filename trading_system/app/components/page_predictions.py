"""
components/page_predictions.py — Prediction Center + AI Trading Copilot
─────────────────────────────────────────────────────────────────────────────
WHAT CHANGED FROM THE ORIGINAL:
  1. Added "🧠 Explain Decision" button on every signal card
  2. When clicked, calls copilot_engine.explain_prediction() and shows:
       - Plain-English reasons for the model's decision
       - Feature contribution % breakdown (bar chart)
       - Risk warnings if signals conflict
       - Combo alerts if multiple signals agree strongly
  3. Raw indicator values panel: RSI, MACD, Bollinger, Momentum, Sentiment
  4. Fixed hardcoded BASE path — uses Path(__file__) so it works on any machine
─────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from utils.charts import BLUE, GREEN, RED, AMBER, TEAL, SLATE
from utils.table import dark_table
from utils.copilot_engine import explain_prediction

BASE      = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE / "data" / "processed"

DARK_BG = "#0f172a"
CARD_BG = "#1e293b"
BORDER  = "#334155"


def dark_layout(**kwargs):
    base = dict(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color="#e2e8f0", size=12),
        margin=dict(l=50, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor=CARD_BG, bordercolor=BORDER),
    )
    base.update(kwargs)
    return base


@st.cache_data
def load():
    xgb = pd.read_csv(PROCESSED / "xgb_predictions.csv")
    lst = pd.read_csv(PROCESSED / "lstm_predictions.csv")
    wsb = pd.read_csv(PROCESSED / "wsb_sentiment.csv")
    xgb["Date"] = pd.to_datetime(xgb["Date"])
    lst["Date"] = pd.to_datetime(lst["Date"])
    return xgb, lst, wsb


def render_copilot_panel(ticker: str, date: pd.Timestamp):
    """
    Renders the AI Copilot explanation panel for one stock.

    Called when user clicks the Explain Decision button.
    Fetches explanation from copilot_engine.explain_prediction() and
    displays reasons, indicators, contribution chart, and warnings.
    """
    with st.spinner(f"Analysing {ticker} signals..."):
        exp = explain_prediction(ticker, date)

    if exp is None:
        st.warning("No indicator data available for this stock on this date.")
        return

    signal     = exp["signal"]
    confidence = exp["confidence"]
    sig_color  = "#10b981" if signal == "BUY" else "#ef4444"

    # Header banner
    st.markdown(f"""
    <div style='background:#0f172a;border:1px solid {sig_color};
                border-radius:12px;padding:16px 20px;margin-bottom:14px;'>
      <div style='font-size:0.7rem;color:#64748b;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;'>
        AI Copilot — Decision Explanation
      </div>
      <div style='display:flex;align-items:center;gap:14px;'>
        <div style='font-size:1.9rem;font-weight:800;color:{sig_color};'>{signal}</div>
        <div>
          <div style='font-size:1rem;font-weight:700;color:#f1f5f9;'>{ticker}</div>
          <div style='font-size:0.82rem;color:#94a3b8;'>
            Model confidence: {confidence:.1%} &nbsp;·&nbsp; {exp["date"]}
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Combo alerts shown at top (these are special multi-signal confirmations)
    for combo in exp["combo_alerts"]:
        st.markdown(f"""
        <div style='background:rgba(6,182,212,0.08);border:1px solid #06b6d4;
                    border-radius:8px;padding:10px 14px;margin-bottom:8px;
                    color:#06b6d4;font-weight:600;font-size:0.85rem;'>
            {combo}
        </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    # Left column: rule-based reasons
    with col1:
        st.markdown("**Why the model made this call:**")
        for r in exp["reasons"]:
            direction = r["direction"]
            color = ("#10b981" if direction == "bullish"
                     else "#ef4444" if direction == "bearish"
                     else "#64748b")
            st.markdown(f"""
            <div style='display:flex;gap:10px;align-items:flex-start;
                        margin-bottom:7px;padding:8px 12px;
                        background:#1e293b;border-radius:8px;
                        border-left:3px solid {color};'>
              <span style='font-size:0.95rem;flex-shrink:0;'>{r["emoji"]}</span>
              <span style='font-size:0.8rem;color:#cbd5e1;line-height:1.5;'>{r["text"]}</span>
            </div>""", unsafe_allow_html=True)

    # Right column: indicator values + contribution chart
    with col2:
        st.markdown("**Raw indicator values:**")
        ind  = exp["indicators"]
        sent = exp.get("sentiment_score", 0)
        rows_html = ""

        def metric_row(label, value_str, color):
            return (f"<div style='display:flex;justify-content:space-between;"
                    f"padding:5px 0;border-bottom:1px solid #334155;'>"
                    f"<span style='color:#64748b;font-size:0.76rem;'>{label}</span>"
                    f"<span style='color:{color};font-weight:700;font-size:0.84rem;'>{value_str}</span>"
                    f"</div>")

        if ind:
            rsi = ind.get("RSI")
            if rsi is not None:
                c = "#10b981" if rsi < 40 else ("#ef4444" if rsi > 60 else "#f59e0b")
                rows_html += metric_row("RSI", str(rsi), c)

            mh = ind.get("MACD_hist")
            if mh is not None:
                c = "#10b981" if mh > 0 else "#ef4444"
                rows_html += metric_row("MACD Histogram", f"{mh:+.4f}", c)

            bp = ind.get("BB_pos")
            if bp is not None:
                c = "#10b981" if bp < 0.3 else ("#ef4444" if bp > 0.7 else "#94a3b8")
                rows_html += metric_row("Bollinger Position", f"{bp:.0%}", c)

            mom = ind.get("Momentum5d")
            if mom is not None:
                c = "#10b981" if mom > 0 else "#ef4444"
                rows_html += metric_row("5-Day Momentum", f"{mom:+.1f}%", c)

        sc = "#10b981" if sent > 0 else ("#ef4444" if sent < 0 else "#64748b")
        rows_html += metric_row("WSB Sentiment", f"{sent:+.3f}", sc)

        st.markdown(f"""
        <div style='background:#1e293b;border-radius:8px;padding:10px 14px;'>
          {rows_html}
        </div>""", unsafe_allow_html=True)

        # Confidence breakdown bar chart
        st.markdown("<br>**Confidence breakdown:**", unsafe_allow_html=True)
        contrib = exp["contributions"]
        fig = go.Figure(go.Bar(
            x=list(contrib.values()),
            y=list(contrib.keys()),
            orientation="h",
            marker_color=["#3b82f6", "#10b981", "#8b5cf6"],
            text=[f"{v:.1f}%" for v in contrib.values()],
            textposition="inside",
            insidetextanchor="middle",
        ))
        fig.update_layout(
            **dark_layout(),
            height=155,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Risk warnings
    if exp["warnings"]:
        st.markdown("**Risk notes:**")
        for w in exp["warnings"]:
            st.markdown(f"""
            <div style='background:rgba(245,158,11,0.08);border:1px solid #f59e0b;
                        border-radius:8px;padding:8px 14px;margin-bottom:6px;
                        color:#f59e0b;font-size:0.8rem;'>
                {w}
            </div>""", unsafe_allow_html=True)


def render():
    st.markdown("## 🎯 Prediction Center")
    st.markdown(
        "<p style='color:#94a3b8;margin-top:-12px;'>"
        "Latest model signals — click 🧠 on any card to explain the AI's decision</p>",
        unsafe_allow_html=True)
    st.markdown("---")

    xgb, lst, wsb = load()

    # Date picker
    all_dates = sorted(xgb["Date"].unique(), reverse=True)
    col1, _ = st.columns([1, 3])
    with col1:
        selected_date = st.selectbox(
            "Select Date", all_dates,
            format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"),
            key="pred_date"
        )
    selected_date = pd.Timestamp(selected_date)

    # Build merged predictions for that day
    xgb_day = xgb[xgb["Date"] == selected_date].copy()
    lst_day  = lst[lst["Date"] == selected_date].copy()

    if len(xgb_day) > 0 and len(lst_day) > 0:
        merged = pd.merge(
            xgb_day[["Ticker","Close","Target","xgb_prob","xgb_pred"]],
            lst_day[["Ticker","lstm_prob","lstm_pred"]],
            on="Ticker", how="outer"
        )
        merged["ensemble_prob"] = (merged["xgb_prob"].fillna(0.5) +
                                   merged["lstm_prob"].fillna(0.5)) / 2
    elif len(xgb_day) > 0:
        merged = xgb_day[["Ticker","Close","Target","xgb_prob","xgb_pred"]].copy()
        merged["ensemble_prob"] = merged["xgb_prob"]
    else:
        st.warning("No predictions for this date.")
        return

    merged = merged.sort_values("ensemble_prob", ascending=False).reset_index(drop=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Date",            selected_date.strftime("%Y-%m-%d"))
    c2.metric("Stocks Analysed", f"{len(merged)}")
    c3.metric("Buy Signals",     f"{(merged['ensemble_prob'] > 0.5).sum()}")
    c4.metric("Avg Confidence",  f"{merged['ensemble_prob'].mean()*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Track which copilot panel is open
    if "copilot_ticker" not in st.session_state:
        st.session_state.copilot_ticker = None

    st.subheader("📊 All Signals")
    tab1, tab2 = st.tabs(["🃏 Signal Cards", "📋 Table View"])

    with tab1:
        # Render cards in rows of 4
        for row_start in range(0, len(merged), 4):
            chunk = merged.iloc[row_start: row_start + 4]
            cols  = st.columns(4)

            for ci, (_, row) in enumerate(chunk.iterrows()):
                with cols[ci]:
                    prob     = float(row.get("ensemble_prob", 0.5))
                    ticker   = row["Ticker"]
                    signal   = "BUY" if prob > 0.5 else "SELL"
                    color    = "#10b981" if signal == "BUY" else "#ef4444"
                    pct      = int(prob * 100)
                    xgb_str  = f"{row['xgb_prob']:.2f}" if pd.notna(row.get("xgb_prob")) else "N/A"
                    lstm_str = f"{row['lstm_prob']:.2f}" if pd.notna(row.get("lstm_prob")) else "N/A"

                    st.markdown(f"""
                    <div style='background:#1e293b;border:1px solid {color};
                                border-radius:12px;padding:14px;
                                margin-bottom:6px;text-align:center;'>
                        <div style='color:#e2e8f0;font-size:1.05rem;font-weight:700;'>{ticker}</div>
                        <div style='color:{color};font-size:1.65rem;font-weight:800;margin:4px 0;'>{signal}</div>
                        <div style='color:{color};font-size:0.83rem;margin-bottom:8px;'>{prob:.1%} confidence</div>
                        <div style='background:#0f172a;border-radius:4px;height:5px;margin-bottom:8px;'>
                            <div style='background:{color};width:{pct}%;height:5px;border-radius:4px;'></div>
                        </div>
                        <div style='color:#64748b;font-size:0.66rem;'>XGB: {xgb_str} · LSTM: {lstm_str}</div>
                    </div>""", unsafe_allow_html=True)

                    # Toggle button for copilot
                    is_open   = st.session_state.copilot_ticker == ticker
                    btn_label = "🔽 Close" if is_open else "🧠 Explain"
                    if st.button(btn_label, key=f"copilot_{ticker}",
                                 use_container_width=True):
                        st.session_state.copilot_ticker = None if is_open else ticker
                        st.rerun()

            # Copilot panel renders below its row (full width)
            open_t = st.session_state.copilot_ticker
            if open_t and open_t in chunk["Ticker"].values:
                st.markdown("""
                <div style='background:#0a1628;border:1px solid #3b82f6;
                            border-radius:12px;padding:20px;margin:6px 0 14px;'>
                """, unsafe_allow_html=True)
                render_copilot_panel(open_t, selected_date)
                st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        display = merged.copy()
        display["Signal"]     = display["ensemble_prob"].apply(
            lambda x: "🟢 BUY" if x > 0.5 else "🔴 SELL")
        display["Confidence"] = display["ensemble_prob"].apply(
            lambda x: f"{x:.1%}")
        show_cols = ["Ticker","Signal","Confidence","xgb_prob","lstm_prob"]
        avail = [c for c in show_cols if c in display.columns]
        dark_table(display[avail].reset_index(drop=True))

    st.markdown("---")

    # Top 5 chart
    st.subheader("🏆 Top 5 Portfolio Picks")
    top5 = merged.head(5)
    fig  = go.Figure(go.Bar(
        x=top5["Ticker"],
        y=top5["ensemble_prob"],
        marker_color=[GREEN if p > 0.5 else RED for p in top5["ensemble_prob"]],
        text=[f"{p:.1%}" for p in top5["ensemble_prob"]],
        textposition="outside",
        opacity=0.85,
    ))
    fig.add_hline(y=0.5, line_color=SLATE, line_dash="dash",
                  annotation_text="Decision threshold (0.5)")
    fig.update_layout(**dark_layout(),
        title=f"Top 5 Picks — {selected_date.strftime('%Y-%m-%d')}", height=360)
    fig.update_yaxes(title_text="Buy Probability", tickformat=".0%",
                     range=[0,1], showgrid=True, gridcolor="#334155")
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment context
    st.subheader("💬 Sentiment Context for Top Picks")
    top5_tickers = top5["Ticker"].tolist()
    wsb_top = wsb[wsb["ticker"].isin(top5_tickers)].copy()
    wsb_top["date"] = pd.to_datetime(wsb_top["date"], errors="coerce")

    if len(wsb_top) > 0:
        for ticker in top5_tickers:
            t_sent = wsb_top[wsb_top["ticker"] == ticker]
            if len(t_sent) > 0:
                avg_s = t_sent["sentiment_score"].mean()
                with st.expander(f"{ticker} — Avg Sentiment: {avg_s:+.3f} ({len(t_sent)} posts)"):
                    avail = [c for c in ["date","title_clean","sentiment_score"] if c in t_sent.columns]
                    dark_table(t_sent[avail].sort_values("date", ascending=False).head(10).reset_index(drop=True))
    else:
        st.info("No recent WSB sentiment data for top picks on this date.")
