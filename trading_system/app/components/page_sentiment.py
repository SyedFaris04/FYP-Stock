"""
components/page_sentiment.py
Sentiment explorer — WSB, news trends + heatmap
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
PURPLE  = "#8b5cf6"
TEAL    = "#06b6d4"
SLATE   = "#64748b"


def dark_layout(**kwargs):
    base = dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor =DARK_BG,
        font         =dict(color="#e2e8f0", size=12),
        margin       =dict(l=50, r=20, t=50, b=40),
        hoverlabel   =dict(bgcolor=CARD_BG, bordercolor=BORDER),
    )
    base.update(kwargs)
    return base


@st.cache_data
def load():
    wsb  = pd.read_csv(os.path.join(PROCESSED, "wsb_sentiment.csv"))
    news = pd.read_csv(os.path.join(PROCESSED, "yahoo_news_sentiment.csv"))
    wsb["date"]  = pd.to_datetime(wsb["date"],  errors="coerce")
    news["date"] = pd.to_datetime(news["date"], errors="coerce")
    return wsb, news


def render():
    st.markdown("## 💬 Sentiment Explorer")
    st.markdown("<p style='color:#94a3b8;margin-top:-12px'>"
                "Explore WSB Reddit and news sentiment trends across stocks</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    wsb, news = load()

    # ── Overview metrics ───────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("WSB Posts",         f"{len(wsb):,}")
    col2.metric("News Articles",     f"{len(news):,}")
    col3.metric("Avg WSB Sentiment", f"{wsb['sentiment_score'].mean():+.3f}")
    pos_pct = (wsb["sentiment_score"] > 0).mean() * 100
    col4.metric("Positive Posts",    f"{pos_pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overall Trends",
        "🔥 Sentiment Heatmap",
        "🎯 Per Ticker",
        "📰 News Headlines"
    ])

    # ── Tab 1: Overall trends ──────────────────────────────
    with tab1:
        st.subheader("WSB Community Sentiment Timeline")

        daily = wsb.groupby("date")["sentiment_score"].agg(
            ["mean", "count"]).reset_index()
        daily.columns = ["date", "avg_sentiment", "post_count"]
        daily = daily.sort_values("date")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35], vertical_spacing=0.06,
            subplot_titles=["Average Daily Sentiment", "Post Volume"])
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["avg_sentiment"],
            name="Avg Sentiment", mode="lines",
            line=dict(color=TEAL, width=2),
            fill="tozeroy", fillcolor="rgba(6,182,212,0.08)"),
            row=1, col=1)
        fig.add_hline(y=0, line_color=SLATE,
                      line_dash="dash", row=1, col=1)
        fig.add_trace(go.Bar(
            x=daily["date"], y=daily["post_count"],
            name="Post Count", marker_color=PURPLE, opacity=0.7),
            row=2, col=1)
        fig.update_layout(**dark_layout(
            title="WSB Community Sentiment Over Time", height=480))
        fig.update_yaxes(title_text="Sentiment", row=1, col=1,
                         showgrid=True, gridcolor=BORDER)
        fig.update_yaxes(title_text="Posts", row=2, col=1,
                         showgrid=True, gridcolor=BORDER)
        fig.update_xaxes(showgrid=True, gridcolor=BORDER)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution")
            fig2 = go.Figure(go.Histogram(
                x=wsb["sentiment_score"], nbinsx=50,
                marker_color=TEAL, opacity=0.8))
            fig2.add_vline(x=0, line_color=SLATE, line_dash="dash")
            fig2.update_layout(**dark_layout(
                title="Distribution of Sentiment Scores", height=320))
            fig2.update_xaxes(title_text="Sentiment Score",
                               showgrid=True, gridcolor=BORDER)
            fig2.update_yaxes(title_text="Count",
                               showgrid=True, gridcolor=BORDER)
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.subheader("Posts Per Year")
            wsb["year"] = wsb["date"].dt.year
            yearly = wsb.groupby("year").agg(
                count=("sentiment_score", "count"),
                avg_sent=("sentiment_score", "mean")
            ).reset_index()

            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            fig3.add_trace(go.Bar(
                x=yearly["year"].astype(str), y=yearly["count"],
                name="Post Count", marker_color=BLUE, opacity=0.8),
                secondary_y=False)
            fig3.add_trace(go.Scatter(
                x=yearly["year"].astype(str), y=yearly["avg_sent"],
                name="Avg Sentiment", mode="lines+markers",
                line=dict(color=AMBER, width=2)),
                secondary_y=True)
            fig3.update_layout(**dark_layout(
                title="WSB Posts & Sentiment by Year", height=320,
                legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)")))
            fig3.update_yaxes(title_text="Post Count",
                               showgrid=True, gridcolor=BORDER,
                               secondary_y=False)
            fig3.update_yaxes(title_text="Avg Sentiment",
                               showgrid=False, secondary_y=True)
            fig3.update_xaxes(showgrid=False)
            st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 2: Sentiment Heatmap ───────────────────────────
    with tab2:
        st.subheader("🔥 Sentiment Heatmap — All Tickers Over Time")
        st.markdown(
            "<p style='color:#94a3b8;font-size:0.85rem;margin-top:-8px;'>"
            "Each cell shows the average sentiment score for that ticker "
            "on that date. Green = positive, Red = negative, "
            "Grey = no data.</p>",
            unsafe_allow_html=True)

        # Controls
        col1, col2 = st.columns([1, 2])
        with col1:
            period = st.selectbox(
                "Time Period",
                ["Last 30 days", "Last 90 days",
                 "Last 180 days", "All time"],
                key="heatmap_period"
            )
        with col2:
            granularity = st.selectbox(
                "Granularity",
                ["Weekly", "Monthly"],
                key="heatmap_gran"
            )

        # Filter by period
        wsb_h = wsb.copy()
        max_date = wsb_h["date"].max()

        if period == "Last 30 days":
            cutoff = max_date - pd.DateOffset(days=30)
        elif period == "Last 90 days":
            cutoff = max_date - pd.DateOffset(days=90)
        elif period == "Last 180 days":
            cutoff = max_date - pd.DateOffset(days=180)
        else:
            cutoff = wsb_h["date"].min()

        wsb_h = wsb_h[wsb_h["date"] >= cutoff]

        # Aggregate by period
        if granularity == "Weekly":
            wsb_h["period"] = wsb_h["date"].dt.to_period("W").astype(str)
        else:
            wsb_h["period"] = wsb_h["date"].dt.to_period("M").astype(str)

        pivot = wsb_h.groupby(["ticker", "period"])[
            "sentiment_score"].mean().reset_index()
        pivot = pivot.pivot(
            index="ticker",
            columns="period",
            values="sentiment_score"
        )

        if pivot.empty:
            st.info("No sentiment data for this period.")
        else:
            # Sort tickers by average sentiment
            pivot = pivot.loc[
                pivot.mean(axis=1).sort_values(ascending=False).index
            ]

            # Build heatmap
            z    = pivot.values
            x    = [str(c)[:10] for c in pivot.columns]
            y    = pivot.index.tolist()
            text = np.where(
                np.isnan(z.astype(float)),
                "",
                np.round(z.astype(float), 2).astype(str)
            )

            fig_h = go.Figure(go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale=[
                    [0.0,  "#7f1d1d"],
                    [0.25, "#ef4444"],
                    [0.45, "#374151"],
                    [0.55, "#374151"],
                    [0.75, "#10b981"],
                    [1.0,  "#064e3b"],
                ],
                zmid=0,
                zmin=-0.5,
                zmax=0.5,
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=10, color="#e2e8f0"),
                hoverongaps=False,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Period: %{x}<br>"
                    "Sentiment: %{z:.3f}<extra></extra>"
                ),
                colorbar=dict(
                    title=dict(text="Sentiment",
                               font=dict(color="#94a3b8")),
                    tickfont=dict(color="#94a3b8"),
                    bgcolor=CARD_BG,
                    bordercolor=BORDER,
                    borderwidth=1,
                    thickness=15,
                    len=0.8
                )
            ))

            height = max(300, len(y) * 45 + 100)
            fig_h.update_layout(**dark_layout(
                title=f"WSB Sentiment Heatmap — {granularity} ({period})",
                height=height,
                xaxis=dict(showgrid=False, tickangle=-45,
                           tickfont=dict(size=10)),
                yaxis=dict(showgrid=False,
                           tickfont=dict(size=11))
            ))
            st.plotly_chart(fig_h, use_container_width=True)

            # Summary below heatmap
            st.markdown("### 📊 Ticker Sentiment Summary")
            summary = pivot.mean(axis=1).reset_index()
            summary.columns = ["Ticker", "Avg Sentiment"]
            summary["Signal"] = summary["Avg Sentiment"].apply(
                lambda x: "🟢 Bullish" if x > 0.05
                     else "🔴 Bearish" if x < -0.05
                     else "⚪ Neutral")
            summary = summary.sort_values(
                "Avg Sentiment", ascending=False)

            cols = st.columns(5)
            for i, (_, row) in enumerate(summary.iterrows()):
                with cols[i % 5]:
                    val   = row["Avg Sentiment"]
                    color = ("#10b981" if val > 0.05
                             else "#ef4444" if val < -0.05
                             else "#64748b")
                    st.markdown(f"""
                    <div style='background:#1e293b;
                                border:1px solid #334155;
                                border-radius:10px;
                                padding:12px;
                                text-align:center;
                                margin-bottom:8px;'>
                        <div style='color:#e2e8f0;font-weight:700;
                                    font-size:1rem;'>{row['Ticker']}</div>
                        <div style='color:{color};font-size:1.1rem;
                                    font-weight:700;margin:4px 0;'>
                            {val:+.3f}</div>
                        <div style='color:#64748b;font-size:0.72rem;'>
                            {row['Signal']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Tab 3: Per Ticker ──────────────────────────────────
    with tab3:
        TICKERS = sorted(wsb["ticker"].dropna().unique().tolist())
        ticker  = st.selectbox("Select Ticker", TICKERS,
                               key="sent_ticker")
        t_data  = wsb[wsb["ticker"] == ticker].sort_values("date")

        if t_data.empty:
            st.warning("No data for this ticker.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Posts",
                        f"{len(t_data):,}")
            col2.metric("Avg Sentiment",
                        f"{t_data['sentiment_score'].mean():+.3f}")
            col3.metric("Most Positive",
                        f"{t_data['sentiment_score'].max():+.3f}")
            col4.metric("Most Negative",
                        f"{t_data['sentiment_score'].min():+.3f}")

            daily_t = t_data.groupby("date")[
                "sentiment_score"].mean().reset_index()
            colors = [GREEN if v >= 0 else RED
                      for v in daily_t["sentiment_score"]]

            fig4 = go.Figure(go.Bar(
                x=daily_t["date"],
                y=daily_t["sentiment_score"],
                marker_color=colors, opacity=0.85))
            fig4.add_hline(y=0, line_color=SLATE, line_dash="dash")
            fig4.update_layout(**dark_layout(
                title=f"{ticker} — Daily WSB Sentiment",
                height=350))
            fig4.update_yaxes(title_text="Sentiment Score",
                               showgrid=True, gridcolor=BORDER)
            fig4.update_xaxes(showgrid=True, gridcolor=BORDER)
            st.plotly_chart(fig4, use_container_width=True)

            st.subheader(f"Recent {ticker} Posts")
            show_cols = ["date", "title_clean",
                         "score", "sentiment_score"]
            avail = [c for c in show_cols if c in t_data.columns]
            dark_table(t_data[avail].sort_values("date", ascending=False).head(20).reset_index(drop=True))

    # ── Tab 4: News Headlines ──────────────────────────────
    with tab4:
        st.subheader("Yahoo Finance News Sentiment")
        if news.empty:
            st.warning("No news data available.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Articles", f"{len(news):,}")
            col2.metric("Avg Sentiment",
                        f"{news['sentiment_score'].mean():+.3f}")
            pos_news = (news["sentiment_score"] > 0).mean() * 100
            col3.metric("Positive Articles", f"{pos_news:.1f}%")

            news_tickers = sorted(
                news["ticker"].dropna().unique().tolist())
            sel = st.multiselect(
                "Filter by Ticker", news_tickers,
                default=news_tickers[:3])
            filtered = (news[news["ticker"].isin(sel)]
                        if sel else news)

            nb = (filtered.groupby("ticker")["sentiment_score"]
                  .mean().reset_index()
                  .sort_values("sentiment_score", ascending=False))
            colors2 = [GREEN if v >= 0 else RED
                       for v in nb["sentiment_score"]]

            fig5 = go.Figure(go.Bar(
                x=nb["ticker"], y=nb["sentiment_score"],
                marker_color=colors2, opacity=0.85,
                text=[f"{v:+.3f}" for v in nb["sentiment_score"]],
                textposition="outside"))
            fig5.add_hline(y=0, line_color=SLATE, line_dash="dash")
            fig5.update_layout(**dark_layout(
                title="Average News Sentiment by Ticker",
                height=360))
            fig5.update_yaxes(title_text="Avg Sentiment Score",
                               showgrid=True, gridcolor=BORDER)
            fig5.update_xaxes(showgrid=False)
            st.plotly_chart(fig5, use_container_width=True)

            st.subheader("Latest Headlines")
            show = ["ticker", "date", "headline",
                    "sentiment_score", "source"]
            avail = [c for c in show if c in filtered.columns]
            dark_table(filtered[avail].sort_values("date", ascending=False).head(30).reset_index(drop=True))