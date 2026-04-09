"""
utils/charts.py
Reusable Plotly chart builders for SentimentTrader.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

DARK_BG  = "#0f172a"
CARD_BG  = "#1e293b"
BORDER   = "#334155"
BLUE     = "#3b82f6"
GREEN    = "#10b981"
RED      = "#ef4444"
AMBER    = "#f59e0b"
PURPLE   = "#8b5cf6"
TEAL     = "#06b6d4"
SLATE    = "#64748b"

LAYOUT = dict(
    paper_bgcolor = DARK_BG,
    plot_bgcolor  = DARK_BG,
    font          = dict(color="#e2e8f0", size=12),
    margin        = dict(l=50, r=20, t=50, b=40),
    xaxis         = dict(showgrid=True, gridcolor=BORDER,
                         linecolor=BORDER, zeroline=False),
    yaxis         = dict(showgrid=True, gridcolor=BORDER,
                         linecolor=BORDER, zeroline=False),
    legend        = dict(bgcolor="rgba(0,0,0,0)",
                         bordercolor=BORDER),
    hoverlabel    = dict(bgcolor=CARD_BG, bordercolor=BORDER),
)


def equity_curve(portfolio_df):
    fig = go.Figure()
    strategies = [
        ("XGBoost_Value",  "XGBoost",  BLUE),
        ("LSTM_Value",     "LSTM",      GREEN),
        ("Ensemble_Value", "Ensemble",  AMBER),
        ("SPY_Value",      "SPY",       SLATE),
    ]
    for col, name, color in strategies:
        if col not in portfolio_df.columns:
            continue
        dash  = "dash"  if name == "SPY" else "solid"
        width = 1.5     if name == "SPY" else 2.5
        fig.add_trace(go.Scatter(
            x=portfolio_df["Date"], y=portfolio_df[col],
            name=name, mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"<b>{name}</b><br>%{{x}}<br>$%{{y:,.0f}}<extra></extra>"
        ))
    fig.add_hline(y=100000, line_dash="dot", line_color=SLATE,
                  annotation_text="Starting $100k",
                  annotation_font_color=SLATE)
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color="#e2e8f0", size=12),
        margin=dict(l=50, r=20, t=50, b=40),
        xaxis=dict(showgrid=True, gridcolor=BORDER, linecolor=BORDER, zeroline=False),
        hoverlabel=dict(bgcolor=CARD_BG, bordercolor=BORDER),
        title="Portfolio Growth — All Strategies vs SPY (2024)",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)")
    )
    fig.update_yaxes(tickprefix="$", tickformat=",",
                     showgrid=True, gridcolor=BORDER,
                     linecolor=BORDER, zeroline=False)
    return fig


def stock_chart(df, ticker):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.04,
        subplot_titles=[f"{ticker} — Price", "RSI", "MACD"])
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_upper"],
        line=dict(color=SLATE, width=1, dash="dot"),
        showlegend=False, name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_lower"],
        fill="tonexty", fillcolor="rgba(100,116,139,0.1)",
        line=dict(color=SLATE, width=1, dash="dot"),
        showlegend=False, name="BB Lower"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"],
        name="Price", line=dict(color=BLUE, width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"],
        name="SMA 20", line=dict(color=GREEN, width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"],
        name="RSI", line=dict(color=PURPLE, width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_color=RED,   line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_color=GREEN, line_dash="dot", row=2, col=1)
    macd_hist = df["MACD"] - df["MACD_signal"]
    colors = [GREEN if v >= 0 else RED for v in macd_hist]
    fig.add_trace(go.Bar(x=df["Date"], y=macd_hist,
        name="MACD Hist", marker_color=colors, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"],
        name="MACD", line=dict(color=BLUE, width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_signal"],
        name="Signal", line=dict(color=AMBER, width=1.5)), row=3, col=1)
    fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color="#e2e8f0", size=12),
        margin=dict(l=50, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor=CARD_BG, bordercolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)"), height=620)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1,
                     showgrid=True, gridcolor=BORDER)
    fig.update_yaxes(title_text="RSI", row=2, col=1,
                     range=[0, 100], showgrid=True, gridcolor=BORDER)
    fig.update_yaxes(title_text="MACD", row=3, col=1,
                     showgrid=True, gridcolor=BORDER)
    fig.update_xaxes(showgrid=True, gridcolor=BORDER)
    return fig


def sentiment_chart(wsb_df, ticker):
    df = wsb_df[wsb_df["ticker"] == ticker].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.groupby("date")["sentiment_score"].mean().reset_index()
    df = df.sort_values("date")
    colors = [GREEN if v >= 0 else RED for v in df["sentiment_score"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["date"], y=df["sentiment_score"],
        name="WSB Sentiment", marker_color=colors, opacity=0.8,
        hovertemplate="<b>%{x}</b><br>Sentiment: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line_color=SLATE, line_dash="dash")
    fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color="#e2e8f0", size=12),
        margin=dict(l=50, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor=CARD_BG, bordercolor=BORDER),
        title=f"{ticker} — WSB Sentiment Over Time", height=320)
    fig.update_yaxes(title_text="Sentiment Score",
                     showgrid=True, gridcolor=BORDER, zeroline=False)
    fig.update_xaxes(showgrid=True, gridcolor=BORDER)
    return fig


def prediction_chart(pred_df, ticker):
    df = pred_df[pred_df["Ticker"] == ticker].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4], vertical_spacing=0.06,
        subplot_titles=["Buy Probability", "Actual vs Predicted"])
    if "xgb_prob" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["xgb_prob"],
            name="XGBoost", line=dict(color=GREEN, width=2)), row=1, col=1)
    if "lstm_prob" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["lstm_prob"],
            name="LSTM", line=dict(color=BLUE, width=2)), row=1, col=1)
    fig.add_hline(y=0.5, line_color=SLATE, line_dash="dot", row=1, col=1)
    if "Target" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Target"],
            name="Actual", mode="markers",
            marker=dict(color=AMBER, size=4)), row=2, col=1)
    fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color="#e2e8f0", size=12),
        margin=dict(l=50, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor=CARD_BG, bordercolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)"), height=480)
    fig.update_yaxes(title_text="Probability", row=1, col=1,
                     range=[0, 1], showgrid=True, gridcolor=BORDER)
    fig.update_yaxes(title_text="Signal", row=2, col=1,
                     showgrid=True, gridcolor=BORDER)
    fig.update_xaxes(showgrid=True, gridcolor=BORDER)
    return fig


def backtest_metrics_bar(backtest_df):
    df = backtest_df.copy()
    df["return_val"] = df["Total Return"].str.replace("%","").astype(float)
    colors = []
    for _, row in df.iterrows():
        if "XGBoost"  in row["Strategy"]: colors.append(BLUE)
        elif "LSTM"   in row["Strategy"]: colors.append(GREEN)
        elif "Ensemble" in row["Strategy"]: colors.append(AMBER)
        else: colors.append(SLATE)
    fig = go.Figure(go.Bar(
        x=df["Strategy"], y=df["return_val"],
        marker_color=colors,
        text=[f"{v:.1f}%" for v in df["return_val"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>"
    ))
    fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color="#e2e8f0", size=12),
        margin=dict(l=50, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor=CARD_BG, bordercolor=BORDER),
        title="Total Return Comparison (2024)", height=360)
    fig.update_yaxes(title_text="Total Return (%)", ticksuffix="%",
                     showgrid=True, gridcolor=BORDER, zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def wsb_sentiment_timeline(wsb_df):
    df = wsb_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    daily = df.groupby("date")["sentiment_score"].agg(
        ["mean","count"]).reset_index()
    daily.columns = ["date","avg_sentiment","post_count"]
    daily = daily.sort_values("date")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.06,
        subplot_titles=["Average Daily Sentiment","Post Volume"])
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["avg_sentiment"],
        name="Avg Sentiment", mode="lines",
        line=dict(color=TEAL, width=2),
        fill="tozeroy", fillcolor="rgba(6,182,212,0.08)"), row=1, col=1)
    fig.add_hline(y=0, line_color=SLATE, line_dash="dash", row=1, col=1)
    fig.add_trace(go.Bar(x=daily["date"], y=daily["post_count"],
        name="Post Count", marker_color=PURPLE, opacity=0.7), row=2, col=1)
    fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        font=dict(color="#e2e8f0", size=12),
        margin=dict(l=50, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor=CARD_BG, bordercolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title="WSB Community Sentiment Over Time", height=480)
    fig.update_yaxes(title_text="Sentiment", row=1, col=1,
                     showgrid=True, gridcolor=BORDER)
    fig.update_yaxes(title_text="Posts", row=2, col=1,
                     showgrid=True, gridcolor=BORDER)
    fig.update_xaxes(showgrid=True, gridcolor=BORDER)
    return fig
