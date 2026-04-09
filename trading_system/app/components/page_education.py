"""
components/page_education.py
Educational section
"""
import streamlit as st

CARD_BG = "#1e293b"
BORDER  = "#334155"
BLUE    = "#3b82f6"
GREEN   = "#10b981"
RED     = "#ef4444"
AMBER   = "#f59e0b"
SLATE   = "#64748b"


def html_table(headers, rows):
    """Render a dark-themed HTML table."""
    ths = "".join(f"<th>{h}</th>" for h in headers)
    trs = ""
    for i, row in enumerate(rows):
        bg  = "#0f172a" if i % 2 == 0 else "#0a1020"
        tds = "".join(f"<td>{c}</td>" for c in row)
        trs += f"<tr style='background:{bg}'>{tds}</tr>"
    st.markdown(f"""
    <div style='border:1px solid {BORDER};border-radius:10px;
                overflow:hidden;margin:12px 0;'>
    <table style='width:100%;border-collapse:collapse;font-size:0.85rem;'>
      <thead><tr style='background:{CARD_BG}'>{ths}</tr></thead>
      <tbody>{trs}</tbody>
    </table></div>
    <style>
    th{{color:#94a3b8;font-size:0.72rem;font-weight:700;text-transform:uppercase;
        letter-spacing:0.05em;padding:12px 16px;border-bottom:1px solid {BORDER};text-align:left;}}
    td{{color:#e2e8f0;padding:11px 16px;border-bottom:1px solid #1e293b;}}
    </style>
    """, unsafe_allow_html=True)


def card(icon, color, title, desc):
    st.markdown(f"""
    <div style='background:{CARD_BG};border:1px solid {BORDER};
                border-radius:10px;padding:16px;height:100%;'>
        <div style='color:{color};font-size:1.4rem;margin-bottom:8px;'>{icon}</div>
        <div style='color:#f1f5f9;font-weight:700;margin-bottom:6px;'>{title}</div>
        <div style='color:#94a3b8;font-size:0.82rem;line-height:1.6;'>{desc}</div>
    </div>""", unsafe_allow_html=True)


def step_card(color, title, desc):
    st.markdown(f"""
    <div style='background:{CARD_BG};border-left:4px solid {color};
                border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:10px;'>
        <div style='color:{color};font-weight:700;margin-bottom:4px;'>{title}</div>
        <div style='color:#94a3b8;font-size:0.85rem;'>{desc}</div>
    </div>""", unsafe_allow_html=True)


def render():
    st.markdown("## 📚 Learn Quant Trading")
    st.markdown("<p style='color:#94a3b8;margin-top:-12px;'>"
                "Understand the concepts behind SentimentTrader</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    topic = st.selectbox("Choose a Topic", [
        "🏗️ System Architecture",
        "💬 What is Sentiment Analysis?",
        "🌲 How does XGBoost work?",
        "🧠 How does Transformer + LSTM work?",
        "📊 What is Backtesting?",
        "📈 Understanding Technical Indicators",
        "⚡ What is the Sharpe Ratio?",
        "🎯 How are Trading Signals Generated?",
    ])
    st.markdown("---")

    # ── System Architecture ────────────────────────────────
    if topic == "🏗️ System Architecture":
        st.subheader("How SentimentTrader Works")
        st.markdown("A **multi-source quantitative trading platform** that combines "
                    "financial price data with natural language sentiment to predict stock movements.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📥 Data Layer")
            for item in ["Yahoo Finance — 44 stocks, OHLCV data",
                         "Reddit WallStreetBets — 11,213 posts",
                         "Yahoo Finance News — 89 articles",
                         "Coverage: 2015–2024"]:
                st.markdown(f"- {item}")
            st.markdown("#### ⚙️ Processing Layer")
            for item in ["Data cleaning and alignment",
                         "FinBERT sentiment scoring",
                         "Technical indicator calculation",
                         "Feature engineering (34 features)"]:
                st.markdown(f"- {item}")
        with col2:
            st.markdown("#### 🤖 Model Layer")
            for item in ["XGBoost — tree-based classifier",
                         "Transformer + LSTM — deep learning",
                         "Binary classification (up/down)",
                         "Weekly rebalancing strategy"]:
                st.markdown(f"- {item}")
            st.markdown("#### 📊 Output Layer")
            for item in ["Buy/sell signals with confidence",
                         "Portfolio construction (Top 5)",
                         "Backtesting vs SPY benchmark",
                         "Interactive web dashboard"]:
                st.markdown(f"- {item}")

    # ── Sentiment Analysis ─────────────────────────────────
    elif topic == "💬 What is Sentiment Analysis?":
        st.subheader("Sentiment Analysis with FinBERT")
        st.markdown("**Sentiment analysis** uses AI to determine the emotional tone of text — "
                    "positive, negative, or neutral.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### How FinBERT Works")
            st.markdown("""FinBERT is BERT fine-tuned on **financial text** — earnings calls,
analyst reports, and news. It outputs three probability scores:""")
            html_table(
                ["Score", "Meaning"],
                [["Positive", "Bullish signal"],
                 ["Negative", "Bearish signal"],
                 ["Neutral",  "No clear direction"]])
            st.markdown("**Sentiment Score = Positive − Negative**")
        with col2:
            st.markdown("#### Examples")
            examples = [
                ("Apple smashes Q3 earnings expectations", "+0.91", GREEN, "Very Positive"),
                ("Tesla faces massive class action lawsuit","-0.83", RED,   "Very Negative"),
                ("Fed holds interest rates steady",        "+0.02", SLATE,  "Neutral"),
                ("NVDA to the moon!! 🚀🚀",               "+0.65", GREEN,  "Positive"),
                ("I lost everything on GME puts",          "-0.72", RED,    "Negative"),
            ]
            for text, score, color, label in examples:
                st.markdown(f"""
                <div style='background:{CARD_BG};border-left:3px solid {color};
                            border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:8px;'>
                    <div style='color:#e2e8f0;font-size:0.83rem;'>"{text}"</div>
                    <div style='color:{color};font-weight:700;margin-top:4px;'>
                        Score: {score} — {label}</div>
                </div>""", unsafe_allow_html=True)
        st.info("**Transfer Learning:** FinBERT was pre-trained on billions of words, "
                "then fine-tuned on financial text — professional-grade sentiment scoring "
                "without training from scratch!")

    # ── XGBoost ────────────────────────────────────────────
    elif topic == "🌲 How does XGBoost work?":
        st.subheader("XGBoost — Extreme Gradient Boosting")
        st.markdown("XGBoost builds hundreds of **decision trees** sequentially, "
                    "where each tree learns from the mistakes of the previous one.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### The Boosting Process")
            for i, s in enumerate(["Start with a simple prediction",
                "Measure where it was wrong",
                "Build a tree focused on those errors",
                "Add it to the ensemble",
                "Repeat 300 times",
                "Final prediction = all trees voting"], 1):
                st.markdown(f"**{i}.** {s}")
        with col2:
            st.markdown("#### Why it works for finance")
            for item in ["Handles non-linear patterns","Robust to outliers",
                         "Fast training time","Built-in feature importance",
                         "No distribution assumptions",
                         "Competitive with deep learning on tabular data"]:
                st.markdown(f"✅ {item}")
        st.markdown("#### Our XGBoost Configuration")
        st.code("""XGBClassifier(
    n_estimators=300,    # 300 trees
    max_depth=4,         # shallow = less overfit
    learning_rate=0.05,  # slow learning
    subsample=0.8,       # 80% of data per tree
    scale_pos_weight=0.97 # handle class imbalance
)""", language="python")

    # ── Transformer + LSTM ─────────────────────────────────
    elif topic == "🧠 How does Transformer + LSTM work?":
        st.subheader("Transformer + LSTM Hybrid Model")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Transformer Encoder")
            st.markdown("""Uses **self-attention** to find relationships between
different days in the 20-day window.

*"Which previous days are most important for predicting today?"*

- Processes all 20 days simultaneously
- Assigns attention weights to each day
- Highlights the most informative patterns""")
        with col2:
            st.markdown("#### LSTM")
            st.markdown("""Reads the sequence **in order**, remembering
important patterns over time.

*"What trend has been building over 20 days?"*

- Reads day 1 → day 2 → ... → day 20
- Maintains memory of past states
- Captures temporal dependencies""")
        st.markdown("#### Combined Architecture")
        st.code("""Input (20 days × 24 features)
→ Linear Projection  (→ 32 dimensions)
→ Layer Normalization
→ Transformer Encoder (self-attention)
→ LSTM               (hidden size 32)
→ Take last timestep
→ Dropout            (50%)
→ Fully Connected    (32 → 16 → 1)
→ Sigmoid Output     (buy probability 0–1)""")

    # ── Backtesting ────────────────────────────────────────
    elif topic == "📊 What is Backtesting?":
        st.subheader("Backtesting — Testing on Historical Data")
        st.markdown("Backtesting simulates trading using historical data to evaluate "
                    "how a strategy **would have performed** in the past.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Our Strategy")
            for s in ["Every week, run the model on all stocks",
                      "Rank stocks by buy probability",
                      "Select Top 5 highest probability",
                      "Allocate equal weight (20% each)",
                      "Apply 0.1% transaction cost per trade",
                      "Hold for one week, then rebalance",
                      "Compare final returns vs SPY"]:
                st.markdown(f"→ {s}")
        with col2:
            st.markdown("#### Key Metrics")
            html_table(
                ["Metric", "Meaning"],
                [["Total Return",  "Overall profit/loss %"],
                 ["Sharpe Ratio",  "Return per unit of risk (>1 = good)"],
                 ["Sortino Ratio", "Like Sharpe but only penalises losses"],
                 ["Max Drawdown",  "Worst peak-to-trough loss"],
                 ["Win Rate",      "% of weeks with positive return"],
                 ["Alpha",         "Return above SPY benchmark"]])

    # ── Technical Indicators ───────────────────────────────
    elif topic == "📈 Understanding Technical Indicators":
        st.subheader("Technical Indicators Used")
        indicators = {
            "SMA (Simple Moving Average)": {
                "desc"  : "Average closing price over N days. Shows overall trend direction.",
                "signal": "Price above SMA = uptrend. Price below SMA = downtrend.",
                "config": "SMA 5, 10, 20 days"},
            "EMA (Exponential Moving Average)": {
                "desc"  : "Like SMA but gives more weight to recent prices.",
                "signal": "Reacts faster than SMA to recent price changes.",
                "config": "EMA 12, 26 days"},
            "RSI (Relative Strength Index)": {
                "desc"  : "Measures if a stock is overbought or oversold. Range: 0–100.",
                "signal": "RSI > 70 = overbought (sell). RSI < 30 = oversold (buy).",
                "config": "14-day period"},
            "MACD": {
                "desc"  : "Difference between EMA 12 and EMA 26. Shows momentum changes.",
                "signal": "MACD above signal line = bullish. Below = bearish.",
                "config": "12, 26, 9 periods"},
            "Bollinger Bands": {
                "desc"  : "Price bands 2 standard deviations above and below SMA 20.",
                "signal": "Price at upper band = expensive. Lower band = cheap.",
                "config": "20-day, 2 std devs"},
            "ATR (Average True Range)": {
                "desc"  : "Measures average daily price movement.",
                "signal": "High ATR = volatile. Low ATR = stable.",
                "config": "14-day period"},
        }
        for name, info in indicators.items():
            with st.expander(name):
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"**What it is:**\n\n{info['desc']}")
                col2.markdown(f"**Signal:**\n\n{info['signal']}")
                col3.markdown(f"**Our settings:**\n\n{info['config']}")

    # ── Sharpe Ratio ───────────────────────────────────────
    elif topic == "⚡ What is the Sharpe Ratio?":
        st.subheader("The Sharpe Ratio — Risk-Adjusted Return")
        st.markdown("Measures how much **return you get per unit of risk**.")
        st.latex(r"\text{Sharpe} = \frac{R_p - R_f}{\sigma_p}")
        st.markdown("Where **Rp** = Portfolio return, **Rf** = Risk-free rate (0%), "
                    "**σp** = Portfolio volatility")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Interpretation")
            html_table(
                ["Sharpe", "Rating"],
                [["< 0",   "🔴 Losing money"],
                 ["0 – 1", "🟡 Below average"],
                 ["1 – 2", "🟢 Good"],
                 ["2 – 3", "🟢 Very good"],
                 ["> 3",   "⭐ Exceptional"]])
        with col2:
            st.markdown("#### Our Results")
            results = [("XGBoost",11.02,BLUE),("Ensemble",8.99,AMBER),
                       ("LSTM",7.66,GREEN),("SPY",6.61,SLATE)]
            for name, val, color in results:
                st.markdown(f"""
                <div style='background:{CARD_BG};border-radius:8px;
                            padding:10px 16px;margin-bottom:6px;
                            display:flex;justify-content:space-between;'>
                    <span style='color:#e2e8f0;'>{name}</span>
                    <span style='color:{color};font-weight:700;'>{val}</span>
                </div>""", unsafe_allow_html=True)
        st.success("Our XGBoost Sharpe of **11.02** is exceptional — "
                   "most hedge funds target 1–3.")

    # ── Trading Signals ────────────────────────────────────
    elif topic == "🎯 How are Trading Signals Generated?":
        st.subheader("From Raw Data to Trading Signals")
        steps = [
            (BLUE,  "1️⃣ Data Collection",      "Download OHLCV prices + news + WSB posts"),
            (AMBER, "2️⃣ Sentiment Scoring",     "FinBERT scores each text → sentiment_score"),
            (BLUE,  "3️⃣ Feature Engineering",   "Calculate RSI, MACD, BB, sentiment aggregates"),
            (GREEN, "4️⃣ Model Prediction",      "XGBoost outputs P(price up in 5 days)"),
            (AMBER, "5️⃣ Signal Ranking",        "Sort all stocks by probability"),
            (RED,   "6️⃣ Portfolio Construction","Buy Top 5 stocks, equal weight"),
        ]
        for color, title, desc in steps:
            step_card(color, title, desc)

        st.markdown("#### Signal Interpretation")
        html_table(
            ["Probability", "Signal", "Action"],
            [["> 0.65",      "Strong BUY",  "High confidence"],
             ["0.50 – 0.65", "BUY",         "Moderate confidence"],
             ["0.35 – 0.50", "SELL",        "Below threshold"],
             ["< 0.35",      "Strong SELL", "Low confidence"]])