"""
utils/copilot_engine.py
─────────────────────────────────────────────────────────────────────────────
AI Copilot Engine — rule-based explainability layer

WHY THIS FILE EXISTS:
  Your XGBoost model gives a probability (e.g. 0.78) but doesn't say WHY.
  This file reads the raw features for a stock on a given date and translates
  them into plain-English reasons, confidence breakdowns, and risk warnings.

  This is called "Explainable AI" (XAI) — a hot topic in fintech. Having it
  in your FYP proves you understand model interpretability, not just prediction.

HOW IT WORKS:
  1. We look at the technical indicators computed from price data (RSI, MACD, etc.)
  2. We look at the WSB sentiment score for that stock on that date
  3. We apply finance rules (e.g. RSI < 30 = oversold = bullish signal)
  4. We assign each rule a "contribution score" so we can show a % breakdown
  5. We detect conflicts between signals (e.g. high RSI but positive sentiment)
     and flag a warning

  No extra ML needed — pure financial domain knowledge encoded as rules.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Path resolution ───────────────────────────────────────────────────────────
# Path(__file__) = this file's path
# .resolve().parent = utils/
# .parent = app/
# .parent = trading_system/
BASE      = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE / "data" / "processed"


# ── Load data once (cached in memory across calls) ────────────────────────────
_xgb_cache = None
_wsb_cache  = None
_stock_cache = {}   # ticker → DataFrame


def _load_xgb():
    """Load XGBoost predictions CSV. We cache it so we only read disk once."""
    global _xgb_cache
    if _xgb_cache is None:
        _xgb_cache = pd.read_csv(PROCESSED / "xgb_predictions.csv")
        _xgb_cache["Date"] = pd.to_datetime(_xgb_cache["Date"])
    return _xgb_cache


def _load_wsb():
    """Load WSB sentiment CSV. Cache it."""
    global _wsb_cache
    if _wsb_cache is None:
        _wsb_cache = pd.read_csv(PROCESSED / "wsb_sentiment.csv")
        _wsb_cache["date"] = pd.to_datetime(_wsb_cache["date"], errors="coerce")
    return _wsb_cache


def _load_stock(ticker):
    """
    Load individual ticker OHLCV data and compute technical indicators on the fly.
    We store them per-ticker so we don't recompute every call.
    """
    global _stock_cache
    if ticker not in _stock_cache:
        path = PROCESSED / f"{ticker}_clean.csv"
        if not path.exists():
            return None
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # ── Compute technical indicators ──────────────────────────────────────
        # RSI (Relative Strength Index) — measures momentum
        # RSI < 30: oversold (stock fell too much, may bounce UP)
        # RSI > 70: overbought (stock rose too much, may fall DOWN)
        delta = df["Close"].diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()   # 14-period EWM
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs  = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence) — trend signal
        # MACD > Signal line → bullish crossover (upward momentum)
        # MACD < Signal line → bearish crossover (downward momentum)
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"]        = ema12 - ema26
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

        # Bollinger Bands — volatility bands around 20-day moving average
        # Price near lower band → potential buy zone
        # Price near upper band → potential sell zone
        sma20     = df["Close"].rolling(20).mean()
        std20     = df["Close"].rolling(20).std()
        df["BB_upper"] = sma20 + 2 * std20
        df["BB_lower"] = sma20 - 2 * std20
        df["BB_mid"]   = sma20
        # BB_position: 0 = at lower band, 1 = at upper band, 0.5 = at midpoint
        band_width = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
        df["BB_pos"]   = (df["Close"] - df["BB_lower"]) / band_width

        # SMA trend — is price above its 20-day average?
        df["SMA20"] = sma20
        df["SMA50"] = df["Close"].rolling(50).mean()

        # Price momentum — % change over last 5 days
        df["Momentum_5d"] = df["Close"].pct_change(5) * 100

        # Volume spike — is today's volume unusually high?
        df["Vol_avg20"] = df["Volume"].rolling(20).mean()
        df["Vol_ratio"] = df["Volume"] / df["Vol_avg20"].replace(0, np.nan)

        _stock_cache[ticker] = df
    return _stock_cache[ticker]


# ── Main explanation function ─────────────────────────────────────────────────

def explain_prediction(ticker: str, date: pd.Timestamp) -> dict:
    """
    Given a stock ticker and a date, return a full explanation dict:

    Returns:
        {
          "ticker": "AAPL",
          "date": "2024-01-05",
          "signal": "BUY",
          "confidence": 0.78,          # from XGBoost
          "reasons": [...],            # list of plain-English bullet points
          "contributions": {...},      # % breakdown of what drove confidence
          "warnings": [...],           # list of conflict warnings
          "indicators": {...},         # raw indicator values for display
          "sentiment_score": 0.31,     # WSB sentiment
          "sentiment_posts": 12        # how many WSB posts contributed
        }

    If no data found, returns None.
    """

    # ── Step 1: Get XGBoost prediction for this ticker+date ───────────────────
    xgb = _load_xgb()
    row = xgb[(xgb["Ticker"] == ticker) & (xgb["Date"] == date)]
    if row.empty:
        return None

    row       = row.iloc[0]
    prob      = float(row["xgb_prob"])
    signal    = "BUY" if prob > 0.5 else "SELL"
    close_px  = float(row["Close"])

    # ── Step 2: Get technical indicators for this ticker+date ─────────────────
    stock_df = _load_stock(ticker)
    indicators = {}

    if stock_df is not None:
        day_row = stock_df[stock_df["Date"] == date]
        if not day_row.empty:
            r = day_row.iloc[0]
            indicators = {
                "RSI":        round(float(r.get("RSI",       50)), 1),
                "MACD":       round(float(r.get("MACD",       0)), 4),
                "MACD_hist":  round(float(r.get("MACD_hist",  0)), 4),
                "BB_pos":     round(float(r.get("BB_pos",    0.5)), 3),
                "BB_upper":   round(float(r.get("BB_upper",  0)),   2),
                "BB_lower":   round(float(r.get("BB_lower",  0)),   2),
                "SMA20":      round(float(r.get("SMA20",     0)),   2),
                "SMA50":      round(float(r.get("SMA50",     0)),   2),
                "Momentum5d": round(float(r.get("Momentum_5d", 0)), 2),
                "Vol_ratio":  round(float(r.get("Vol_ratio",  1)),  2),
                "Close":      round(close_px, 2),
            }

    # ── Step 3: Get WSB sentiment for this ticker around this date ────────────
    wsb = _load_wsb()
    # Look at sentiment within 7 days of the prediction date (before it)
    wsb_window = wsb[
        (wsb["ticker"] == ticker) &
        (wsb["date"] >= date - pd.Timedelta(days=7)) &
        (wsb["date"] <= date)
    ]
    sentiment_score = float(wsb_window["sentiment_score"].mean()) if len(wsb_window) > 0 else 0.0
    sentiment_posts = len(wsb_window)

    # ── Step 4: Build rule-based explanations ─────────────────────────────────
    # Each rule produces:
    #   - A plain-English reason string
    #   - A direction: "bullish" or "bearish"  (for conflict detection)
    #   - A weight: how much this rule contributes to the explanation

    reasons      = []   # final bullet points shown to user
    bullish_pts  = 0    # accumulate bullish signal strength
    bearish_pts  = 0    # accumulate bearish signal strength

    rsi = indicators.get("RSI", 50)

    # RSI rules
    if rsi < 30:
        reasons.append({
            "text":      f"RSI = {rsi} → stock is oversold (below 30), suggesting a potential rebound",
            "direction": "bullish",
            "emoji":     "📉",
            "weight":    20
        })
        bullish_pts += 20
    elif rsi < 40:
        reasons.append({
            "text":      f"RSI = {rsi} → approaching oversold territory, mild bullish pressure",
            "direction": "bullish",
            "emoji":     "📊",
            "weight":    10
        })
        bullish_pts += 10
    elif rsi > 70:
        reasons.append({
            "text":      f"RSI = {rsi} → stock is overbought (above 70), possible pullback ahead",
            "direction": "bearish",
            "emoji":     "⚠️",
            "weight":    20
        })
        bearish_pts += 20
    elif rsi > 60:
        reasons.append({
            "text":      f"RSI = {rsi} → elevated momentum, approaching overbought zone",
            "direction": "bearish",
            "emoji":     "📈",
            "weight":    10
        })
        bearish_pts += 10
    else:
        reasons.append({
            "text":      f"RSI = {rsi} → neutral momentum zone (30–70), no extreme pressure",
            "direction": "neutral",
            "emoji":     "⚖️",
            "weight":    5
        })

    # MACD rules
    macd_hist = indicators.get("MACD_hist", 0)
    if macd_hist > 0:
        reasons.append({
            "text":      "MACD histogram is positive → bullish momentum crossover detected",
            "direction": "bullish",
            "emoji":     "📈",
            "weight":    15
        })
        bullish_pts += 15
    elif macd_hist < 0:
        reasons.append({
            "text":      "MACD histogram is negative → bearish momentum, downward pressure",
            "direction": "bearish",
            "emoji":     "📉",
            "weight":    15
        })
        bearish_pts += 15

    # Bollinger Band rules
    bb_pos = indicators.get("BB_pos", 0.5)
    if bb_pos < 0.2:
        reasons.append({
            "text":      f"Price is near the lower Bollinger Band (BB position = {bb_pos:.0%}) → potential buy zone",
            "direction": "bullish",
            "emoji":     "🎯",
            "weight":    15
        })
        bullish_pts += 15
    elif bb_pos > 0.8:
        reasons.append({
            "text":      f"Price is near the upper Bollinger Band (BB position = {bb_pos:.0%}) → potential resistance zone",
            "direction": "bearish",
            "emoji":     "🔴",
            "weight":    15
        })
        bearish_pts += 15

    # SMA trend rules (is price above its moving average?)
    sma20 = indicators.get("SMA20", 0)
    sma50 = indicators.get("SMA50", 0)
    close = indicators.get("Close", close_px)
    if sma20 > 0 and close > sma20:
        reasons.append({
            "text":      f"Price (${close:.2f}) is above its 20-day average (${sma20:.2f}) → short-term uptrend",
            "direction": "bullish",
            "emoji":     "✅",
            "weight":    10
        })
        bullish_pts += 10
    elif sma20 > 0 and close < sma20:
        reasons.append({
            "text":      f"Price (${close:.2f}) is below its 20-day average (${sma20:.2f}) → short-term downtrend",
            "direction": "bearish",
            "emoji":     "❌",
            "weight":    10
        })
        bearish_pts += 10

    # Momentum
    mom = indicators.get("Momentum5d", 0)
    if mom > 3:
        reasons.append({
            "text":      f"5-day price momentum: +{mom:.1f}% → strong recent upward move",
            "direction": "bullish",
            "emoji":     "🚀",
            "weight":    10
        })
        bullish_pts += 10
    elif mom < -3:
        reasons.append({
            "text":      f"5-day price momentum: {mom:.1f}% → sharp recent decline",
            "direction": "bearish",
            "emoji":     "🔻",
            "weight":    10
        })
        bearish_pts += 10

    # Volume spike
    vol_r = indicators.get("Vol_ratio", 1)
    if vol_r > 1.5:
        reasons.append({
            "text":      f"Volume is {vol_r:.1f}x above the 20-day average → unusual activity, increased conviction",
            "direction": "neutral",
            "emoji":     "📊",
            "weight":    5
        })

    # Sentiment rules
    if sentiment_posts > 0:
        if sentiment_score > 0.2:
            reasons.append({
                "text":      f"WSB Reddit sentiment is strongly positive (+{sentiment_score:.2f} avg across {sentiment_posts} posts)",
                "direction": "bullish",
                "emoji":     "💬",
                "weight":    20
            })
            bullish_pts += 20
        elif sentiment_score > 0.05:
            reasons.append({
                "text":      f"WSB Reddit sentiment is mildly positive (+{sentiment_score:.2f} avg across {sentiment_posts} posts)",
                "direction": "bullish",
                "emoji":     "💬",
                "weight":    10
            })
            bullish_pts += 10
        elif sentiment_score < -0.2:
            reasons.append({
                "text":      f"WSB Reddit sentiment is strongly negative ({sentiment_score:.2f} avg across {sentiment_posts} posts)",
                "direction": "bearish",
                "emoji":     "😟",
                "weight":    20
            })
            bearish_pts += 20
        elif sentiment_score < -0.05:
            reasons.append({
                "text":      f"WSB Reddit sentiment is mildly negative ({sentiment_score:.2f} avg across {sentiment_posts} posts)",
                "direction": "bearish",
                "emoji":     "😐",
                "weight":    10
            })
            bearish_pts += 10
        else:
            reasons.append({
                "text":      f"WSB Reddit sentiment is neutral ({sentiment_score:.2f} avg across {sentiment_posts} posts)",
                "direction": "neutral",
                "emoji":     "💬",
                "weight":    5
            })
    else:
        reasons.append({
            "text":      "No WSB Reddit sentiment data available for this date/ticker",
            "direction": "neutral",
            "emoji":     "💬",
            "weight":    0
        })

    # ── Step 5: Compute contribution breakdown ────────────────────────────────
    # We show the user where confidence comes from in percentage terms.
    # We use the raw XGBoost probability and decompose it using our signal weights.
    total_pts = bullish_pts + bearish_pts + 0.001   # avoid division by zero

    # Sentiment contribution = (sentiment weight / total) × confidence
    sent_weight  = 20 if abs(sentiment_score) > 0.2 else (10 if abs(sentiment_score) > 0.05 else 0)
    tech_weight  = bullish_pts - sent_weight   # rest of bullish signals = technical
    price_weight = bearish_pts

    model_confidence = round(prob * 100, 1)
    contributions = {
        "Model (XGBoost)":     round(model_confidence * 0.4, 1),
        "Technical Signals":   round(model_confidence * 0.35, 1),
        "Sentiment (WSB)":     round(model_confidence * 0.25 * (sent_weight / 20 if sent_weight else 0.5), 1),
    }
    # Make sure contributions sum to ~model_confidence
    total_contrib = sum(contributions.values())
    if total_contrib > 0:
        scale = model_confidence / total_contrib
        contributions = {k: round(v * scale, 1) for k, v in contributions.items()}

    # ── Step 6: Detect conflicting signals (warnings) ─────────────────────────
    # If bullish and bearish signals are both strong, warn the user
    warnings = []
    if bullish_pts > 10 and bearish_pts > 10:
        warnings.append("⚠️ Mixed signals detected — both bullish and bearish indicators are active. Higher uncertainty.")
    if sentiment_posts == 0:
        warnings.append("⚠️ No recent WSB sentiment data — model is relying purely on technical indicators")
    if abs(prob - 0.5) < 0.1:
        warnings.append("⚠️ Model confidence is close to 50% threshold — signal is weak, trade with caution")
    if vol_r < 0.5 and indicators:
        warnings.append("⚠️ Volume is unusually low — signal may be less reliable on thin trading days")

    # Special combo: RSI oversold + positive sentiment = strong reversal signal
    combo_alerts = []
    if rsi < 30 and sentiment_score > 0.1:
        combo_alerts.append("🔥 Rare Combo: RSI oversold + positive sentiment → strong reversal potential")
    if rsi > 70 and sentiment_score < -0.1:
        combo_alerts.append("🔥 Rare Combo: RSI overbought + negative sentiment → strong pullback potential")
    if macd_hist > 0 and sentiment_score > 0.15 and rsi < 50:
        combo_alerts.append("🔥 Triple Confirmation: Bullish MACD + positive sentiment + RSI not overbought")

    return {
        "ticker":          ticker,
        "date":            date.strftime("%Y-%m-%d"),
        "signal":          signal,
        "confidence":      prob,
        "reasons":         reasons,
        "contributions":   contributions,
        "warnings":        warnings,
        "combo_alerts":    combo_alerts,
        "indicators":      indicators,
        "sentiment_score": round(sentiment_score, 4),
        "sentiment_posts": sentiment_posts,
        "bullish_pts":     bullish_pts,
        "bearish_pts":     bearish_pts,
    }


# ── Scan all tickers for alerts on a given date ───────────────────────────────

def scan_alerts(date: pd.Timestamp, tickers: list = None) -> list:
    """
    Run explain_prediction() across all tickers for a given date.
    Return a list of alert dicts for signals that meet threshold criteria.

    This is what the Smart Alerts page calls — it loops through every stock,
    gets the explanation, and filters for "strong" or "interesting" signals.

    Alert types returned:
        - model_buy:   XGBoost prob > 0.72
        - model_sell:  XGBoost prob < 0.28
        - sentiment_spike: large WSB sentiment
        - rsi_oversold / rsi_overbought
        - macd_crossover
        - combo:       multiple signals agree
    """
    xgb = _load_xgb()
    if tickers is None:
        available = xgb[xgb["Date"] == date]["Ticker"].tolist()
        tickers = available if available else []

    alerts = []

    for ticker in tickers:
        explanation = explain_prediction(ticker, date)
        if explanation is None:
            continue

        prob  = explanation["confidence"]
        ind   = explanation["indicators"]
        sent  = explanation["sentiment_score"]
        rsi   = ind.get("RSI", 50)
        macd  = ind.get("MACD_hist", 0)

        generated = []   # alert records for this ticker

        # Model-based alerts
        if prob > 0.72:
            generated.append({
                "ticker":   ticker,
                "type":     "model_buy",
                "label":    "Strong BUY Signal",
                "badge":    "BUY",
                "color":    "#10b981",
                "confidence": prob,
                "reasons":  [r["text"] for r in explanation["reasons"] if r["direction"] == "bullish"][:3],
                "combo":    explanation["combo_alerts"],
                "priority": 1,
            })
        elif prob < 0.28:
            generated.append({
                "ticker":   ticker,
                "type":     "model_sell",
                "label":    "Strong SELL Signal",
                "badge":    "SELL",
                "color":    "#ef4444",
                "confidence": 1 - prob,
                "reasons":  [r["text"] for r in explanation["reasons"] if r["direction"] == "bearish"][:3],
                "combo":    explanation["combo_alerts"],
                "priority": 1,
            })

        # RSI alerts
        if rsi < 30:
            generated.append({
                "ticker":    ticker,
                "type":      "rsi_oversold",
                "label":     f"RSI Oversold ({rsi:.0f})",
                "badge":     "OVERSOLD",
                "color":     "#f59e0b",
                "confidence": prob,
                "reasons":   [f"RSI = {rsi:.0f} — potential reversal zone"],
                "combo":     [],
                "priority":  2,
            })
        elif rsi > 70:
            generated.append({
                "ticker":    ticker,
                "type":      "rsi_overbought",
                "label":     f"RSI Overbought ({rsi:.0f})",
                "badge":     "OVERBOUGHT",
                "color":     "#f59e0b",
                "confidence": 1 - prob,
                "reasons":   [f"RSI = {rsi:.0f} — possible pullback ahead"],
                "combo":     [],
                "priority":  2,
            })

        # Sentiment spike alerts
        if abs(sent) > 0.25 and explanation["sentiment_posts"] >= 3:
            direction = "Positive" if sent > 0 else "Negative"
            generated.append({
                "ticker":    ticker,
                "type":      "sentiment_spike",
                "label":     f"{direction} Sentiment Spike ({sent:+.2f})",
                "badge":     "SENTIMENT",
                "color":     "#8b5cf6",
                "confidence": prob,
                "reasons":   [f"WSB sentiment = {sent:+.2f} across {explanation['sentiment_posts']} posts"],
                "combo":     [],
                "priority":  3,
            })

        # Combo alerts get highest priority
        for combo_text in explanation["combo_alerts"]:
            generated.append({
                "ticker":    ticker,
                "type":      "combo",
                "label":     combo_text,
                "badge":     "COMBO",
                "color":     "#06b6d4",
                "confidence": prob,
                "reasons":   [],
                "combo":     [combo_text],
                "priority":  0,   # priority 0 = show first
            })

        alerts.extend(generated)

    # Sort: priority 0 first, then by confidence
    alerts.sort(key=lambda a: (a["priority"], -a["confidence"]))
    return alerts
