import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ── Settings ──────────────────────────────────────────────
PROCESSED_FOLDER = "../data/processed/"
OUTPUTS_FOLDER   = "../outputs/"
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)

TRANSACTION_COST = 0.001   # 0.1% per trade
TOP_N            = 5       # pick top 5 stocks each week
INITIAL_CAPITAL  = 100000  # start with $100,000
# ──────────────────────────────────────────────────────────

# ── Load predictions ──────────────────────────────────────
print("Loading predictions...")
xgb_df  = pd.read_csv(os.path.join(PROCESSED_FOLDER,
                                    "xgb_predictions.csv"))
lstm_df = pd.read_csv(os.path.join(PROCESSED_FOLDER,
                                    "lstm_predictions.csv"))

xgb_df["Date"]  = pd.to_datetime(xgb_df["Date"])
lstm_df["Date"] = pd.to_datetime(lstm_df["Date"])

print(f"XGBoost predictions : {len(xgb_df):,} rows, "
      f"{xgb_df['Ticker'].nunique()} tickers")
print(f"LSTM predictions    : {len(lstm_df):,} rows, "
      f"{lstm_df['Ticker'].nunique()} tickers")

# ── Create ensemble by merging both predictions ───────────
print("\nCreating ensemble predictions...")
ensemble_df = pd.merge(
    xgb_df[["Date", "Ticker", "Close", "Target", "xgb_prob"]],
    lstm_df[["Date", "Ticker", "lstm_prob"]],
    on=["Date", "Ticker"],
    how="inner"
)
ensemble_df["ensemble_prob"] = (
    ensemble_df["xgb_prob"] + ensemble_df["lstm_prob"]
) / 2
print(f"Ensemble predictions: {len(ensemble_df):,} rows, "
      f"{ensemble_df['Ticker'].nunique()} tickers")

# ── Load SPY for benchmark ────────────────────────────────
print("\nLoading SPY benchmark...")
spy_df = lstm_df[lstm_df["Ticker"] == "SPY"][
    ["Date", "Close"]].copy()
spy_df = spy_df.rename(columns={"Close": "SPY_Close"})
spy_df = spy_df.sort_values("Date").reset_index(drop=True)

# ── Helper: calculate performance metrics ─────────────────
def calculate_metrics(returns, name="Strategy"):
    total_return  = (1 + returns).prod() - 1
    n_days        = len(returns)
    annual_return = (1 + total_return) ** (252 / n_days) - 1
    annual_vol    = returns.std() * np.sqrt(252)
    sharpe        = annual_return / annual_vol if annual_vol > 0 else 0

    downside      = returns[returns < 0].std() * np.sqrt(252)
    sortino       = annual_return / downside if downside > 0 else 0

    cumulative    = (1 + returns).cumprod()
    rolling_max   = cumulative.cummax()
    max_drawdown  = ((cumulative - rolling_max) / rolling_max).min()
    win_rate      = (returns > 0).mean()

    return {
        "Strategy"      : name,
        "Total Return"  : f"{total_return*100:.2f}%",
        "Annual Return" : f"{annual_return*100:.2f}%",
        "Annual Vol"    : f"{annual_vol*100:.2f}%",
        "Sharpe Ratio"  : f"{sharpe:.3f}",
        "Sortino Ratio" : f"{sortino:.3f}",
        "Max Drawdown"  : f"{max_drawdown*100:.2f}%",
        "Win Rate"      : f"{win_rate*100:.2f}%",
    }

# ── Core backtest function ────────────────────────────────
def run_backtest(df, prob_col, name):
    """
    Runs weekly rebalancing backtest on given predictions
    df       : dataframe with Date, Ticker, Close, prob_col
    prob_col : column name for buy probability
    name     : strategy name for display
    """
    print(f"\n── Running {name} ───────────────────────────")

    df          = df.copy()
    df["Week"]  = df["Date"].dt.to_period("W")
    weeks       = sorted(df["Week"].unique())

    portfolio_returns = []
    portfolio_dates   = []
    weekly_holdings   = []

    for week in weeks:
        week_data = df[df["Week"] == week].copy()
        if len(week_data) == 0:
            continue

        # Pick top N stocks by probability
        top_stocks = (week_data
                      .sort_values(prob_col, ascending=False)
                      .drop_duplicates("Ticker")
                      .head(TOP_N))

        if len(top_stocks) == 0:
            continue

        # Calculate actual price returns for selected stocks
        actual_returns = []
        for _, row in top_stocks.iterrows():
            ticker      = row["Ticker"]
            week_start  = week_data["Date"].min()
            week_end    = week_data["Date"].max()

            ticker_data = df[
                (df["Ticker"] == ticker) &
                (df["Date"] >= week_start) &
                (df["Date"] <= week_end)
            ].sort_values("Date")

            if len(ticker_data) >= 2:
                start_price = ticker_data.iloc[0]["Close"]
                end_price   = ticker_data.iloc[-1]["Close"]
                ret         = (end_price - start_price) / start_price
                actual_returns.append(ret)

        if actual_returns:
            avg_return = np.mean(actual_returns) - TRANSACTION_COST
            portfolio_returns.append(avg_return)
            portfolio_dates.append(week_data["Date"].min())
            weekly_holdings.append(top_stocks["Ticker"].tolist())

    # Build portfolio value series
    returns_series  = pd.Series(portfolio_returns,
                                index=portfolio_dates)
    portfolio_value = INITIAL_CAPITAL * (1 + returns_series).cumprod()

    print(f"Weeks traded       : {len(portfolio_returns)}")
    print(f"Starting capital   : ${INITIAL_CAPITAL:,}")
    print(f"Final value        : ${portfolio_value.iloc[-1]:,.2f}")
    print(f"Top stocks example : {weekly_holdings[0] if weekly_holdings else []}")

    return returns_series, portfolio_value, weekly_holdings

# ── Run all 3 strategies ──────────────────────────────────
xgb_returns,  xgb_value,  xgb_holdings  = run_backtest(
    xgb_df, "xgb_prob", "XGBoost Strategy")

lstm_returns, lstm_value, lstm_holdings = run_backtest(
    lstm_df, "lstm_prob", "LSTM Strategy")

ens_returns,  ens_value,  ens_holdings  = run_backtest(
    ensemble_df, "ensemble_prob", "Ensemble Strategy")

# ── SPY benchmark ─────────────────────────────────────────
spy_weekly = (spy_df
              .set_index("Date")["SPY_Close"]
              .resample("W").last()
              .pct_change()
              .dropna())
spy_value  = INITIAL_CAPITAL * (1 + spy_weekly).cumprod()

# ── Calculate metrics for all strategies ──────────────────
xgb_metrics  = calculate_metrics(xgb_returns,  "XGBoost")
lstm_metrics  = calculate_metrics(lstm_returns, "LSTM")
ens_metrics   = calculate_metrics(ens_returns,  "Ensemble")
spy_metrics   = calculate_metrics(spy_weekly,   "SPY Buy&Hold")

# ── Print comparison table ────────────────────────────────
print("\n")
print("="*75)
print("BACKTESTING RESULTS — FULL COMPARISON")
print("="*75)

metrics_keys = ["Total Return", "Annual Return", "Annual Vol",
                "Sharpe Ratio", "Sortino Ratio",
                "Max Drawdown", "Win Rate"]

print(f"\n{'Metric':<18} {'XGBoost':<16} {'LSTM':<16} "
      f"{'Ensemble':<16} {'SPY'}")
print("-"*75)
for key in metrics_keys:
    print(f"{key:<18} {xgb_metrics[key]:<16} {lstm_metrics[key]:<16} "
          f"{ens_metrics[key]:<16} {spy_metrics[key]}")

# ── Save all results ──────────────────────────────────────
print("\nSaving results...")

# Metrics comparison
results_df = pd.DataFrame([xgb_metrics, lstm_metrics,
                            ens_metrics, spy_metrics])
results_df.to_csv(os.path.join(OUTPUTS_FOLDER,
                  "backtest_results.csv"), index=False)

# Portfolio values over time
portfolio_df = pd.DataFrame({
    "Date"            : xgb_returns.index,
    "XGBoost_Value"   : xgb_value.values,
    "XGBoost_Return"  : xgb_returns.values,
})

# Add LSTM values aligned to same dates
lstm_value_aligned = lstm_value.reindex(
    xgb_returns.index, method="nearest")
portfolio_df["LSTM_Value"] = lstm_value_aligned.values

# Add Ensemble values
ens_value_aligned = ens_value.reindex(
    xgb_returns.index, method="nearest")
portfolio_df["Ensemble_Value"] = ens_value_aligned.values

# Add SPY values
spy_value_aligned = spy_value.reindex(
    xgb_returns.index, method="nearest")
portfolio_df["SPY_Value"] = spy_value_aligned.values

portfolio_df.to_csv(os.path.join(OUTPUTS_FOLDER,
                    "portfolio_values.csv"), index=False)

# Weekly holdings for each strategy
holdings_df = pd.DataFrame({
    "Week"             : [str(i+1) for i in range(len(xgb_holdings))],
    "XGBoost_Holdings" : [", ".join(h) for h in xgb_holdings],
})
holdings_df.to_csv(os.path.join(OUTPUTS_FOLDER,
                   "weekly_holdings.csv"), index=False)

print(f"\n✅ Full backtesting complete!")
print(f"   Results saved    → outputs/backtest_results.csv")
print(f"   Portfolio values → outputs/portfolio_values.csv")
print(f"   Weekly holdings  → outputs/weekly_holdings.csv")