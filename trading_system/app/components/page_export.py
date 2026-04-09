"""
components/page_export.py
Export reports to Excel and CSV
"""

import streamlit as st
import pandas as pd
import io
import os
from datetime import datetime
from utils.table import dark_table

from pathlib import Path
BASE      = Path(__file__).resolve().parent.parent.parent
OUTPUTS   = BASE / "outputs"
PROCESSED = BASE / "data" / "processed"


@st.cache_data
def load():
    bt  = pd.read_csv(os.path.join(OUTPUTS,   "backtest_results.csv"))
    pv  = pd.read_csv(os.path.join(OUTPUTS,   "portfolio_values.csv"))
    xgb = pd.read_csv(os.path.join(PROCESSED, "xgb_predictions.csv"))
    wsb = pd.read_csv(os.path.join(PROCESSED, "wsb_sentiment.csv"))
    xgb["Date"] = pd.to_datetime(xgb["Date"])
    wsb["date"] = pd.to_datetime(wsb["date"], errors="coerce")
    return bt, pv, xgb, wsb


def build_excel(bt, pv, xgb, wsb):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        bt.to_excel(writer, sheet_name="Backtest Results",   index=False)
        pv.to_excel(writer, sheet_name="Portfolio Values",   index=False)
        xgb[["Date","Ticker","Close","Target",
             "xgb_prob","xgb_pred"]].to_excel(
            writer, sheet_name="XGBoost Predictions", index=False)
        wsb[["ticker","date","title_clean",
             "score","sentiment_score"]].to_excel(
            writer, sheet_name="WSB Sentiment", index=False)
        pd.DataFrame({
            "Metric": [
                "Total Stocks","Data Range","WSB Posts",
                "XGBoost AUC","LSTM AUC",
                "XGBoost Return","SPY Return",
                "XGBoost Sharpe","XGBoost Win Rate",
                "XGBoost Max Drawdown",
            ],
            "Value": [
                "44","2015–2024",f"{len(wsb):,}",
                "0.53","0.51",
                bt[bt["Strategy"]=="XGBoost"]["Total Return"].values[0],
                bt[bt["Strategy"]=="SPY Buy&Hold"]["Total Return"].values[0],
                bt[bt["Strategy"]=="XGBoost"]["Sharpe Ratio"].values[0],
                bt[bt["Strategy"]=="XGBoost"]["Win Rate"].values[0],
                bt[bt["Strategy"]=="XGBoost"]["Max Drawdown"].values[0],
            ]
        }).to_excel(writer, sheet_name="Summary", index=False)
    buffer.seek(0)
    return buffer


def dl_button(label, data, filename, mime, key):
    """Styled download button using HTML + st.download_button."""
    st.markdown(f"""
    <style>
    div[data-testid="stDownloadButton"] > button {{
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 10px 20px !important;
        width: 100% !important;
        text-align: left !important;
    }}
    div[data-testid="stDownloadButton"] > button:hover {{
        background: #334155 !important;
        border-color: #3b82f6 !important;
        color: #ffffff !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    st.download_button(
        label=label, data=data,
        file_name=filename, mime=mime, key=key,
        use_container_width=True)


def render():
    st.markdown("## 📤 Export Reports")
    st.markdown("<p style='color:#94a3b8;margin-top:-12px'>"
                "Download your analysis as Excel or CSV</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    bt, pv, xgb, wsb = load()
    today = datetime.now().strftime("%Y%m%d")

    col1, col2 = st.columns(2)

    # ── Excel export ───────────────────────────────────────
    with col1:
        st.markdown("""
        <div style='background:#1e293b;border:1px solid #334155;
                    border-radius:12px;padding:24px;height:260px;'>
            <div style='font-size:2rem;margin-bottom:10px;'>📊</div>
            <div style='color:#f1f5f9;font-size:1.05rem;
                        font-weight:700;margin-bottom:8px;'>
                Excel Workbook</div>
            <div style='color:#94a3b8;font-size:0.82rem;
                        line-height:1.7;'>
                5 sheets: Backtest Results · Portfolio Values ·
                XGBoost Predictions · WSB Sentiment · Summary Stats
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("📊 Build Excel Report",
                     type="primary", key="build_excel"):
            with st.spinner("Building Excel workbook..."):
                buf = build_excel(bt, pv, xgb, wsb)
            dl_button(
                label    = "⬇️  Download Excel Workbook",
                data     = buf,
                filename = f"SentimentTrader_{today}.xlsx",
                mime     = ("application/vnd.openxmlformats-"
                            "officedocument.spreadsheetml.sheet"),
                key      = "save_excel"
            )

    # ── CSV exports ────────────────────────────────────────
    with col2:
        st.markdown("""
        <div style='background:#1e293b;border:1px solid #334155;
                    border-radius:12px;padding:24px;height:260px;'>
            <div style='font-size:2rem;margin-bottom:10px;'>📋</div>
            <div style='color:#f1f5f9;font-size:1.05rem;
                        font-weight:700;margin-bottom:8px;'>
                Individual CSV Files</div>
            <div style='color:#94a3b8;font-size:0.82rem;
                        line-height:1.7;'>
                Download each dataset separately.
                Perfect for importing into Python, R,
                Excel or Tableau for further analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        datasets = {
            "⬇️  Backtest Results"     : (bt, "backtest_results"),
            "⬇️  Portfolio Values"     : (pv, "portfolio_values"),
            "⬇️  XGBoost Predictions"  : (
                xgb[["Date","Ticker","Close",
                      "Target","xgb_prob","xgb_pred"]],
                "xgb_predictions"),
            "⬇️  WSB Sentiment"        : (
                wsb[["ticker","date","title_clean",
                      "score","sentiment_score"]],
                "wsb_sentiment"),
        }

        for label, (df, fname) in datasets.items():
            dl_button(
                label    = label,
                data     = df.to_csv(index=False).encode("utf-8"),
                filename = f"{fname}_{today}.csv",
                mime     = "text/csv",
                key      = f"csv_{fname}"
            )

    st.markdown("---")

    # ── Preview ────────────────────────────────────────────
    st.subheader("👁️ Data Preview")

    choice = st.selectbox(
        "Select dataset to preview",
        ["Backtest Results","Portfolio Values",
         "XGBoost Predictions","WSB Sentiment"],
        key="preview_choice"
    )

    preview_map = {
        "Backtest Results"   : bt,
        "Portfolio Values"   : pv.head(50),
        "XGBoost Predictions": xgb[["Date","Ticker","Close",
                                     "Target","xgb_prob",
                                     "xgb_pred"]].head(50),
        "WSB Sentiment"      : wsb[["ticker","date","title_clean",
                                     "score","sentiment_score"]].head(50),
    }

    df_preview = preview_map[choice]
    st.caption(f"Showing {len(df_preview)} rows")
    dark_table(df_preview.reset_index(drop=True))

    st.markdown("---")

    # ── Summary stats ──────────────────────────────────────
    st.subheader("📋 Export Contents")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Backtest Rows",   f"{len(bt)}")
    col2.metric("Portfolio Dates", f"{len(pv):,}")
    col3.metric("Predictions",     f"{len(xgb):,}")
    col4.metric("Sentiment Posts", f"{len(wsb):,}")

    st.info("💡 The Excel workbook is best for sharing with your "
            "supervisor or company. CSVs are best for further "
            "analysis in Python or other tools.")
