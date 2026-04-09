"""
utils/table.py
Dark-themed HTML table for SentimentTrader
"""

import streamlit as st
import pandas as pd


def dark_table(df: pd.DataFrame):
    """Renders a fully dark-themed table matching the design system."""

    def style_cell(val):
        val_str = str(val)
        # Positive percentages
        if isinstance(val, str):
            if any(val_str.startswith(p) for p in ["+","40","27","58","61","59","60","56","52"]) and "%" in val_str and not val_str.startswith("-"):
                return f"<td class='pos'>{val}</td>"
            elif val_str.startswith("-") and "%" in val_str:
                return f"<td class='neg'>{val}</td>"
            elif "BUY" in val_str:
                return f"<td class='pos bold'>{val}</td>"
            elif "SELL" in val_str:
                return f"<td class='neg bold'>{val}</td>"
        return f"<td>{val}</td>"

    headers = "".join(f"<th>{col}</th>" for col in df.columns)

    rows = ""
    for i, (_, row) in enumerate(df.iterrows()):
        cells = "".join(style_cell(v) for v in row)
        cls = "alt" if i % 2 == 1 else ""
        rows += f"<tr class='{cls}'>{cells}</tr>"

    html = f"""
<style>
.dt-wrap {{
  border: 1px solid #1e293b;
  border-radius: 12px;
  overflow: hidden;
  margin: 8px 0 20px 0;
  background: #0f172a;
}}
.dt-wrap table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.84rem;
}}
.dt-wrap thead tr {{
  background: #1e293b;
  border-bottom: 1px solid #334155;
}}
.dt-wrap th {{
  color: #64748b;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  padding: 13px 18px;
  text-align: left;
  white-space: nowrap;
}}
.dt-wrap td {{
  color: #cbd5e1;
  padding: 12px 18px;
  border-bottom: 1px solid #0f172a;
  white-space: nowrap;
  font-size: 0.84rem;
}}
.dt-wrap tr.alt td {{ background: #0a1020; }}
.dt-wrap tr:last-child td {{ border-bottom: none; }}
.dt-wrap tr:hover td {{ background: #1e293b !important; }}
.dt-wrap .pos {{ color: #10b981 !important; font-weight: 600; }}
.dt-wrap .neg {{ color: #ef4444 !important; font-weight: 600; }}
.dt-wrap .bold {{ font-weight: 700; }}
</style>
<div class='dt-wrap'>
  <table>
    <thead><tr>{headers}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)
