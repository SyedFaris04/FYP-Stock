import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st

st.set_page_config(page_title="SentimentTrader",page_icon="📈",
                   layout="wide",initial_sidebar_state="expanded")

st.markdown("""<style>
.stApp{background-color:#0f172a!important}
.stApp>header{background-color:#0f172a!important;border-bottom:1px solid #1e293b!important}
*{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif!important}
section[data-testid="stSidebar"]{background-color:#1e293b!important;border-right:1px solid #334155!important}
section[data-testid="stSidebar"]>div{padding:0!important}
section[data-testid="stSidebar"] .stRadio>label{display:none!important}
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]>div:first-child{display:none!important}
section[data-testid="stSidebar"] .stRadio label{display:flex!important;align-items:center!important;padding:10px 20px!important;margin:2px 8px!important;border-radius:8px!important;cursor:pointer!important;color:#94a3b8!important;font-size:0.875rem!important;font-weight:500!important}
section[data-testid="stSidebar"] .stRadio label:hover{background:#334155!important;color:#e2e8f0!important}
section[data-testid="stSidebar"] .stRadio div[data-baseweb="radio"]:has(input:checked) label{background:#2563eb!important;color:#fff!important;font-weight:600!important}
h1,h2,h3,h4{color:#f1f5f9!important;font-weight:700!important}
p,li{color:#cbd5e1!important}
label{color:#94a3b8!important}
small,.stCaption{color:#64748b!important}
div[data-testid="metric-container"]{background:#1e293b!important;border:1px solid #334155!important;border-radius:12px!important;padding:20px!important}
div[data-testid="metric-container"] label{color:#64748b!important;font-size:0.72rem!important;font-weight:700!important;text-transform:uppercase!important;letter-spacing:0.08em!important}
div[data-testid="stMetricValue"]{color:#f1f5f9!important;font-size:1.65rem!important;font-weight:700!important}
.stTabs [data-baseweb="tab-list"]{background:#1e293b!important;border-radius:10px!important;padding:4px!important;border:1px solid #334155!important;gap:2px!important}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#94a3b8!important;border-radius:7px!important;font-weight:500!important;font-size:0.85rem!important;border:none!important;padding:8px 18px!important}
.stTabs [aria-selected="true"]{background:#3b82f6!important;color:#fff!important;font-weight:600!important}
.stButton>button{background:#3b82f6!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important}
.stButton>button:hover{background:#2563eb!important}
.stSelectbox>div>div,.stMultiSelect>div>div{background:#1e293b!important;border:1px solid #334155!important;border-radius:8px!important;color:#e2e8f0!important}
.stTextInput input,.stNumberInput input,.stDateInput input{background:#1e293b!important;border:1px solid #334155!important;border-radius:8px!important;color:#e2e8f0!important}
.stSelectbox label,.stMultiSelect label,.stNumberInput label,.stDateInput label,.stSlider label,.stTextInput label{color:#64748b!important;font-size:0.72rem!important;font-weight:700!important;text-transform:uppercase!important;letter-spacing:0.06em!important}
.stProgress>div>div{background:#1e293b!important;border-radius:99px!important;height:6px!important}
.stProgress>div>div>div{background:linear-gradient(90deg,#3b82f6,#06b6d4)!important;border-radius:99px!important}
details{background:#1e293b!important;border:1px solid #334155!important;border-radius:10px!important;overflow:hidden!important}
details summary{color:#e2e8f0!important;font-weight:500!important;padding:14px 16px!important}
div[data-testid="stForm"]{background:#1e293b!important;border:1px solid #334155!important;border-radius:12px!important;padding:20px!important}
div[data-testid="stPlotlyChart"]{border:1px solid #1e293b!important;border-radius:12px!important;overflow:hidden!important}
hr{border-color:#1e293b!important;margin:1.5rem 0!important}
#MainMenu,footer{visibility:hidden!important}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#0f172a}
::-webkit-scrollbar-thumb{background:#334155;border-radius:99px}

/* Fix expander arrow overlap */
.streamlit-expanderHeader { position: relative !important; }
.streamlit-expanderHeader svg { flex-shrink: 0 !important; }
details > summary > div { display: flex !important; align-items: center !important; gap: 8px !important; }
details > summary p { margin: 0 !important; color: #e2e8f0 !important; }

/* Fix selectbox dropdown options */
[data-baseweb="popover"] { background: #1e293b !important; border: 1px solid #334155 !important; border-radius: 10px !important; }
[data-baseweb="menu"] { background: #1e293b !important; }
[data-baseweb="option"] { background: #1e293b !important; color: #e2e8f0 !important; }
[data-baseweb="option"]:hover { background: #334155 !important; color: #ffffff !important; }
[aria-selected="true"][data-baseweb="option"] { background: #1d4ed8 !important; color: #ffffff !important; }
li[role="option"] { color: #e2e8f0 !important; background: #1e293b !important; }
li[role="option"]:hover { background: #334155 !important; color: #ffffff !important; }

/* Fix markdown table contrast */
table { border-collapse: collapse !important; width: 100% !important; }
table thead tr { background: #1e293b !important; }
table th { color: #94a3b8 !important; font-size: 0.78rem !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; padding: 12px 16px !important; border-bottom: 1px solid #334155 !important; }
table td { color: #e2e8f0 !important; padding: 10px 16px !important; border-bottom: 1px solid #1e293b !important; background: #0f172a !important; }
table tr:nth-child(even) td { background: #0a1020 !important; }

/* Fix latex/math text */
.katex { color: #e2e8f0 !important; }
.katex-html { color: #e2e8f0 !important; }

/* Fix general text visibility */
.stMarkdown, .stMarkdown p, .stMarkdown li { color: #cbd5e1 !important; }
.stMarkdown strong { color: #f1f5f9 !important; }
.stMarkdown code { color: #06b6d4 !important; background: #1e293b !important; padding: 2px 6px !important; border-radius: 4px !important; }

/* Fix multiselect tags */
[data-baseweb="tag"] { background: #2563eb !important; color: #ffffff !important; }
[data-baseweb="tag"] span { color: #ffffff !important; }

/* Fix date input */
.stDateInput [data-baseweb="input"] { background: #1e293b !important; }

/* Improve caption/small text */
.stCaption p { color: #64748b !important; }
</style>""", unsafe_allow_html=True)


# Apply theme
if st.session_state.get("theme", "dark") == "light":
    st.markdown("""<style>
    .stApp{background-color:#f8fafc!important;color:#0f172a!important}
    section[data-testid="stSidebar"]{background-color:#f1f5f9!important;border-right:1px solid #e2e8f0!important}
    div[data-testid="metric-container"]{background:#ffffff!important;border:1px solid #e2e8f0!important}
    div[data-testid="stMetricValue"]{color:#0f172a!important}
    h1,h2,h3,h4{color:#0f172a!important}
    p,li{color:#334155!important}
    .stTabs [data-baseweb="tab-list"]{background:#e2e8f0!important;border:1px solid #cbd5e1!important}
    .stTabs [data-baseweb="tab"]{color:#475569!important}
    
/* Fix expander arrow overlap */
.streamlit-expanderHeader { position: relative !important; }
.streamlit-expanderHeader svg { flex-shrink: 0 !important; }
details > summary > div { display: flex !important; align-items: center !important; gap: 8px !important; }
details > summary p { margin: 0 !important; color: #e2e8f0 !important; }

/* Fix selectbox dropdown options */
[data-baseweb="popover"] { background: #1e293b !important; border: 1px solid #334155 !important; border-radius: 10px !important; }
[data-baseweb="menu"] { background: #1e293b !important; }
[data-baseweb="option"] { background: #1e293b !important; color: #e2e8f0 !important; }
[data-baseweb="option"]:hover { background: #334155 !important; color: #ffffff !important; }
[aria-selected="true"][data-baseweb="option"] { background: #1d4ed8 !important; color: #ffffff !important; }
li[role="option"] { color: #e2e8f0 !important; background: #1e293b !important; }
li[role="option"]:hover { background: #334155 !important; color: #ffffff !important; }

/* Fix markdown table contrast */
table { border-collapse: collapse !important; width: 100% !important; }
table thead tr { background: #1e293b !important; }
table th { color: #94a3b8 !important; font-size: 0.78rem !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; padding: 12px 16px !important; border-bottom: 1px solid #334155 !important; }
table td { color: #e2e8f0 !important; padding: 10px 16px !important; border-bottom: 1px solid #1e293b !important; background: #0f172a !important; }
table tr:nth-child(even) td { background: #0a1020 !important; }

/* Fix latex/math text */
.katex { color: #e2e8f0 !important; }
.katex-html { color: #e2e8f0 !important; }

/* Fix general text visibility */
.stMarkdown, .stMarkdown p, .stMarkdown li { color: #cbd5e1 !important; }
.stMarkdown strong { color: #f1f5f9 !important; }
.stMarkdown code { color: #06b6d4 !important; background: #1e293b !important; padding: 2px 6px !important; border-radius: 4px !important; }

/* Fix multiselect tags */
[data-baseweb="tag"] { background: #2563eb !important; color: #ffffff !important; }
[data-baseweb="tag"] span { color: #ffffff !important; }

/* Fix date input */
.stDateInput [data-baseweb="input"] { background: #1e293b !important; }

/* Improve caption/small text */
.stCaption p { color: #64748b !important; }
</style>""", unsafe_allow_html=True)

from components import (
    page_dashboard, page_stock_analysis, page_model_comparison,
    page_backtesting, page_sentiment, page_education,
    page_predictions, page_portfolio, page_export,
    page_game, page_auth,
    page_simulator, page_alerts,
)

PAGES = {
    "📊 Dashboard"       : page_dashboard,
    "📈 Stock Analysis"  : page_stock_analysis,
    "🤖 Model Comparison": page_model_comparison,
    "💰 Backtesting"     : page_backtesting,
    "💬 Sentiment"       : page_sentiment,
    "📚 Education"       : page_education,
    "🎯 Predictions"     : page_predictions,
    "🧠 AI Copilot"      : page_predictions,   # alias — copilot lives inside predictions
    "🚨 Smart Alerts"    : page_alerts,
    "🎮 Paper Simulator" : page_simulator,
    "👤 My Portfolio"    : page_portfolio,
    "📤 Export"          : page_export,
    "🎲 Stock Game"      : page_game,
}

user = st.session_state.get("user")

if not user:
    page_auth.render()
    st.stop()

username = user["username"]

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:20px 20px 12px;'>
      <div style='font-size:1.2rem;font-weight:800;color:#3b82f6;'>📈 SentimentTrader</div>
      <div style='font-size:0.7rem;color:#475569;margin-top:3px;font-weight:500;letter-spacing:0.02em;'>
        QUANTITATIVE RESEARCH PLATFORM</div>
    </div>
    <div style='height:1px;background:#334155;margin:0 16px 8px;'></div>
    """, unsafe_allow_html=True)

    selected = st.radio("", list(PAGES.keys()),
                        label_visibility="hidden", key="main_nav")

    st.markdown("<div style='height:1px;background:#334155;margin:8px 16px;'></div>",
                unsafe_allow_html=True)

    # ── User profile card ──────────────────────────────────
    st.markdown(f"""
    <div style='padding:12px 20px 8px;'>
      <div style='color:#475569;font-size:0.65rem;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;'>
        Signed In</div>
      <div style='display:flex;align-items:center;gap:10px;margin-bottom:12px;'>
        <div style='width:34px;height:34px;border-radius:50%;background:#2563eb;
                    display:flex;align-items:center;justify-content:center;
                    color:white;font-weight:700;font-size:1rem;flex-shrink:0;'>
          {username[0].upper()}</div>
        <div>
          <div style='color:#f1f5f9;font-weight:600;font-size:0.88rem;'>{username}</div>
          <div style='color:#475569;font-size:0.7rem;'>{user["email"]}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Account actions as buttons ─────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("👤 Profile",  use_container_width=True, key="btn_profile"):
            st.session_state["sidebar_view"] = "profile"
    with col_b:
        if st.button("⚙️ Settings", use_container_width=True, key="btn_settings"):
            st.session_state["sidebar_view"] = "settings"

    if st.button("🎮 Game Stats", use_container_width=True, key="btn_gamestats"):
        st.session_state["sidebar_view"] = "gamestats"

    if st.button("🚪 Sign Out", use_container_width=True, key="btn_signout"):
        for k in ["user","game","auth_page","sidebar_view","leaderboard"]:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown("""
    <div style='height:1px;background:#334155;margin:8px 16px 12px;'></div>
    <div style='padding:0 20px 20px;font-size:0.68rem;color:#334155;
                font-weight:600;text-transform:uppercase;letter-spacing:0.05em;line-height:2;'>
      FYP × Alpha Technologies<br>Data: 2015–2024
    </div>""", unsafe_allow_html=True)

# ── Sidebar overlay views ──────────────────────────────────
sidebar_view = st.session_state.get("sidebar_view")

if sidebar_view == "profile":
    from utils.db import get_leaderboard
    st.markdown(f"## 👤 Profile — {username}")
    st.markdown("---")
    lb  = get_leaderboard()
    me  = next((e for e in lb if e["username"]==username), None)
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Total Score",  f"{me['total_score']:,}"  if me else "0")
    col2.metric("Games Played", f"{me['games_played']}"   if me else "0")
    col3.metric("Accuracy",     f"{me['accuracy']:.1f}%"  if me else "0%")
    col4.metric("Best Streak",  f"{me['best_streak']}🔥"  if me else "0🔥")
    st.markdown("---")
    st.info("Play **🎮 Stock Game** to earn points!")
    if st.button("← Back", key="back_profile"):
        del st.session_state["sidebar_view"]
        st.rerun()
    st.stop()

elif sidebar_view == "settings":
    from utils.db import change_password, update_user
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["👤 Edit Profile","🔐 Change Password","🎨 Appearance"])
    with tab1:
        with st.form("edit_profile"):
            new_name = st.text_input("Display Name", value=username)
            if st.form_submit_button("Save Changes", type="primary"):
                update_user(username, {"username": new_name})
                st.session_state["user"]["username"] = new_name
                st.success("✅ Profile updated!")
    with tab2:
        with st.form("change_pw"):
            old_pw  = st.text_input("Current Password", type="password")
            new_pw  = st.text_input("New Password",     type="password")
            new_pw2 = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Update Password", type="primary"):
                if new_pw != new_pw2:
                    st.error("Passwords don't match")
                else:
                    ok, msg = change_password(username, old_pw, new_pw)
                    st.success(msg) if ok else st.error(msg)
    with tab3:
        st.markdown("### 🎨 Theme")
        current = st.session_state.get("theme", "dark")
        st.markdown(f"Current theme: **{current.title()} Mode**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌙 Dark Mode",
                         type="primary" if current=="dark" else "secondary",
                         use_container_width=True, key="set_dark"):
                st.session_state["theme"] = "dark"
                st.success("Dark mode enabled!")
                st.rerun()
        with col2:
            if st.button("☀️ Light Mode",
                         type="primary" if current=="light" else "secondary",
                         use_container_width=True, key="set_light"):
                st.session_state["theme"] = "light"
                st.success("Light mode enabled!")
                st.rerun()
        st.info("💡 Changes apply immediately across all pages.")
    if st.button("← Back", key="back_settings"):
        del st.session_state["sidebar_view"]
        st.rerun()
    st.stop()

elif sidebar_view == "gamestats":
    from utils.db import get_leaderboard
    from utils.table import dark_table
    import pandas as pd
    st.markdown("## 🎮 My Game Stats")
    st.markdown("---")
    lb = get_leaderboard()
    me = next((e for e in lb if e["username"]==username), None)
    if me:
        col1,col2,col3,col4 = st.columns(4)
        col1.metric("Total Score",  f"{me['total_score']:,}")
        col2.metric("Games Played", f"{me['games_played']}")
        col3.metric("Accuracy",     f"{me['accuracy']:.1f}%")
        col4.metric("Best Streak",  f"{me['best_streak']}🔥")
        st.markdown("---")
        st.subheader("🏆 Leaderboard Standing")
        rank = next((i+1 for i,e in enumerate(lb) if e["username"]==username), None)
        if rank:
            st.success(f"You are ranked **#{rank}** out of {len(lb)} players!")
    else:
        st.info("No games played yet. Go to **🎮 Stock Game** to start!")
    if st.button("← Back", key="back_gamestats"):
        del st.session_state["sidebar_view"]
        st.rerun()
    st.stop()

# ── Main page ──────────────────────────────────────────────
PAGES[selected].render()