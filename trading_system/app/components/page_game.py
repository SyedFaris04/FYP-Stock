"""
components/page_game.py — Stock Prediction Game
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random, time, os
from utils.table import dark_table

from pathlib import Path
BASE      = Path(__file__).resolve().parent.parent.parent
PROCESSED = BASE / "data" / "processed"

DARK_BG="#0f172a"; CARD_BG="#1e293b"; BORDER="#334155"
BLUE="#3b82f6"; GREEN="#10b981"; RED="#ef4444"; AMBER="#f59e0b"; SLATE="#64748b"

TICKERS = ["AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","JPM","JNJ","SPY"]
DIFFICULTY = {
    "🟢 Easy"  : {"days_shown":30,"points":10,"label":"Easy"},
    "🟡 Medium": {"days_shown":10,"points":20,"label":"Medium"},
    "🔴 Hard"  : {"days_shown": 5,"points":30,"label":"Hard"},
}

def dark_layout(**kw):
    b = dict(paper_bgcolor=DARK_BG,plot_bgcolor=DARK_BG,
             font=dict(color="#e2e8f0",size=12),
             margin=dict(l=40,r=20,t=40,b=40),
             hoverlabel=dict(bgcolor=CARD_BG,bordercolor=BORDER))
    b.update(kw); return b

@st.cache_data
def load_stock_data():
    df = pd.read_csv(os.path.join(PROCESSED,"final_dataset.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def get_question(df, diff_cfg):
    ticker = random.choice(TICKERS)
    t_df   = df[df["Ticker"]==ticker].sort_values("Date").reset_index(drop=True)
    days   = diff_cfg["days_shown"]
    # Need enough rows: days shown + 5 future days
    if len(t_df) < days + 10:
        return None
    max_idx = len(t_df) - 6
    min_idx = days
    if max_idx <= min_idx:
        return None
    idx     = random.randint(min_idx, max_idx)
    shown   = t_df.iloc[idx-days:idx].copy()
    future  = t_df.iloc[idx:idx+5].copy()
    start_p = float(shown["Close"].iloc[-1])
    end_p   = float(future["Close"].iloc[-1])
    answer  = "UP" if end_p > start_p else "DOWN"
    pct_chg = ((end_p - start_p) / start_p) * 100
    return {"ticker":ticker,"shown":shown,"future":future,
            "answer":answer,"pct_chg":pct_chg,
            "start_p":start_p,"end_p":end_p}

def render_leaderboard():
    st.subheader("🏆 Live Leaderboard")
    st.caption("Updates every time a score is saved this session")

    # Demo entries always shown
    demo = [
        {"username":"AlphaTrader", "score":2450,"games":12,"accuracy":74.2,"streak":8},
        {"username":"QuantKing",   "score":1980,"games":9, "accuracy":80.0,"streak":6},
        {"username":"BullRunner",  "score":1650,"games":8, "accuracy":76.3,"streak":5},
        {"username":"SentimentPro","score":1200,"games":6, "accuracy":80.0,"streak":4},
        {"username":"WSBLegend",   "score":980, "games":5, "accuracy":76.0,"streak":3},
    ]

    # Merge with session scores
    session_lb = st.session_state.get("leaderboard", [])
    session_names = {e["username"] for e in session_lb}
    combined = session_lb + [d for d in demo if d["username"] not in session_names]
    combined.sort(key=lambda x: x["score"], reverse=True)

    current_user = st.session_state.get("user", {})
    if isinstance(current_user, dict):
        me = current_user.get("username","")
    else:
        me = ""

    for i, e in enumerate(combined[:20]):
        rank  = i+1
        uname = e["username"]
        score = e["score"]
        acc   = e.get("accuracy",0)
        games = e.get("games",0)
        streak= e.get("streak",0)
        is_me = uname == me
        medal = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"#{rank}"
        bg    = "#1d4ed820" if is_me else CARD_BG
        bdr   = BLUE if is_me else BORDER

        st.markdown(f"""
        <div style='background:{bg};border:1px solid {bdr};border-radius:10px;
                    padding:12px 18px;margin-bottom:6px;
                    display:flex;justify-content:space-between;align-items:center;'>
          <div style='display:flex;align-items:center;gap:14px;'>
            <span style='font-size:1rem;min-width:32px;'>{medal}</span>
            <span style='color:#f1f5f9;font-weight:{"700" if is_me else "500"};'>
              {uname}{"  ← you" if is_me else ""}</span>
          </div>
          <div style='display:flex;gap:20px;text-align:right;'>
            <div><div style='color:{BLUE};font-weight:700;'>{score:,}</div>
                 <div style='color:{SLATE};font-size:0.68rem;'>POINTS</div></div>
            <div><div style='color:{GREEN};font-weight:600;'>{acc:.1f}%</div>
                 <div style='color:{SLATE};font-size:0.68rem;'>ACCURACY</div></div>
            <div><div style='color:#e2e8f0;'>{games}</div>
                 <div style='color:{SLATE};font-size:0.68rem;'>GAMES</div></div>
            <div><div style='color:{AMBER};font-weight:600;'>{streak}🔥</div>
                 <div style='color:{SLATE};font-size:0.68rem;'>STREAK</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

    if st.button("🔄 Refresh", key="lb_refresh"):
        st.rerun()

def render():
    user_obj = st.session_state.get("user", {})
    username = user_obj.get("username","Guest") if isinstance(user_obj,dict) else "Guest"

    st.markdown("## 🎮 Stock Prediction Game")
    st.markdown("<p style='color:#94a3b8;margin-top:-12px;'>"
                "Predict UP or DOWN — earn points and climb the leaderboard!</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2 = st.tabs(["🎯 Play", "🏆 Leaderboard"])

    with tab1:
        df = load_stock_data()

        if "game" not in st.session_state:
            st.session_state.game = {
                "score":0,"round":0,"correct":0,"streak":0,
                "best_streak":0,"question":None,"answered":False,
                "last_result":None,"difficulty":"🟢 Easy",
                "active":False,"total_rounds":10,
            }
        g = st.session_state.game

        # ── Start screen ────────────────────────────────
        if not g["active"]:
            st.markdown("### ⚙️ Game Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                diff = st.selectbox("Difficulty",list(DIFFICULTY.keys()),key="diff_sel")
            with col2:
                rounds = st.selectbox("Rounds",[5,10,15,20],index=1,key="rounds_sel")
            with col3:
                st.markdown("<br>",unsafe_allow_html=True)
                if st.button("▶  Start Game",type="primary",use_container_width=True):
                    st.session_state.game.update({
                        "score":0,"round":0,"correct":0,"streak":0,
                        "best_streak":0,"answered":False,"active":True,
                        "question":None,"last_result":None,
                        "difficulty":diff,"total_rounds":rounds,
                    })
                    st.rerun()

            st.markdown("---")
            st.markdown("### 📖 How to Play")
            col1,col2,col3 = st.columns(3)
            cards = [
                ("📈","#3b82f6","Read the Chart","A stock chart is shown. Study the recent price action carefully."),
                ("🎯","#10b981","Make Your Call","Predict if the price will go UP or DOWN over the next 5 trading days."),
                ("⚡","#f59e0b","Score Points","Easy:10 · Medium:20 · Hard:30 pts. Streak x2 after 3, x3 after 5!"),
            ]
            for col,(icon,color,title,desc) in zip([col1,col2,col3],cards):
                with col:
                    st.markdown(f"""
                    <div style='background:{CARD_BG};border:1px solid {BORDER};
                                border-radius:10px;padding:16px;'>
                        <div style='color:{color};font-size:1.5rem;margin-bottom:8px;'>{icon}</div>
                        <div style='color:#f1f5f9;font-weight:700;margin-bottom:6px;'>{title}</div>
                        <div style='color:#94a3b8;font-size:0.82rem;'>{desc}</div>
                    </div>""",unsafe_allow_html=True)
            return

        diff_cfg = DIFFICULTY[g["difficulty"]]

        # ── Score bar ────────────────────────────────────
        pct = int((g["round"]/g["total_rounds"])*100)
        col1,col2,col3,col4,col5 = st.columns(5)
        col1.metric("Round",    f"{g['round']}/{g['total_rounds']}")
        col2.metric("Score",    f"{g['score']:,}")
        col3.metric("Correct",  f"{g['correct']}/{g['round']}")
        col4.metric("Streak",   f"{g['streak']}🔥")
        col5.metric("Difficulty",diff_cfg["label"])
        st.progress(pct)
        st.markdown("<br>",unsafe_allow_html=True)

        # ── Game over ────────────────────────────────────
        if g["round"] >= g["total_rounds"]:
            acc = g["correct"]/g["total_rounds"]*100 if g["total_rounds"]>0 else 0
            emoji = "🏆" if acc>=70 else "🎯" if acc>=50 else "📚"
            st.markdown(f"""
            <div style='background:{CARD_BG};border:2px solid {BLUE};
                        border-radius:16px;padding:32px;text-align:center;'>
                <div style='font-size:3rem;margin-bottom:12px;'>{emoji}</div>
                <div style='color:#f1f5f9;font-size:1.8rem;font-weight:800;'>Game Over!</div>
                <div style='color:{BLUE};font-size:2.5rem;font-weight:800;'>{g["score"]:,} pts</div>
                <div style='color:#94a3b8;margin-top:8px;'>
                    Accuracy: {acc:.1f}% · Best Streak: {g["best_streak"]}🔥</div>
            </div>""",unsafe_allow_html=True)

            st.markdown("<br>",unsafe_allow_html=True)
            col1,col2 = st.columns(2)
            with col1:
                if st.button("💾 Save to Leaderboard",type="primary",
                             use_container_width=True,key="save_score"):
                    if "leaderboard" not in st.session_state:
                        st.session_state.leaderboard = []
                    lb = st.session_state.leaderboard
                    existing = next((e for e in lb if e["username"]==username),None)
                    if existing:
                        existing["score"]    += g["score"]
                        existing["games"]    += 1
                        total_c  = round(existing.get("accuracy",0)/100 * existing["games"] * g["total_rounds"])
                        total_c  = min(total_c + g["correct"], existing["games"] * g["total_rounds"])
                        existing["accuracy"] = round(total_c/(existing["games"]*g["total_rounds"])*100,1)
                        existing["streak"]   = max(existing.get("streak",0),g["best_streak"])
                    else:
                        lb.append({"username":username,"score":g["score"],
                                   "games":1,"accuracy":round(acc,1),
                                   "streak":g["best_streak"]})
                    st.success("✅ Score saved!")
            with col2:
                if st.button("🔄 Play Again",use_container_width=True,key="play_again"):
                    st.session_state.game["active"] = False
                    st.rerun()
            return

        # ── Load question ────────────────────────────────
        if g["question"] is None:
            for _ in range(10):  # retry up to 10 times
                q = get_question(df, diff_cfg)
                if q is not None:
                    break
            if q is None:
                st.error("Could not generate question.")
                return
            g["question"] = q
            g["answered"] = False
            g["q_time"]   = time.time()

        q = g["question"]

        # ── Show last result ─────────────────────────────
        if g["last_result"] is not None:
            last_q = g.get("last_question", q)
            if g["last_result"]:
                st.success(f"✅ Correct! The stock went {last_q['answer']} "
                           f"({last_q['pct_chg']:+.2f}%)")
            else:
                st.error(f"❌ Wrong! The stock went {last_q['answer']} "
                         f"({last_q['pct_chg']:+.2f}%)")

        # ── Chart ────────────────────────────────────────
        shown = q["shown"]
        title = f"What happens next? ({diff_cfg['label']} — {diff_cfg['days_shown']} days shown)"
        if diff_cfg["label"] != "Hard":
            title = f"{q['ticker']} — {title}"

        use_candle = diff_cfg["days_shown"] <= 30
        fig = go.Figure()
        if use_candle:
            fig.add_trace(go.Candlestick(
                x=shown["Date"], open=shown["Open"],
                high=shown["High"], low=shown["Low"], close=shown["Close"],
                name=q["ticker"],
                increasing_line_color=GREEN,
                decreasing_line_color=RED,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=shown["Date"], y=shown["Close"],
                name=q["ticker"], line=dict(color=BLUE,width=2)
            ))
        fig.update_layout(**dark_layout(title=title,height=360,
                          xaxis_rangeslider_visible=False))
        fig.update_yaxes(title_text="Price ($)",showgrid=True,gridcolor=BORDER)
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

        # Streak multiplier
        mult = 3 if g["streak"]>=5 else 2 if g["streak"]>=3 else 1
        pts  = diff_cfg["points"] * mult
        if mult > 1:
            st.markdown(f"<div style='text-align:center;color:{AMBER};"
                        f"font-weight:700;'>🔥 {g['streak']} streak! "
                        f"x{mult} multiplier — {pts} pts this round!</div>",
                        unsafe_allow_html=True)

        # ── Answer or Next ───────────────────────────────
        if not g["answered"]:
            st.markdown("<br>",unsafe_allow_html=True)
            col1,col2 = st.columns(2)
            with col1:
                up   = st.button("📈  UP",  use_container_width=True,
                                 key="btn_up",  type="primary")
            with col2:
                down = st.button("📉  DOWN",use_container_width=True,
                                 key="btn_down")
            if up or down:
                guess   = "UP" if up else "DOWN"
                correct = (guess == q["answer"])
                if correct:
                    g["score"]   += pts
                    g["correct"] += 1
                    g["streak"]  += 1
                    g["best_streak"] = max(g["best_streak"],g["streak"])
                else:
                    g["streak"] = 0
                g["round"]       += 1
                g["answered"]     = True
                g["last_result"]  = correct
                g["last_question"]= q
                g["question"]     = None   # clear so next loads fresh
                st.rerun()
        else:
            st.markdown("<br>",unsafe_allow_html=True)
            if st.button("➡️  Next Question",type="primary",
                         use_container_width=True,key="btn_next"):
                g["last_result"] = None
                g["answered"]    = False
                st.rerun()

    with tab2:
        render_leaderboard()