"""
utils/db.py
Demo version — uses session state instead of MongoDB
No database required for FYP demo
"""

import streamlit as st
import hashlib
from datetime import datetime

# ── Demo user store (in-memory) ───────────────────────────
DEMO_USERS = {}


def _hash(password):
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username, email, password):
    if email in [u["email"] for u in DEMO_USERS.values()]:
        return False, "Email already registered"
    if username in DEMO_USERS:
        return False, "Username already taken"
    DEMO_USERS[username] = {
        "username"    : username,
        "email"       : email,
        "password"    : _hash(password),
        "created_at"  : datetime.now(),
        "total_score" : 0,
        "games_played": 0,
        "correct"     : 0,
        "best_streak" : 0,
    }
    return True, "Account created!"


def login_user(email, password):
    for username, user in DEMO_USERS.items():
        if user["email"] == email:
            if user["password"] == _hash(password):
                return True, user, "Login successful"
            return False, None, "Incorrect password"
    return False, None, "Email not found"


def get_user(username):
    return DEMO_USERS.get(username)


def update_user(username, updates):
    if username in DEMO_USERS:
        DEMO_USERS[username].update(updates)


def change_password(username, old_pw, new_pw):
    user = get_user(username)
    if not user:
        return False, "User not found"
    if user["password"] != _hash(old_pw):
        return False, "Current password incorrect"
    DEMO_USERS[username]["password"] = _hash(new_pw)
    return True, "Password updated!"


def save_game_score(username, score, correct, total, streak):
    """Save score to session state leaderboard."""
    if "leaderboard" not in st.session_state:
        st.session_state.leaderboard = []

    # Update or insert
    existing = next((e for e in st.session_state.leaderboard
                     if e["username"] == username), None)
    if existing:
        existing["total_score"]  += score
        existing["games_played"] += 1
        existing["total_correct"] += correct
        existing["total_rounds"]  += total
        existing["best_streak"]   = max(
            existing["best_streak"], streak)
    else:
        st.session_state.leaderboard.append({
            "username"     : username,
            "total_score"  : score,
            "games_played" : 1,
            "total_correct": correct,
            "total_rounds" : total,
            "best_streak"  : streak,
        })

    # Also update demo user stats
    if username in DEMO_USERS:
        DEMO_USERS[username]["total_score"]  += score
        DEMO_USERS[username]["games_played"] += 1
        DEMO_USERS[username]["correct"]      += correct
        DEMO_USERS[username]["best_streak"]   = max(
            DEMO_USERS[username].get("best_streak", 0), streak)


def get_leaderboard(limit=20):
    """Return leaderboard from session state."""
    lb = st.session_state.get("leaderboard", [])

    # Add some demo entries so it's not empty
    demo_entries = [
        {"username":"AlphaTrader",  "total_score":2450,
         "games_played":12, "total_correct":89,
         "total_rounds":120, "best_streak":8},
        {"username":"QuantKing",    "total_score":1980,
         "games_played":9,  "total_correct":72,
         "total_rounds":90, "best_streak":6},
        {"username":"BullRunner",   "total_score":1650,
         "games_played":8,  "total_correct":61,
         "total_rounds":80, "best_streak":5},
        {"username":"SentimentPro", "total_score":1200,
         "games_played":6,  "total_correct":48,
         "total_rounds":60, "best_streak":4},
        {"username":"WSBLegend",    "total_score":980,
         "games_played":5,  "total_correct":38,
         "total_rounds":50, "best_streak":3},
    ]

    # Merge real session scores with demo entries
    real_names = {e["username"] for e in lb}
    combined   = lb + [d for d in demo_entries
                       if d["username"] not in real_names]

    # Calculate accuracy and sort
    result = []
    for e in combined:
        rounds   = e.get("total_rounds", e.get("games",1) * 10)
        correct  = e.get("total_correct", 0)
        accuracy = e.get("accuracy", (correct/rounds*100) if rounds>0 else 0)
        result.append({
            "_id"         : e.get("username","?"),
            "username"    : e.get("username","?"),
            "total_score" : e.get("total_score", e.get("score", 0)),
            "games_played": e.get("games_played", e.get("games", 0)),
            "accuracy"    : round(accuracy, 1),
            "best_streak" : e.get("best_streak", e.get("streak", 0)),
        })

    result.sort(key=lambda x: x["total_score"], reverse=True)
    return result[:limit]


def get_db():
    """Stub — returns None in demo mode."""
    return None