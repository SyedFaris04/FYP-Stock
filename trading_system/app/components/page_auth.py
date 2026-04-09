"""
components/page_auth.py - Demo auth (no database)
"""
import streamlit as st

# Demo accounts pre-loaded
DEMO_USERS = {
    "demo@sentimenttrader.com": {"username": "DemoUser", "password": "demo123"},
    "admin@sentimenttrader.com": {"username": "Admin", "password": "admin123"},
}

# In-memory user store for new signups
if "registered_users" not in st.session_state:
    st.session_state.registered_users = dict(DEMO_USERS)


def render_login():
    st.markdown("""
    <div style='text-align:center;margin:60px 0 32px;'>
        <div style='font-size:2.5rem;font-weight:800;color:#3b82f6;'>
            📈 SentimentTrader</div>
        <div style='color:#64748b;font-size:0.9rem;margin-top:6px;'>
            Sign in to your account</div>
    </div>""", unsafe_allow_html=True)

    col = st.columns([1,2,1])[1]
    with col:
        with st.form("login_form"):
            email    = st.text_input("Email address")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", type="primary",
                                              use_container_width=True)
            if submitted:
                users = st.session_state.registered_users
                if email in users and users[email]["password"] == password:
                    st.session_state["user"] = {
                        "username": users[email]["username"],
                        "email"   : email,
                    }
                    st.rerun()
                else:
                    st.error("Invalid email or password")

        st.markdown("<p style='text-align:center;color:#64748b;"
                    "font-size:0.85rem;margin-top:16px;'>"
                    "Don't have an account?</p>",
                    unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#475569;"
                    "font-size:0.78rem;'>Demo: demo@sentimenttrader.com "
                    "/ demo123</p>", unsafe_allow_html=True)
        if st.button("Create an account →",
                     use_container_width=True, key="go_signup"):
            st.session_state["auth_page"] = "signup"
            st.rerun()


def render_signup():
    st.markdown("""
    <div style='text-align:center;margin:60px 0 32px;'>
        <div style='font-size:2.5rem;font-weight:800;color:#3b82f6;'>
            📈 SentimentTrader</div>
        <div style='color:#64748b;font-size:0.9rem;margin-top:6px;'>
            Create your account</div>
    </div>""", unsafe_allow_html=True)

    col = st.columns([1,2,1])[1]
    with col:
        with st.form("signup_form"):
            username  = st.text_input("Username")
            email     = st.text_input("Email address")
            password  = st.text_input("Password", type="password")
            password2 = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create Account", type="primary",
                                              use_container_width=True)
            if submitted:
                if not all([username, email, password, password2]):
                    st.error("Please fill in all fields")
                elif password != password2:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                elif email in st.session_state.registered_users:
                    st.error("Email already registered")
                else:
                    st.session_state.registered_users[email] = {
                        "username": username,
                        "password": password,
                    }
                    st.success("Account created! Please sign in.")
                    st.session_state["auth_page"] = "login"
                    st.rerun()

        if st.button("← Back to sign in",
                     use_container_width=True, key="go_login"):
            st.session_state["auth_page"] = "login"
            st.rerun()


def render():
    page = st.session_state.get("auth_page", "login")
    if page == "login":
        render_login()
    else:
        render_signup()
