from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Signal Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

_HERE = Path(__file__).parent

pg = st.navigation([
    st.Page(str(_HERE / "pages" / "signals.py"),  title="Live Signals",   icon="📡"),
    st.Page(str(_HERE / "pages" / "watchlist.py"), title="Watchlist",      icon="📋"),
    st.Page(str(_HERE / "pages" / "history.py"),   title="Signal History", icon="📊"),
    st.Page(str(_HERE / "pages" / "model.py"),     title="Model",          icon="🤖"),
])

pg.run()
