import importlib

import streamlit as st

st.set_page_config(page_title="Signal Engine", layout="wide", page_icon="📈")

_PAGES = {
    "Live Signals": "src.dashboard.pages.signals",
    "Watchlist": "src.dashboard.pages.watchlist",
    "Signal History": "src.dashboard.pages.history",
    "Model": "src.dashboard.pages.model",
}

page = st.sidebar.radio("Navigation", list(_PAGES.keys()))
module = importlib.import_module(_PAGES[page])
module.render()
