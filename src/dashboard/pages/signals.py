import pandas as pd
import streamlit as st

from src.dashboard.db import get_connection
from src.db.repository import get_signals, get_watchlist


def render() -> None:
    st.title("Live Signals")

    conn = get_connection()
    tickers = ["All"] + [r["ticker"] for r in get_watchlist(conn)]
    selected = st.selectbox("Filter by Ticker", tickers)
    signal_filter = st.multiselect(
        "Signal Type", ["BUY", "SELL", "HOLD"], default=["BUY", "SELL", "HOLD"]
    )

    rows = get_signals(conn, ticker=None if selected == "All" else selected, limit=200)

    if not rows:
        st.info("No signals yet. Add tickers to the Watchlist and run ingestion.")
        return

    df = pd.DataFrame([dict(r) for r in rows])
    df = df[df["signal"].isin(signal_filter)]

    if df.empty:
        st.info("No signals match the current filter.")
        return

    def _color_signal(val: str) -> str:
        colors = {
            "BUY": "background-color: #d4edda; color: #155724",
            "SELL": "background-color: #f8d7da; color: #721c24",
            "HOLD": "background-color: #fff3cd; color: #856404",
        }
        return colors.get(val, "")

    display_cols = ["ticker", "signal", "confidence", "headline", "created_at"]
    existing = [c for c in display_cols if c in df.columns]
    styled = df[existing].style.map(_color_signal, subset=["signal"])
    st.dataframe(styled, use_container_width=True)
