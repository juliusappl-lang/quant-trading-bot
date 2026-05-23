import streamlit as st

from src.dashboard.db import get_connection
from src.db.repository import add_to_watchlist, get_watchlist, remove_from_watchlist


def render() -> None:
    st.title("Watchlist")
    conn = get_connection()

    st.subheader("Add Ticker")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        new_ticker = st.text_input("Ticker Symbol", placeholder="AAPL").strip().upper()
    with col2:
        asset_type = st.selectbox("Asset Type", ["stock", "crypto"])
    with col3:
        st.write("")
        st.write("")
        if st.button("Add") and new_ticker:
            add_to_watchlist(conn, ticker=new_ticker, asset_type=asset_type)
            st.success(f"Added {new_ticker}")
            st.rerun()

    st.subheader("Active Tickers")
    watchlist = get_watchlist(conn)
    if not watchlist:
        st.info("No tickers in watchlist. Add one above.")
        return

    for row in watchlist:
        c1, c2, c3 = st.columns([3, 2, 1])
        c1.write(f"**{row['ticker']}**")
        c2.write(row["asset_type"])
        if c3.button("Remove", key=f"rm_{row['ticker']}"):
            remove_from_watchlist(conn, row["ticker"])
            st.rerun()
