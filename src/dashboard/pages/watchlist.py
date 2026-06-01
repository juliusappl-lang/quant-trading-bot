import streamlit as st

from src.dashboard.db import get_connection
from src.db.repository import add_to_watchlist, get_watchlist, remove_from_watchlist
from src.ingestion.historical import run_historical_ingestion

conn = get_connection()

st.title("Watchlist")
st.caption("Manage the tickers that are monitored for signals.")

# ── Add ticker ───────────────────────────────────────────────────────────────
with st.form("add_ticker", clear_on_submit=True):
    col1, col2, col3 = st.columns([2, 2, 1])
    new_ticker = col1.text_input("Ticker symbol", placeholder="AAPL").strip().upper()
    asset_type = col2.selectbox("Asset type", ["stock", "crypto"])
    col3.write("")
    col3.write("")
    submitted = col3.form_submit_button("Add", use_container_width=True)

if submitted:
    if new_ticker:
        add_to_watchlist(conn, ticker=new_ticker, asset_type=asset_type)
        with st.spinner(f"Fetching historical data for {new_ticker}…"):
            result = run_historical_ingestion(conn, new_ticker)
        st.success(
            f"Added **{new_ticker}** — "
            f"{result['market_rows']} price rows, "
            f"{result['news']} news headlines, "
            f"{result['earnings_synthetic']} earnings events ingested."
        )
        st.rerun()
    else:
        st.warning("Enter a ticker symbol first.")

st.divider()

# ── Active tickers ───────────────────────────────────────────────────────────
watchlist = get_watchlist(conn)

if not watchlist:
    st.info("No tickers yet. Add one above.")
    st.stop()

st.subheader(f"Active tickers ({len(watchlist)})")

header = st.columns([3, 2, 1])
header[0].markdown("**Ticker**")
header[1].markdown("**Type**")

for row in watchlist:
    c1, c2, c3 = st.columns([3, 2, 1])
    c1.write(f"**{row['ticker']}**")
    badge = "🟡 crypto" if row["asset_type"] == "crypto" else "🔵 stock"
    c2.write(badge)
    if c3.button("Remove", key=f"rm_{row['ticker']}", use_container_width=True):
        remove_from_watchlist(conn, row["ticker"])
        st.rerun()
