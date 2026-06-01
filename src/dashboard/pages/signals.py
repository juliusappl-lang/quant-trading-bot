import pandas as pd
import streamlit as st

from src.dashboard.db import get_connection
from src.db.repository import get_signals, get_watchlist

conn = get_connection()

st.title("Live Signals")
st.caption("BUY / SELL / HOLD signals generated from recent headlines.")

# ── Filters ────────────────────────────────────────────────────────────────
tickers = ["All"] + [r["ticker"] for r in get_watchlist(conn)]
col1, col2 = st.columns([2, 3])
with col1:
    selected = st.selectbox("Ticker", tickers)
with col2:
    signal_filter = st.multiselect(
        "Signal type", ["BUY", "SELL", "HOLD"], default=["BUY", "SELL", "HOLD"]
    )

rows = get_signals(conn, ticker=None if selected == "All" else selected, limit=200)

if not rows:
    st.info("No signals yet. Add tickers to the Watchlist and run ingestion.")
    st.stop()

df = pd.DataFrame([dict(r) for r in rows])
df = df[df["signal"].isin(signal_filter)]

if df.empty:
    st.info("No signals match the current filter.")
    st.stop()

# ── Summary metrics ─────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total", len(df))
m2.metric("BUY",  int((df["signal"] == "BUY").sum()),  delta_color="normal")
m3.metric("SELL", int((df["signal"] == "SELL").sum()), delta_color="inverse")
m4.metric("HOLD", int((df["signal"] == "HOLD").sum()))

st.divider()

# ── Table ───────────────────────────────────────────────────────────────────
_SIGNAL_COLORS = {
    "BUY":  "background-color: #d4edda; color: #155724",
    "SELL": "background-color: #f8d7da; color: #721c24",
    "HOLD": "background-color: #fff3cd; color: #856404",
}

display_cols = ["ticker", "signal", "confidence", "headline", "created_at"]
existing = [c for c in display_cols if c in df.columns]

styled = (
    df[existing]
    .style
    .map(lambda v: _SIGNAL_COLORS.get(v, ""), subset=["signal"])
    .format({"confidence": "{:.1%}"})
)

st.dataframe(styled, use_container_width=True, hide_index=True)
