import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.dashboard.db import get_connection
from src.db.repository import get_market_data, get_signals, get_watchlist

conn = get_connection()

st.title("Signal History")
st.caption("Price chart with BUY/SELL markers and the full signal log.")

tickers = [r["ticker"] for r in get_watchlist(conn)]
if not tickers:
    st.info("No tickers in watchlist.")
    st.stop()

ticker = st.selectbox("Ticker", tickers)

market_rows = get_market_data(conn, ticker)
signal_rows = get_signals(conn, ticker=ticker, limit=500)

if not market_rows:
    st.warning(f"No market data for **{ticker}**. Run ingestion first.")
    st.stop()

# ── Price chart ──────────────────────────────────────────────────────────────
price_df = pd.DataFrame([dict(r) for r in market_rows])
price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()
price_df.sort_values("date", inplace=True)

fig, ax = plt.subplots(figsize=(12, 4))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")

ax.plot(price_df["date"], price_df["close"],
        color="#4C9BE8", linewidth=1.5, label="Close Price")

if signal_rows:
    sig_df = pd.DataFrame([dict(r) for r in signal_rows])
    sig_df["date"] = pd.to_datetime(sig_df["created_at"]).dt.normalize()
    sig_df.sort_values("date", inplace=True)

    merged = pd.merge_asof(
        sig_df[["date", "signal"]],
        price_df[["date", "close"]].rename(columns={"date": "price_date"}),
        left_on="date",
        right_on="price_date",
        direction="nearest",
    )

    buy_df  = merged[merged["signal"] == "BUY"]
    sell_df = merged[merged["signal"] == "SELL"]

    if not buy_df.empty:
        ax.scatter(buy_df["date"], buy_df["close"],
                   marker="^", color="#00c853", s=90, label="BUY", zorder=5)
    if not sell_df.empty:
        ax.scatter(sell_df["date"], sell_df["close"],
                   marker="v", color="#ff1744", s=90, label="SELL", zorder=5)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.tick_params(colors="#aaaaaa")
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")
ax.set_title(f"{ticker} — Close Price with Signals", color="#eeeeee", pad=10)
ax.legend(facecolor="#1a1a2e", labelcolor="#eeeeee")
ax.grid(alpha=0.15, color="#444444")
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# ── Signal table ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("Signal log")

if not signal_rows:
    st.info("No signals generated yet for this ticker.")
    st.stop()

df = pd.DataFrame([dict(r) for r in signal_rows])
display = ["signal", "confidence", "headline", "created_at"]
existing = [c for c in display if c in df.columns]

_SIGNAL_COLORS = {
    "BUY":  "background-color: #d4edda; color: #155724",
    "SELL": "background-color: #f8d7da; color: #721c24",
    "HOLD": "background-color: #fff3cd; color: #856404",
}

styled = (
    df[existing]
    .style
    .map(lambda v: _SIGNAL_COLORS.get(v, ""), subset=["signal"])
    .format({"confidence": "{:.1%}"})
)

st.dataframe(styled, use_container_width=True, hide_index=True)
