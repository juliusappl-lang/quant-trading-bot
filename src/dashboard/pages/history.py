import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.dashboard.db import get_connection
from src.db.repository import get_market_data, get_signals, get_watchlist


def render() -> None:
    st.title("Signal History")
    conn = get_connection()

    tickers = [r["ticker"] for r in get_watchlist(conn)]
    if not tickers:
        st.info("No tickers in watchlist.")
        return

    ticker = st.selectbox("Select Ticker", tickers)
    market_rows = get_market_data(conn, ticker)
    signal_rows = get_signals(conn, ticker=ticker, limit=500)

    if not market_rows:
        st.warning(f"No market data for {ticker}. Run ingestion first.")
        return

    price_df = pd.DataFrame([dict(r) for r in market_rows])
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df.sort_values("date", inplace=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(price_df["date"], price_df["close"], color="#4C72B0", linewidth=1.5, label="Close Price")

    if signal_rows:
        sig_df = pd.DataFrame([dict(r) for r in signal_rows])
        sig_df["date"] = pd.to_datetime(sig_df["created_at"]).dt.normalize()
        price_indexed = price_df.set_index("date")["close"]

        buy_df = sig_df[sig_df["signal"] == "BUY"]
        sell_df = sig_df[sig_df["signal"] == "SELL"]

        if not buy_df.empty:
            buy_prices = price_indexed.reindex(buy_df["date"]).values
            ax.scatter(buy_df["date"], buy_prices, marker="^", color="green", s=80,
                       label="BUY", zorder=5)
        if not sell_df.empty:
            sell_prices = price_indexed.reindex(sell_df["date"]).values
            ax.scatter(sell_df["date"], sell_prices, marker="v", color="red", s=80,
                       label="SELL", zorder=5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_title(f"{ticker} — Price with Signal Markers")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("All Signals")
    if signal_rows:
        df = pd.DataFrame([dict(r) for r in signal_rows])
        display = ["signal", "confidence", "headline", "created_at"]
        st.dataframe(df[[c for c in display if c in df.columns]], use_container_width=True)
    else:
        st.info("No signals generated yet for this ticker.")
