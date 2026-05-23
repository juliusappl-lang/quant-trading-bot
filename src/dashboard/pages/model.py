import datetime
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

from src.dashboard.db import get_connection


def render() -> None:
    st.title("Model")
    conn = get_connection()

    model_path = os.getenv("MODEL_PATH", "models/signal_classifier.pkl")
    confusion_path = "models/confusion_matrix.png"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Status")
        if Path(model_path).exists():
            mtime = Path(model_path).stat().st_mtime
            last_trained = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            st.success(f"Model trained — last updated: {last_trained}")
        else:
            st.warning("No trained model found. Click 'Train Now' to create one.")

        n_processed = conn.execute(
            "SELECT COUNT(*) as n FROM headlines WHERE status = 'processed'"
        ).fetchone()["n"]
        st.metric("Processed Headlines (training data)", n_processed)

    with col2:
        st.subheader("Train Model")
        st.write("Trains on all processed headlines with known 3-day market outcomes.")
        if st.button("Train Now", type="primary"):
            with st.spinner("Training model..."):
                result = subprocess.run(
                    [sys.executable, "-m", "src.train"],
                    capture_output=True, text=True, timeout=300,
                )
            if result.returncode == 0:
                st.success("Training complete!")
                st.text(result.stdout)
            else:
                st.error("Training failed.")
                st.text(result.stderr)

    if Path(confusion_path).exists():
        st.subheader("Confusion Matrix")
        st.image(confusion_path)
