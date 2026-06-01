import datetime
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

from src.dashboard.db import get_connection

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

conn = get_connection()

st.title("Model")
st.caption("Train the MLPClassifier on processed headlines and view performance metrics.")

model_path  = os.getenv("MODEL_PATH", "models/signal_classifier.pkl")
model_abs   = _PROJECT_ROOT / model_path
conf_path   = _PROJECT_ROOT / "models" / "confusion_matrix.png"

# ── Status + training ────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Status")

    if model_abs.exists():
        mtime = model_abs.stat().st_mtime
        last_trained = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        st.success(f"Model present — last trained **{last_trained}**")
    else:
        st.warning("No trained model found.")

    n_processed = conn.execute(
        "SELECT COUNT(*) as n FROM headlines WHERE status = 'processed'"
    ).fetchone()["n"]

    n_total = conn.execute("SELECT COUNT(*) as n FROM headlines").fetchone()["n"]

    m1, m2 = st.columns(2)
    m1.metric("Processed headlines", n_processed)
    m2.metric("Total headlines", n_total)

with col2:
    st.subheader("Train")
    st.write("Fits the model on all processed headlines with known 3-day market outcomes.")

    if n_processed < 10:
        st.warning(f"Need at least 10 processed samples (have {n_processed}).")
    else:
        if st.button("Train Now", type="primary", use_container_width=True):
            with st.spinner("Training…"):
                result = subprocess.run(
                    [sys.executable, "-m", "src.train"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(_PROJECT_ROOT),
                )
            if result.returncode == 0:
                st.success("Training complete!")
                st.code(result.stdout, language="text")
            else:
                st.error("Training failed.")
                st.code(result.stderr, language="text")

# ── Confusion matrix ─────────────────────────────────────────────────────────
if conf_path.exists():
    st.divider()
    st.subheader("Confusion Matrix")
    st.image(str(conf_path))
