import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Load Data", page_icon="📥", layout="wide")
st.title("📥 Load Data")

# Resolve app root → regression_app/
APP_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = APP_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded shape: {df.shape}")
        st.dataframe(df.head(20), use_container_width=True)

        if df.shape[1] < 2:
            st.error("Need at least 2 columns (features + target).")
        else:
            target = st.selectbox("Select target column", options=df.columns)
            if st.button("Save dataset", use_container_width=True):
                bundle = {"df": df, "target": target}
                # 1) keep in session for immediate next page
                st.session_state["data_bundle"] = bundle
                # 2) persist to disk for later runs
                joblib.dump(bundle, ARTIFACTS_DIR / "data.pkl")
                st.success(f"Saved {ARTIFACTS_DIR / 'data.pkl'} with target = `{target}`")
                st.info("Now open the **EDA** page from the sidebar.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

st.caption("Upload a CSV (e.g., 50_Startups.csv), pick the target, then click **Save dataset**.")
