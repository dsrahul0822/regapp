import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="EDA", page_icon="🔎", layout="wide")
st.title("🔎 Quick EDA")

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_PKL = APP_ROOT / "artifacts" / "data.pkl"

# Try session_state first, then disk
if "data_bundle" in st.session_state:
    bundle = st.session_state["data_bundle"]
else:
    if not DATA_PKL.exists():
        st.error("No data found. Please visit **Load Data** and click **Save dataset**.")
        st.stop()
    bundle = joblib.load(DATA_PKL)

df: pd.DataFrame = bundle["df"]
target = bundle["target"]

st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]} | **Target:** `{target}`")

st.subheader("Preview")
st.dataframe(df.head(30), use_container_width=True)

st.subheader("Dtypes & Missing")
c1, c2 = st.columns(2)
with c1:
    st.write(df.dtypes.to_frame("dtype"))
with c2:
    miss = df.isna().sum().to_frame("missing_count")
    miss["missing_%"] = (miss["missing_count"] * 100 / len(df)).round(2)
    st.write(miss)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) > 0:
    st.subheader("Correlation (numeric only)")
    corr = df[num_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values)
    ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=90)
    ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, use_container_width=True)
else:
    st.info("No numeric columns found for correlation.")
