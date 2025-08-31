import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Predict", page_icon="🎯", layout="wide")
st.title("🎯 Predict")

APP_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = APP_ROOT / "artifacts"
DATA_PKL = ARTIFACTS_DIR / "data.pkl"
PREP_PKL = ARTIFACTS_DIR / "preprocessor.pkl"
MODEL_PKL = ARTIFACTS_DIR / "model.pkl"

# Validate artifacts
missing = [p.name for p in [DATA_PKL, PREP_PKL, MODEL_PKL] if not p.exists()]
if missing:
    st.error(f"Missing artifacts: {missing}. Train your model first.")
    st.stop()

# Load artifacts
preprocessor = joblib.load(PREP_PKL)
model = joblib.load(MODEL_PKL)
if "data_bundle" in st.session_state:
    bundle = st.session_state["data_bundle"]
else:
    bundle = joblib.load(DATA_PKL)
df: pd.DataFrame = bundle["df"]
target = bundle["target"]
orig_cols = [c for c in df.columns if c != target]

tab1, tab2 = st.tabs(["Single Input", "Batch CSV"])

with tab1:
    st.subheader("Single-row input")

    inputs = {}
    for col in orig_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            default = float(df[col].dropna().median()) if df[col].dropna().size else 0.0
            inputs[col] = st.number_input(col, value=float(default))
        else:
            opts = sorted(map(str, df[col].dropna().unique().tolist()))[:1000]
            if not opts:
                opts = [""]
            inputs[col] = st.selectbox(col, options=opts, index=0)

    if st.button("Predict", use_container_width=True):
        X_new = pd.DataFrame([inputs], columns=orig_cols)
        try:
            Xt = preprocessor.transform(X_new)
            pred = model.predict(Xt)[0]
            try:
                pred_val = float(pred)
                st.success(f"Prediction: **{pred_val:.4f}**")
            except Exception:
                st.success(f"Prediction: **{pred}**")
        except Exception as e:
            st.error(f"Failed to predict: {e}")

with tab2:
    st.subheader("Batch CSV prediction")
    up = st.file_uploader("Upload CSV with the same raw feature columns (excluding target)", type=["csv"], key="batch")
    if up is not None:
        try:
            Xb = pd.read_csv(up)
            missing_cols = [c for c in orig_cols if c not in Xb.columns]
            if missing_cols:
                st.error(f"CSV missing columns: {missing_cols}")
            else:
                Xt = preprocessor.transform(Xb[orig_cols])
                preds = model.predict(Xt)
                out = Xb.copy()
                out["prediction"] = preds
                st.dataframe(out.head(50), use_container_width=True)
                st.download_button(
                    "Download predictions CSV",
                    data=out.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Failed to run batch prediction: {e}")
