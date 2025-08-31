import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Encoding", page_icon="🧩", layout="wide")
st.title("🧩 Encoding & Scaling")

APP_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = APP_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PKL = ARTIFACTS_DIR / "data.pkl"

# Load data bundle
if "data_bundle" in st.session_state:
    bundle = st.session_state["data_bundle"]
elif DATA_PKL.exists():
    bundle = joblib.load(DATA_PKL)
else:
    st.error("No data found. Please visit **Load Data** and click **Save dataset**.")
    st.stop()

df: pd.DataFrame = bundle["df"]
target = bundle["target"]

X = df.drop(columns=[target])
num_default = X.select_dtypes(include=[np.number]).columns.tolist()
cat_default = [c for c in X.columns if c not in num_default]

st.write("Configure your preprocessing:")
num_cols = st.multiselect("Numeric columns", options=X.columns.tolist(), default=num_default)
cat_cols = st.multiselect("Categorical columns", options=[c for c in X.columns if c not in num_cols], default=cat_default)

st.divider()
st.subheader("Imputation & Scaling")
num_impute_strategy = st.selectbox("Numeric imputation", ["mean", "median", "most_frequent"], index=0)
scale_choice = st.selectbox("Numeric scaling", ["None", "StandardScaler", "MinMaxScaler"], index=1)
cat_impute = st.selectbox("Categorical imputation", ["most_frequent", "constant"], index=0)

if st.button("Build & Fit Preprocessor", use_container_width=True):
    if len(num_cols) + len(cat_cols) == 0:
        st.error("Please select at least one feature.")
        st.stop()

    num_steps = [("imputer", SimpleImputer(strategy=num_impute_strategy))]
    if scale_choice == "StandardScaler":
        num_steps.append(("scaler", StandardScaler()))
    elif scale_choice == "MinMaxScaler":
        num_steps.append(("scaler", MinMaxScaler()))
    num_pipe = Pipeline(steps=num_steps) if len(num_cols) > 0 else "drop"

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cat_impute, fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]) if len(cat_cols) > 0 else "drop"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Fit on full X to preview transformed feature names (model page will refit on TRAIN only)
    preprocessor.fit(X)

    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.pkl")

    try:
        feat_names = preprocessor.get_feature_names_out()
        st.success(f"Saved {ARTIFACTS_DIR / 'preprocessor.pkl'} with {len(feat_names)} output features.")
        st.write("Transformed feature sample:", list(feat_names)[:30], "...")
    except Exception:
        st.success(f"Saved {ARTIFACTS_DIR / 'preprocessor.pkl'}")
        st.info("Feature names not available (older scikit-learn).")
