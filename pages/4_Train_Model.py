import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Train Model", page_icon="🤖", layout="wide")
st.title("🤖 Train & Save Model")

APP_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = APP_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PKL = ARTIFACTS_DIR / "data.pkl"
PREP_PKL = ARTIFACTS_DIR / "preprocessor.pkl"

# Load bundle & preprocessor
if "data_bundle" in st.session_state:
    bundle = st.session_state["data_bundle"]
elif DATA_PKL.exists():
    bundle = joblib.load(DATA_PKL)
else:
    st.error("No data found. Please visit **Load Data** and click **Save dataset**.")
    st.stop()

if not PREP_PKL.exists():
    st.error("No preprocessor found. Please configure it on **Encoding**.")
    st.stop()

df: pd.DataFrame = bundle["df"]
target = bundle["target"]
preprocessor = joblib.load(PREP_PKL)

test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.number_input("Random state", 0, 9999, 42, 1)

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

model_type = st.selectbox("Model", ["LinearRegression", "RandomForestRegressor"], index=0)
if model_type == "RandomForestRegressor":
    n_estimators = st.slider("n_estimators", 50, 500, 200, 50)
    max_depth = st.slider("max_depth", 2, 50, 10, 1)
else:
    n_estimators = None
    max_depth = None

if st.button("Train & Save", use_container_width=True):
    # IMPORTANT: fit preprocessor on TRAIN ONLY to avoid leakage
    preprocessor.fit(X_train)
    Xtr = preprocessor.transform(X_train)
    Xte = preprocessor.transform(X_test)

    if model_type == "LinearRegression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )

    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    # Save artifacts
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.pkl")  # fitted on TRAIN
    joblib.dump(model, ARTIFACTS_DIR / "model.pkl")

    # Save feature names (optional, helps debug/UI)
    try:
        feat_names = preprocessor.get_feature_names_out()
        joblib.dump(list(feat_names), ARTIFACTS_DIR / "feature_list.pkl")
        feat_msg = ", feature_list.pkl"
    except Exception:
        feat_msg = ""

    st.success(f"Saved: preprocessor.pkl, model.pkl{feat_msg}")
    st.write({"rmse": rmse, "r2": r2})
