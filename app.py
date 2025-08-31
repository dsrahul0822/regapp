import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Regression App", page_icon="📈", layout="wide")

APP_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

st.title("📈 Minimal Regression App")
st.markdown("""
Use the sidebar to navigate:

1. **Load Data** → upload CSV and select target  
2. **EDA** → quick explore  
3. **Encoding** → choose categorical & numeric handling  
4. **Train Model** → fit, evaluate, and save pickles  
5. **Predict** → single or batch (CSV)
""")

need = ["data.pkl", "preprocessor.pkl", "model.pkl"]
st.subheader("Artifacts status")
for f in need:
    ok = "✅" if (ARTIFACTS_DIR / f).exists() else "❌"
    st.write(f"- {ok} `{f}`")

st.caption("Tip: After uploading on **Load Data**, click **Save dataset** before moving to EDA.")
