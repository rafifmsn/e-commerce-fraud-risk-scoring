import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Ecom Fraud Scoring", page_icon="🛡️", layout="wide")

# --- ASSETS LOADING ---
MODEL_PATH = "./train/model.pkl"
META_PATH = "./train/model_metadata.json"

@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
    return model, metadata

model, metadata = load_resources()
EXPECTED_FEATURES = metadata.get("features_used", []) if metadata else []

# --- LOGIC ---
def get_risk_tier(score: float) -> str:
    if score >= 80: return "High Risk"
    if score >= 30: return "Medium Risk"
    return "Low Risk"

def get_tier_color(tier: str):
    colors = {"High Risk": "#ff4b4b", "Medium Risk": "#ffa500", "Low Risk": "#28a745"}
    return colors.get(tier, "#000000")

def process_dataframe(df: pd.DataFrame):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Presence of prefix 'www'": "Presence of prefix 'www' "})
    
    missing = set(EXPECTED_FEATURES) - set(df.columns)
    if missing:
        st.error(f"Missing columns: {list(missing)}")
        return None
        
    df_prepped = df.reindex(columns=EXPECTED_FEATURES).replace({"": np.nan, None: np.nan})
    proba = model.predict_proba(df_prepped)[:, 1]
    
    df["probability_fraud"] = proba.astype(float).round(6)
    df["risk_score"] = (proba * 100).astype(float).round(4)
    df["risk_tier"] = df["risk_score"].apply(get_risk_tier)
    
    return df

# --- UI LAYOUT ---
st.title("🛡️ E-commerce Fraud Risk Engine")
st.markdown("Assess shop risk levels using machine learning.")

if not model:
    st.error("Model or metadata files not found in `./train/`.")
    st.stop()

tabs = st.tabs(["📁 Batch Processing", "⌨️ Single Record Input"])

with tabs[0]:
    st.subheader("CSV Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)
        with st.spinner("Analyzing risk..."):
            results = process_dataframe(raw_df)
            
            if results is not None:
                st.success("Analysis Complete!")
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Records", len(results))
                m2.metric("High Risk Identified", len(results[results["risk_tier"] == "High Risk"]))
                m3.metric("Avg Risk Score", f"{results['risk_score'].mean():.2f}")

                st.dataframe(results, use_container_width=True)

                # The 'utf-8-sig' allows Excel to detect the encoding and decimals properly
                csv_data = results.to_csv(index=False).encode('utf-8-sig')
                
                st.download_button(
                    label="📥 Download Scored CSV",
                    data=csv_data,
                    file_name="scored_results.csv",
                    mime="text/csv"
                )

with tabs[1]:
    st.subheader("Manual Data Entry")
    col1, col2 = st.columns([1, 1])
    with col1:
        template = {feat: 0 for feat in EXPECTED_FEATURES}
        json_input = st.text_area("Input JSON features", value=json.dumps(template, indent=2), height=300)
    with col2:
        if st.button("Generate Score", use_container_width=True):
            try:
                data = json.loads(json_input)
                df_single = pd.DataFrame([data])
                result_df = process_dataframe(df_single)
                if result_df is not None:
                    res = result_df.iloc[0]
                    tier = res["risk_tier"]
                    st.markdown(f"### Result: <span style='color:{get_tier_color(tier)}'>{tier}</span>", unsafe_allow_html=True)
                    st.metric("Probability", f"{res['probability_fraud']:.6f}")
                    st.metric("Risk Score", f"{res['risk_score']:.2f}")
                    st.json(res.to_dict())
            except Exception as e:
                st.error(f"Invalid JSON format: {e}")

st.divider()
st.caption(f"Engine: {metadata.get('model_name', 'Production-v1')} | Features: {len(EXPECTED_FEATURES)}")