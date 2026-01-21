import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("xgb_churn_model.pkl")

THRESHOLD = 0.35

st.title("📊 Customer Churn Prediction (Batch CSV)")
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("### Uploaded Preview")
    st.dataframe(df.head())

    if "customerID" not in df.columns:
        st.error("CSV must contain 'customerID'")
        st.stop()

    customer_ids = df["customerID"].copy()
    df = df.drop(columns=["customerID"])

    # ------- FIX TotalCharges only -------
    if "TotalCharges" not in df.columns:
        st.error("CSV missing 'TotalCharges' column.")
        st.stop()

    # strip then convert
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # drop rows where TotalCharges is missing
    before = len(df)
    df = df.dropna(subset=["TotalCharges"])
    after = len(df)

    st.write(f"Dropped {before - after} rows due to missing TotalCharges.")

    if len(df) == 0:
        st.error("No valid rows left for prediction.")
        st.stop()

    # ---- IMPORTANT: Pipeline handles preprocessing ----
    churn_probs = model.predict_proba(df)[:, 1]
    churn_label = (churn_probs >= THRESHOLD).astype(int)

    result = pd.DataFrame({
        "customerID": customer_ids.iloc[df.index],
        "churn_prob": churn_probs,
        "churn_label": churn_label
    })

    st.write("### Prediction Results")
    st.dataframe(result.head())

    st.download_button(
        "📥 Download predictions CSV",
        result.to_csv(index=False).encode(),
        "churn_predictions.csv",
        "text/csv"
    )

    st.success("Done!")
