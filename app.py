import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ===== Load Model & Fitur =====
with open("model_xgb.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.title("📊 Customer Segmentation Prediction")
st.markdown("""
Aplikasi ini memprediksi segmen pelanggan (A, B, C, D) berdasarkan data input.
Model: **XGBoost**
""")

# ===== Form Input =====
st.subheader("Masukkan Data Pelanggan")
user_input = {}
for col in feature_columns:
    user_input[col] = st.number_input(f"{col}", value=0.0)

# ===== Prediksi =====
if st.button("🔍 Prediksi"):
    input_df = pd.DataFrame([user_input], columns=feature_columns)

    # Prediksi
    pred_class = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0]

    seg_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    pred_label = seg_map.get(pred_class, "Unknown")

    st.success(f"**Segmentasi: {pred_label}**")
    st.write("📈 Probabilitas per Segmen:")
    prob_df = pd.DataFrame({
        "Segmen": [seg_map[i] for i in range(len(pred_prob))],
        "Probabilitas": pred_prob
    })
    st.table(prob_df)
