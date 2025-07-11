import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# 🎨 Page Configuration
# ------------------------
st.set_page_config(page_title="Olist Customer Segment Predictor", page_icon="🛍️")

st.title("🛒 Olist Customer Segmentation Predictor")
st.markdown("Use RFM (Recency, Frequency, Monetary) values to predict which **customer segment** a user belongs to using a trained machine learning model.")
st.divider()

# ------------------------
# 🔢 Segment Labels
# ------------------------
segment_labels = {
    0: "🧊 Low-Value",
    1: "🔥 High-Value",
    2: "⏳ At Risk",
    3: "🆕 New Customers",
    4: "🧪 Others"
}

# ------------------------
# 📥 Input Form
# ------------------------
st.header("📥 Enter Customer Metrics")
with st.form("predict_form"):
    recency = st.number_input(
        "🕒 Recency (days since last purchase)",
        min_value=0,
        help="Number of days since customer's last purchase"
    )
    frequency = st.number_input(
        "🔁 Frequency (number of purchases)",
        min_value=0,
        help="Total number of completed purchases"
    )
    monetary = st.number_input(
        "💰 Monetary Value (R$ spent)",
        min_value=0.0,
        step=10.0,
        help="Total amount the customer has spent"
    )

    submitted = st.form_submit_button("🔍 Predict Segment")

# ------------------------
# 🚀 Predict and Display Result
# ------------------------
if submitted:
    url = "http://127.0.0.1:8000/predict"
    payload = {
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            segment = response.json().get("segment")
            label = segment_labels.get(segment, "Unknown Segment")
            st.success(f"🎯 Predicted Segment: {label} (Cluster {segment})")
        else:
            st.error(f"❌ Server error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Could not connect to FastAPI. Make sure the backend is running at `http://127.0.0.1:8000`.")

# ------------------------
# 📊 Cluster Distribution Chart
# ------------------------
st.divider()
st.header("📊 Cluster Distribution")

try:
    rfm_df = pd.read_csv("clustered_rfm.csv")

    cluster_counts = rfm_df['cluster'].value_counts().sort_index()

    fig, ax = plt.subplots()
    cluster_counts.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Distribution of Customers by Cluster")
    st.pyplot(fig)

except FileNotFoundError:
    st.warning("⚠️ clustered_rfm.csv not found. Please ensure it exists in the same folder as this app.")

# ------------------------
# 📝 Footer
# ------------------------
st.divider()