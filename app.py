import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🛡️ Intelligent Fraud Detection & Risk Scoring System")

st.write(
    "Upload a transaction CSV file to detect suspicious transactions "
    "and calculate risk scores."
)

# --------------------------------------------------
# Model Path
# --------------------------------------------------

MODEL_PATH = (
    "/Users/shyamchauhan/Desktop/home/codes/"
    "fraud-detection/saved_models/fraud_model.pkl"
)

# --------------------------------------------------
# Upload CSV
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

# --------------------------------------------------
# Main Application
# --------------------------------------------------

if uploaded_file is not None:

    # Load uploaded data
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(data, use_container_width=True)

    # --------------------------------------------------
    # Load Model
    # --------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found at:\n{MODEL_PATH}"
        )
        st.stop()

    model = joblib.load(MODEL_PATH)

    # --------------------------------------------------
    # Prepare Features
    # --------------------------------------------------

    # IMPORTANT:
    # These columns must match the features used during training.
    #
    # Example:
    # feature_columns = ["Time", "V1", "V2", ..., "V28", "Amount"]
    #
    # Replace this with your actual training features.

    feature_columns = [
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount"
    ]

    missing_columns = [
        col for col in feature_columns
        if col not in data.columns
    ]

    if missing_columns:
        st.error(
            "The uploaded CSV is missing the following "
            "required features:"
        )
        st.write(missing_columns)
        st.stop()

    X = data[feature_columns]

    # --------------------------------------------------
    # Predict Fraud Probability
    # --------------------------------------------------

    probabilities = model.predict_proba(X)[:, 1]

    data["Fraud Probability"] = probabilities

    # --------------------------------------------------
    # Calculate Risk Score
    # --------------------------------------------------

    data["Risk Score"] = probabilities * 100

    # --------------------------------------------------
    # Assign Risk Level
    # --------------------------------------------------

    data["Risk Level"] = data["Risk Score"].apply(
        lambda x: (
            "LOW"
            if x < 30
            else "MEDIUM"
            if x < 70
            else "HIGH"
        )
    )

    # --------------------------------------------------
    # Display Predictions
    # --------------------------------------------------

    st.subheader("Fraud Detection Results")

    st.dataframe(
        data,
        use_container_width=True
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        len(data)
    )

    col2.metric(
        "High-Risk Transactions",
        (data["Risk Level"] == "HIGH").sum()
    )

    col3.metric(
        "Average Risk Score",
        round(data["Risk Score"].mean(), 2)
    )

    col4.metric(
        "Maximum Risk Score",
        round(data["Risk Score"].max(), 2)
    )

    # --------------------------------------------------
    # Risk Distribution Chart
    # --------------------------------------------------

    st.subheader("Risk Level Distribution")

    fig, ax = plt.subplots(figsize=(4,4))

    data["Risk Level"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Risk Level Distribution")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Transactions")

    st.pyplot(fig)

    # --------------------------------------------------
    # Download Predictions
    # --------------------------------------------------

    csv = data.to_csv(index=False)

    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

else:

    st.info(
        "Please upload a transaction CSV file to start "
        "fraud detection."
    )
