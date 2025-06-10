import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained Random Forest model
model = joblib.load("rfc_model.pkl")

# Define the feature columns
feature_cols = [
    'Age', 'Account_Balance', 'Transaction_Amount',
    'Account_Balance_After_Transaction', 'Loan_Amount', 'Interest_Rate',
    'Loan_Term', 'Credit_Limit', 'Credit_Card_Balance',
    'Minimum_Payment_Due', 'Rewards_Points'
]

# Page 1: Manual input prediction
def single_prediction():
    st.title("📊 Predict Customer Risk (Single Input)")

    # Collect user inputs
    input_data = []
    for col in feature_cols:
        val = st.number_input(f"{col.replace('_', ' ')}", value=0.0)
        input_data.append(val)

    if st.button("Predict Risk"):
        input_array = np.array([input_data])
        prediction = model.predict(input_array)[0]
        st.success(f"Predicted Risk: **{prediction}**")

# Page 2: CSV Batch Prediction
def batch_prediction():
    st.title("📂 Predict Customer Risk from CSV")

    uploaded_file = st.file_uploader("Upload CSV file with customer data", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if all(col in df.columns for col in feature_cols):
            predictions = model.predict(df[feature_cols])
            df["Predicted_Risk"] = predictions
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predicted_risk.csv", "text/csv")
        else:
            missing = [col for col in feature_cols if col not in df.columns]
            st.error(f"Missing columns in uploaded CSV: {', '.join(missing)}")

# Simple login mechanism
auth_users = {"RiskPredict": "Secure123"}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in auth_users and auth_users[username] == password:
            st.session_state.logged_in = True
            st.success("✅ Login successful!")
        else:
            st.error("❌ Invalid credentials")

# Routing
def main():
    if not st.session_state.logged_in:
        login()
    else:
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction"])

        if page == "Single Prediction":
            single_prediction()
        elif page == "Batch Prediction":
            batch_prediction()

if __name__ == "__main__":
    main()
