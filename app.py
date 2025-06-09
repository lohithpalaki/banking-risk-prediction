import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Dummy credentials for login
auth_users = {"Risk_User": "Banking123"}

# Session state to manage login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Login"

# Sidebar navigation and logout
if st.session_state.logged_in:
    st.sidebar.title("Navigation")
    st.session_state.page = st.sidebar.selectbox(
        "Go to", 
        ["Predict One", "Predict from CSV"], 
        index=["Predict One", "Predict from CSV"].index(st.session_state.page)
        if st.session_state.page in ["Predict One", "Predict from CSV"] else 0
    )
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "Login"
        st.experimental_rerun()
else:
    st.session_state.page = "Login"

# Page 1: Login
def login_page():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in auth_users and auth_users[username] == password:
            st.session_state.logged_in = True
            st.session_state.page = "Predict One"
            st.success("Login successful! Redirecting to prediction page...")
        else:
            st.error("Invalid credentials")

# Page 2: Single Prediction
def single_prediction_page():
    st.title("Predict Risk Category")

    total_spent = st.number_input("Total Spent", 0.0, 1000000000.0, 5000.0)
    loyalty_points_earned = st.number_input("Loyalty Points Earned", 0, 100000, 250)
    referral_count = st.number_input("Referral Count", 0, 1000, 5)
    cashback_received = st.number_input("Cashback Received", 0.0, 100000.0, 300.0)
    customer_satisfaction_score = st.slider("Customer Satisfaction Score", 0.0, 10.0, 7.5)

    if st.button("Predict"):
        model = joblib.load("lr_model.pkl")
        input_data = np.array([[total_spent, loyalty_points_earned, referral_count, cashback_received, customer_satisfaction_score]])
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Risk Category: {prediction}")

# Page 3: Batch Prediction
def batch_prediction_page():
    st.title("📂 Predict Risk Category from CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:", df.head())

        required_cols = ['Total_Spent', 'Loyalty_Points_Earned', 'Referral_Count', 'Cashback_Received', 'Customer_Satisfaction_Score']
        if all(col in df.columns for col in required_cols):
            model = joblib.load("lr_model.pkl")
            predictions = model.predict(df[required_cols])
            df['Predicted_Risk'] = predictions
            st.write("Predictions:", df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predicted_risk.csv", "text/csv")
        else:
            st.error(f"CSV must contain columns: {', '.join(required_cols)}")

# Router
if st.session_state.page == "Login":
    login_page()
elif st.session_state.page == "Predict One":
    single_prediction_page()
elif st.session_state.page == "Predict from CSV":
    batch_prediction_page()
