import streamlit as st
import pandas as pd
import numpy as np

# --- Login Credentials ---
USERNAME = "media@first economy"
PASSWORD = "Pixel_098"

# --- Login Validation Function ---
def login(user, pwd):
    return user == USERNAME and pwd == PASSWORD

# --- Streamlit App Setup ---
st.set_page_config(page_title="CSV Visualizer", layout="centered")

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- Login Page ---
if not st.session_state.logged_in:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login(username, password):
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password.")

# --- Main App After Login ---
if st.session_state.logged_in:
    st.title("📈 CSV File Visualizer")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.subheader("Sample Data")
            st.dataframe(df.head())

            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                st.warning("No numeric columns found for visualization.")
            else:
                st.subheader("Line Chart")
                st.line_chart(numeric_df)

                st.subheader("Bar Chart")
                st.bar_chart(numeric_df)

                st.subheader("Area Chart")
                st.area_chart(numeric_df)
        except Exception as e:
            st.error(f"Error reading file: {e}")

