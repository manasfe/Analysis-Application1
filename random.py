import streamlit as st
import pandas as pd
import numpy as np

# --- Login Credentials ---
USERNAME = "media@first economy"
PASSWORD = "Pixel_098"

# --- Function to Check Login ---
def login(user, pwd):
    return user == USERNAME and pwd == PASSWORD

# --- Function to Generate Random Dataset ---
def generate_random_dataset(rows=50, cols=4):
    data = np.random.randn(rows, cols)
    columns = [f"Feature_{i+1}" for i in range(cols)]
    df = pd.DataFrame(data, columns=columns)
    return df

# --- Streamlit App ---
st.set_page_config(page_title="Login & Data Viewer", layout="centered")

# Session state to track login
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

# --- Main App after Login ---
if st.session_state.logged_in:
    st.title("📊 Random Dataset Viewer")

    # Generate random dataset
    df = generate_random_dataset()

    st.subheader("Sample Data")
    st.dataframe(df.head())

    st.subheader("Line Chart")
    st.line_chart(df)

    st.subheader("Bar Chart")
    st.bar_chart(df)

    st.subheader("Area Chart")
    st.area_chart(df)
