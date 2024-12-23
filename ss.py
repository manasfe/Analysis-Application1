import streamlit as st
import random
import json
from pathlib import Path

# Configure page with a festive theme
st.set_page_config(
    page_title="Secret Santa 2024 🎅",
    page_icon="🎅",
    layout="wide"
)

# Predefined users with passwords
USERS = {
    'atulGhag': 'Ag#Dec2024$',
    'apurvaDhamane': 'Ad#Dec2024$',
    'manasDesai': 'Md#Dec2024$',
    'manishGanekar': 'Mg#Dec2024$',
    'tanayaPatole': 'Tp#Dec2024$',
    'maitreyaJadhav': 'Mj#Dec2024$'
}

# Custom CSS for a beautiful festive look
st.markdown("""
    <style>
        /* Main theme colors */
        :root {
            --primary-color: #c41e3a;
            --secondary-color: #0e5f3f;
            --background-color: #f8f9fa;
        }
        
        /* Header styling */
        h1, h2, h3 {
            color: var(--primary-color) !important;
            font-family: 'Arial', sans-serif;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: var(--primary-color);
            color: white;
            border-radius: 20px;
            padding: 10px 25px;
            border: none;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #a01830;
            transform: translateY(-2px);
        }
        
        /* Card styling */
        .css-1r6slb0 {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Success message styling */
        .stSuccess {
            background-color: var(--secondary-color);
            padding: 20px;
            border-radius: 10px;
            color: white;
        }
        
        /* Festive decorations */
        .decoration {
            text-align: center;
            font-size: 24px;
            margin: 20px 0;
        }
    </style>
""", unsafe_allow_html=True)

def load_assignments():
    """Load existing assignments from file"""
    try:
        with open('assignments.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_assignments(assignments):
    """Save assignments to file"""
    with open('assignments.json', 'w') as f:
        json.dump(assignments, f)

def generate_assignments(participants):
    """Generate random Secret Santa assignments"""
    if not hasattr(st.session_state, 'assignments'):
        st.session_state.assignments = load_assignments()
    
    # Only generate new assignments for participants who don't have one
    unassigned = [p for p in participants if p not in st.session_state.assignments]
    available_recipients = [p for p in participants if p not in st.session_state.assignments.values()]
    
    for santa in unassigned:
        possible_recipients = [r for r in available_recipients if r != santa]
        if possible_recipients:
            recipient = random.choice(possible_recipients)
            st.session_state.assignments[santa] = recipient
            available_recipients.remove(recipient)
    
    save_assignments(st.session_state.assignments)

def verify_login(username, password):
    """Verify user login credentials"""
    return username in USERS and USERS[username] == password

def main():
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.assignments = load_assignments()

    # Display festive header
    st.markdown("<h1 style='text-align: center;'>🎄 Secret Santa 2024 🎅</h1>", unsafe_allow_html=True)
    st.markdown("<div class='decoration'>❄️ 🎁 ⛄ 🎄 ❄️</div>", unsafe_allow_html=True)
    
    if not st.session_state.logged_in:
        # Login form
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("### 🔐 Login to Your Secret Santa Account")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login 🎄")
                
                if submitted:
                    if verify_login(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        # Generate assignments if needed
                        generate_assignments(list(USERS.keys()))
                        st.success("Ho Ho Ho! Login successful! 🎅")
                        st.experimental_rerun()
                    else:
                        st.error("Invalid username or password! 🎅❌")
    
    else:
        # Logged in view
        col1, col2 = st.columns([3,1])
        
        with col2:
            st.markdown("### 🎄 Welcome!")
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.experimental_rerun()
        
        with col1:
            st.markdown("### 🎁 Your Secret Santa Assignment")
            if st.session_state.username in st.session_state.assignments:
                recipient = st.session_state.assignments[st.session_state.username]
                st.success(f"You are the Secret Santa for: **{recipient}** 🎁")
                
                st.markdown("""
                ### 🎄 Gift Exchange Guidelines:
                * 🎁 Budget: ₹500-1000
                * 📅 Exchange Date: December 25, 2024
                * 🤫 Keep it a secret until the reveal!
                * 🎯 Consider your recipient's interests
                * 📝 Include a festive card with your gift
                """)
                
                # Display festive message
                st.markdown("""
                <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;'>
                    <h3 style='color: #c41e3a; text-align: center;'>🎄 Spread the Holiday Cheer! 🎅</h3>
                    <p style='text-align: center;'>Remember, the joy of Secret Santa is in the giving and the surprise!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("🎄 Assignment pending... Check back soon!")

if __name__ == "__main__":
    main()
