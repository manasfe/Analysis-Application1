import streamlit as st
import random
from hashlib import sha256

# Predefined users with non-obvious passwords
USERS = {
    'atulGhag': 'Ag#Dec2024$',
    'apurvaDhamane': 'Ad#Dec2024$',
    'manasDesai': 'Md#Dec2024$',
    'manishGanekar': 'Mg#Dec2024$',
    'tanayaPatole': 'Tp#Dec2024$',
    'maitreyaJadhav': 'Mj#Dec2024$',
    'admin': 'SS@Admin2024#'
}

# Store assignments in session state
if 'assignments' not in st.session_state:
    st.session_state.assignments = {}

# Configure page with a festive theme
st.set_page_config(
    page_title="Secret Santa 2024 🎅",
    page_icon="🎅",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for a festive look
st.markdown("""
    <style>
        .stButton>button {
            background-color: #c41e3a;
            color: white;
        }
        .stTitle {
            color: #c41e3a;
        }
        .stSuccess {
            background-color: #0e5f3f;
        }
    </style>
""", unsafe_allow_html=True)

# Authentication
def verify_login(username, password):
    return username in USERS and USERS[username] == password

def is_admin(username):
    return username == 'admin'

# Assignment logic with verification
def generate_assignments():
    participants = list(USERS.keys())
    participants.remove('admin')
    
    valid_assignment = False
    max_attempts = 100
    
    while not valid_assignment and max_attempts > 0:
        recipients = participants.copy()
        random.shuffle(recipients)
        assignments = {}
        valid_assignment = True
        
        for santa in participants:
            possible_recipients = [r for r in recipients if r != santa]
            if not possible_recipients:
                valid_assignment = False
                break
            
            recipient = random.choice(possible_recipients)
            assignments[santa] = recipient
            recipients.remove(recipient)
        
        max_attempts -= 1
        if valid_assignment:
            # Verify no self-assignments
            for santa, recipient in assignments.items():
                if santa == recipient:
                    valid_assignment = False
                    break
            if valid_assignment:
                return assignments
    
    return None

# Main app
def main():
    st.title("🎅 Secret Santa 2024")
    st.markdown("---")
    
    # Login section
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
    
    if not st.session_state.logged_in:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.subheader("🔐 Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login 🎄"):
                if verify_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("Ho Ho Ho! Login successful! 🎅")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password! 🎅❌")
    
    else:
        # Logout button in sidebar
        with st.sidebar:
            st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/examples/streamlit_app_demo/example-data/snow.jpg")
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.experimental_rerun()
        
        # Admin view
        if is_admin(st.session_state.username):
            st.subheader("🎅 Admin Dashboard")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎲 Generate New Assignments"):
                    assignments = generate_assignments()
                    if assignments:
                        st.session_state.assignments = assignments
                        st.success("🎉 New Secret Santa assignments generated!")
                    else:
                        st.error("❌ Failed to generate valid assignments. Please try again.")
            
            with col2:
                if st.button("👥 Show All Participants"):
                    st.subheader("Participants List:")
                    participants = [user for user in USERS.keys() if user != 'admin']
                    for participant in participants:
                        st.write(f"🎁 {participant}")
            
            if st.session_state.assignments:
                st.subheader("Current Assignments")
                for santa, recipient in st.session_state.assignments.items():
                    st.write(f"🎅 {santa} → 🎁 {recipient}")
            else:
                st.info("🎄 No assignments generated yet.")
        
        # Participant view
        else:
            st.subheader(f"Welcome, {st.session_state.username}! 🎄")
            
            if st.session_state.assignments:
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
                else:
                    st.warning("🎅 You haven't been assigned as Secret Santa yet.")
            else:
                st.info("🎄 Assignments haven't been generated yet. Check back later!")

if __name__ == "__main__":
    main()
