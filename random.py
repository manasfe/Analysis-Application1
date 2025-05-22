import streamlit as st
import whisper
from textblob import TextBlob
from pydub import AudioSegment
import tempfile
import os

# -------------------------------
# Login credentials
USERNAME = "media@firsteconomy"
PASSWORD = "Pixel_098"

# -------------------------------
# Helper Functions

def check_login(user, pwd):
    return user == USERNAME and pwd == PASSWORD

@st.cache_resource
def load_model():
    return whisper.load_model("base")

def convert_to_wav(audio_file):
    audio = AudioSegment.from_file(audio_file)
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_wav.name, format="wav")
    return temp_wav.name

def transcribe_audio(file_path):
    model = load_model()
    result = model.transcribe(file_path)
    return result["text"]

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, polarity, subjectivity

# -------------------------------
# Streamlit App

st.set_page_config(page_title="Audio Transcription & Sentiment", layout="centered")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- Login Page ---
if not st.session_state.logged_in:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password.")

# --- Main App ---
if st.session_state.logged_in:
    st.title("🎧 Audio Transcription & Sentiment Analysis")

    uploaded_file = st.file_uploader("Upload an audio file (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

    if uploaded_file:
        with st.spinner("Processing audio..."):
            try:
                wav_path = convert_to_wav(uploaded_file)
                transcript = transcribe_audio(wav_path)

                st.subheader("📝 Transcription")
                st.write(transcript)

                sentiment, polarity, subjectivity = analyze_sentiment(transcript)

                st.subheader("💬 Sentiment Analysis")
                st.write(f"**Sentiment:** {sentiment}")
                st.write(f"**Polarity:** {polarity:.2f}")
                st.write(f"**Subjectivity:** {subjectivity:.2f}")

                os.remove(wav_path)

            except Exception as e:
                st.error(f"An error occurred: {e}")
