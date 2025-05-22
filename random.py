import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import io
import tempfile
import os
from textblob import TextBlob
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Audio Transcription & Sentiment Analysis",
    page_icon="🎤",
    layout="wide"
)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    """Display login page"""
    st.title("🔐 Login Required")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("Please enter your credentials")
        
        # Login form
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if username == "media@firsteconomy" and password == "Pixel_098":
                    st.session_state.logged_in = True
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")

def convert_audio_to_wav(audio_file):
    """Convert audio file to WAV format for speech recognition"""
    try:
        # Read the uploaded file
        audio_data = audio_file.read()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_filename = temp_file.name
            
        # Convert audio to WAV format using pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_data))
        # Convert to mono and set sample rate to 16kHz for better recognition
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(temp_filename, format="wav")
        
        return temp_filename
    except Exception as e:
        st.error(f"Error converting audio file: {str(e)}")
        return None

def transcribe_audio(audio_file_path):
    """Transcribe audio file using speech recognition"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_file_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Record the audio data
            audio_data = recognizer.record(source)
        
        # Try to recognize speech using Google Speech Recognition
        try:
            text = recognizer.recognize_google(audio_data)
            return text, "Success"
        except sr.UnknownValueError:
            return "", "Could not understand audio"
        except sr.RequestError as e:
            return "", f"Error with speech recognition service: {e}"
            
    except Exception as e:
        return "", f"Error processing audio file: {str(e)}"

def analyze_sentiment(text):
    """Analyze sentiment of the text using TextBlob"""
    if not text.strip():
        return None
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Determine sentiment category
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return {
        "sentiment": sentiment,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "confidence": abs(polarity)
    }

def main_app():
    """Main application after login"""
    st.title("🎤 Audio Transcription & Sentiment Analysis")
    st.markdown("---")
    
    # Logout button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
    
    st.markdown("### Upload an audio file for transcription and sentiment analysis")
    st.info("Supported formats: WAV, MP3, MP4, M4A, FLAC, OGG")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg'],
        help="Upload an audio file to get its transcription and sentiment analysis"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"File uploaded: {uploaded_file.name}")
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
        st.write(f"File size: {file_size:.2f} MB")
        
        # Process button
        if st.button("🚀 Process Audio", type="primary"):
            with st.spinner("Processing audio file..."):
                # Convert audio to WAV
                wav_file_path = convert_audio_to_wav(uploaded_file)
                
                if wav_file_path:
                    # Transcribe audio
                    with st.spinner("Transcribing audio..."):
                        transcript, status = transcribe_audio(wav_file_path)
                    
                    # Clean up temporary file
                    try:
                        os.unlink(wav_file_path)
                    except:
                        pass
                    
                    if status == "Success" and transcript:
                        st.markdown("---")
                        st.markdown("## 📝 Transcription Results")
                        
                        # Display transcript
                        st.subheader("Transcript:")
                        st.write(transcript)
                        
                        # Analyze sentiment
                        with st.spinner("Analyzing sentiment..."):
                            sentiment_result = analyze_sentiment(transcript)
                        
                        if sentiment_result:
                            st.markdown("---")
                            st.markdown("## 📊 Sentiment Analysis")
                            
                            # Create columns for sentiment display
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Sentiment", sentiment_result["sentiment"])
                            
                            with col2:
                                st.metric("Polarity", f"{sentiment_result['polarity']:.3f}")
                            
                            with col3:
                                st.metric("Subjectivity", f"{sentiment_result['subjectivity']:.3f}")
                            
                            # Sentiment explanation
                            st.markdown("### 📈 Analysis Details")
                            
                            # Polarity explanation
                            if sentiment_result["polarity"] > 0:
                                polarity_color = "green"
                                polarity_desc = "positive"
                            elif sentiment_result["polarity"] < 0:
                                polarity_color = "red"
                                polarity_desc = "negative"
                            else:
                                polarity_color = "gray"
                                polarity_desc = "neutral"
                            
                            st.markdown(f"""
                            - **Polarity** ({sentiment_result['polarity']:.3f}): Indicates the emotional tone. 
                              Range: -1 (most negative) to +1 (most positive). 
                              This text is <span style="color:{polarity_color}">**{polarity_desc}**</span>.
                            
                            - **Subjectivity** ({sentiment_result['subjectivity']:.3f}): Indicates objectivity vs subjectivity. 
                              Range: 0 (objective) to 1 (subjective). 
                              This text is **{"more subjective" if sentiment_result['subjectivity'] > 0.5 else "more objective"}**.
                            """, unsafe_allow_html=True)
                            
                            # Create a simple visualization
                            st.markdown("### 📊 Sentiment Visualization")
                            
                            # Create a DataFrame for the chart
                            sentiment_df = pd.DataFrame({
                                'Metric': ['Polarity', 'Subjectivity'],
                                'Value': [sentiment_result['polarity'], sentiment_result['subjectivity']]
                            })
                            
                            st.bar_chart(sentiment_df.set_index('Metric'))
                            
                        else:
                            st.warning("Could not analyze sentiment - no text found.")
                    
                    else:
                        st.error(f"Transcription failed: {status}")
                        st.info("Please try with a different audio file or check if the audio contains clear speech.")
                
                else:
                    st.error("Failed to process the audio file. Please try a different format.")

def main():
    """Main function to control app flow"""
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
