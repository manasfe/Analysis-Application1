# Sentiment Timeline Chart (Enhanced) with beautiful styling
                                if len(sentiment_result['sentence_sentiments']) > 1:
                                    st.markdown("""
                                    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                                               padding: 15px; border-radius: 10px; margin: 20px 0;'>
                                        <h2 style='color: white; margin: 0; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                                            📈 Sentiment Journey Throughout Speech
                                        </h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Create more detailed timeline
                                    sentence_nums = list(range(1, len(sentiment_result['sentence_sentiments']) + 1))
                                    sentiments = sentiment_result['sentence_sentiments']
                                    
                                    # Create the enhanced timeline chart
                                    fig = go.Figure()
                                    
                                    # Main sentiment line with gradient fill
                                    fig.add_trace(go.Scatter(
                                        x=sentence_nums,
                                        y=sentiments,
                                        mode='lines+markers',
                                        name='Sentiment Score',
                                        line=dict(color='#667eea', width=4),
                                        marker=dict(
                                            size=10,
                                            color=sentiments,
                                            colorscale='RdYlGn',
                                            showscale=True,
                                            colorbar=dict(title="Sentiment", titleside="right")
                                        ),
                                        fill='tonexty',
                                        fillcolor='rgba(102, 126, 234, 0.1)'
                                    ))
                                    
                                    # Add trend line if enough data
                                    if len(sentiments) > 2:
                                        z = np.polyfit(sentence_nums, sentiments, 1)
                                        p = np.poly1d(z)
                                        fig.add_trace(go.Scatter(
                                            x=sentence_nums,
                                            y=p(sentence_nums),
                                            mode='lines',
                                            name='Trend Line',
                                            line=dict(color='#ff6b6b', width=3, dash='dash'),
                                            opacity=0.8
                                        ))
                                    
                                    # Add horizontal reference lines with better styling
                                    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=2,
                                                 annotation_text="Neutral", annotation_position="bottom right")
                                    fig.add_hline(y=0.5, line_dash="dot", line_color="#4caf50", opacity=0.6, line_width=2,
                                                 annotation_text="Positive Zone", annotation_position="top right")
                                    fig.add_hline(y=-0.5, line_dash="dot", line_color="#f44336", opacity=0.6, line_width=2,
                                                 annotation_text="Negative Zone", annotation_position="bottom right")
                                    
                                    fig.update_layout(
                                        title=dict(
                                            text="Sentiment Evolution Across Sentences",
                                            font=dict(size=20, color='#333'),
                                            x=0.5
                                        ),
                                        xaxis_title="Sentence Number",
                                        yaxis_title="Sentiment Score",
                                        height=500,
                                        showlegend=True,
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(248,249,250,0.8)',
                                        font=dict(family="Arial", size=12),
                                        legend=dict(
                                            bgcolor="rgba(255,255,255,0.8)",
                                            bordercolor="rgba(0,0,0,0.2)",
                                            borderwidth=1
                                        )
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Enhanced sentiment trend analysis with beautiful cards
                                    if len(sentiments) >= 3:
                                        trend_start = sum(sentiments[:len(sentiments)//3]) / len(sentiments[:len(sentiments)//3])
                                        trend_middle = sum(sentiments[len(sentiments)//3:2*len(sentiments)//3]) / len(sentiments[len(sentiments)//3:2import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import io
import tempfile
import os
from textblob import TextBlob
import pandas as pd
import re
from collections import Counter
import nltk
import numpy as np

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data for TextBlob"""
    try:
        # Download required corpora
        nltk.download('punkt', quiet=True)
        nltk.download('brown', quiet=True)
        nltk.download('movie_reviews', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")
        return False

# Initialize NLTK data
download_nltk_data()

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
    """Convert audio file to WAV format for speech recognition with better processing"""
    try:
        # Read the uploaded file
        audio_data = audio_file.read()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_filename = temp_file.name
            
        # Convert audio to WAV format using pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_data))
        
        # Enhanced audio processing for better recognition
        # Convert to mono
        audio = audio.set_channels(1)
        
        # Set sample rate (16kHz is optimal for speech recognition)
        audio = audio.set_frame_rate(16000)
        
        # Normalize audio levels
        audio = audio.normalize()
        
        # Apply high-pass filter to reduce low-frequency noise
        if len(audio) > 0:
            audio = audio.high_pass_filter(300)
        
        # Export with higher quality settings
        audio.export(temp_filename, format="wav", parameters=["-q:a", "0"])
        
        return temp_filename
    except Exception as e:
        st.error(f"Error converting audio file: {str(e)}")
        return None

def split_audio_for_processing(audio_file_path, chunk_length_ms=30000):
    """Split long audio files into smaller chunks for better processing"""
    try:
        audio = AudioSegment.from_wav(audio_file_path)
        chunks = []
        
        # If audio is shorter than chunk_length, return as single chunk
        if len(audio) <= chunk_length_ms:
            return [audio_file_path]
        
        # Split audio into chunks
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            
            # Create temporary file for chunk
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk_{i//chunk_length_ms}.wav") as temp_file:
                chunk_filename = temp_file.name
                chunk.export(chunk_filename, format="wav")
                chunks.append(chunk_filename)
        
        return chunks
    except Exception as e:
        st.error(f"Error splitting audio: {str(e)}")
        return [audio_file_path]

def transcribe_audio_chunk(audio_file_path, recognizer, chunk_index=0):
    """Transcribe a single audio chunk with multiple recognition attempts"""
    
    try:
        with sr.AudioFile(audio_file_path) as source:
            # Adjust for ambient noise with longer duration for better results
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            # Record the audio data
            audio_data = recognizer.record(source)
        
        # Try multiple recognition engines in order of preference
        recognition_methods = [
            ("Google", lambda: recognizer.recognize_google(audio_data, language='en-US')),
            ("Google (alternative)", lambda: recognizer.recognize_google(audio_data, language='en-IN')),
            ("Sphinx (offline)", lambda: recognizer.recognize_sphinx(audio_data)),
        ]
        
        for method_name, method_func in recognition_methods:
            try:
                text = method_func()
                if text and len(text.strip()) > 0:
                    return text, f"Success using {method_name}"
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                st.warning(f"{method_name} service error: {e}")
                continue
            except Exception as e:
                st.warning(f"{method_name} error: {e}")
                continue
        
        return "", f"Could not understand audio in chunk {chunk_index + 1}"
        
    except Exception as e:
        return "", f"Error processing audio chunk {chunk_index + 1}: {str(e)}"

def transcribe_audio(audio_file_path):
    """Enhanced transcription with chunking and multiple recognition engines"""
    recognizer = sr.Recognizer()
    
    # Optimize recognizer settings
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.8
    
    try:
        # Get audio duration to decide on chunking strategy
        audio = AudioSegment.from_wav(audio_file_path)
        duration_seconds = len(audio) / 1000.0
        
        st.info(f"Audio duration: {duration_seconds:.1f} seconds")
        
        # Split audio into chunks for better processing
        chunk_length_ms = 30000  # 30 seconds per chunk
        audio_chunks = split_audio_for_processing(audio_file_path, chunk_length_ms)
        
        # Process each chunk
        full_transcript = []
        successful_chunks = 0
        
        for i, chunk_path in enumerate(audio_chunks):
            chunk_text, chunk_status = transcribe_audio_chunk(chunk_path, recognizer, i)
            
            if "Success" in chunk_status and chunk_text.strip():
                full_transcript.append(chunk_text.strip())
                successful_chunks += 1
            
            # Clean up chunk file if it's not the original
            if chunk_path != audio_file_path:
                try:
                    os.unlink(chunk_path)
                except:
                    pass
        
        # Combine all successful transcriptions
        combined_transcript = " ".join(full_transcript)
        
        if combined_transcript.strip():
            success_rate = (successful_chunks / len(audio_chunks)) * 100
            final_status = f"Success - {successful_chunks}/{len(audio_chunks)} chunks processed ({success_rate:.1f}% success rate)"
            
            # Post-process the transcript
            combined_transcript = post_process_transcript(combined_transcript)
            
            return combined_transcript, final_status
        else:
            return "", f"Failed to transcribe audio - 0/{len(audio_chunks)} chunks successful"
            
    except Exception as e:
        return "", f"Error in transcription process: {str(e)}"

def post_process_transcript(text):
    """Post-process the transcript to improve readability and add punctuation"""
    if not text:
        return text
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Fix common speech recognition errors
    corrections = {
        r'\b([a-z])\s+([a-z])\s+([a-z])\b': r'\1\2\3',  # Fix letter spacing
        r'\bur\b': 'your',
        r'\bu\b': 'you',
        r'\bwud\b': 'would',
        r'\bcud\b': 'could',
        r'\bshud\b': 'should',
        r'\bteh\b': 'the',
        r'\band and\b': 'and',
        r'\bthe the\b': 'the',
    }
    
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Enhanced punctuation addition
    # Split text into words for better punctuation placement
    words = text.split()
    if not words:
        return text
    
    # Add punctuation based on speech patterns and context
    processed_words = []
    for i, word in enumerate(words):
        processed_words.append(word)
        
        # Add periods after certain ending patterns
        if i < len(words) - 1:
            next_word = words[i + 1].lower()
            current_word = word.lower()
            
            # Add period before transition words/phrases
            if next_word in ['so', 'then', 'now', 'well', 'actually', 'basically', 'anyway', 'however', 'therefore', 'furthermore', 'moreover', 'additionally', 'meanwhile', 'consequently']:
                if not word.endswith(('.', '!', '?', ',')):
                    processed_words[-1] = word + '.'
            
            # Add period after conclusion words
            elif current_word in ['done', 'finished', 'complete', 'end', 'concluded', 'final', 'lastly', 'finally']:
                if not word.endswith(('.', '!', '?')):
                    processed_words[-1] = word + '.'
            
            # Add comma before connecting words
            elif next_word in ['and', 'but', 'or', 'yet', 'because', 'since', 'while', 'although', 'though', 'whereas']:
                if not word.endswith(('.', '!', '?', ',')):
                    processed_words[-1] = word + ','
            
            # Add question mark for question patterns
            elif current_word in ['who', 'what', 'when', 'where', 'why', 'how', 'can', 'could', 'would', 'should', 'will', 'do', 'does', 'did', 'is', 'are', 'was', 'were'] and i < 3:
                # This might be a question at the beginning
                pass
            
    # Join processed words
    text = ' '.join(processed_words)
    
    # Add question marks for question patterns
    text = re.sub(r'\b(who|what|when|where|why|how|can|could|would|should|will|do|does|did|is|are|was|were)\b([^.!?]*?)(?=\s+(so|then|now|well|and|but|\.|$))', r'\1\2?', text, flags=re.IGNORECASE)
    
    # Add periods at natural sentence breaks (longer pauses in speech)
    # Look for patterns that indicate sentence endings
    text = re.sub(r'\b(okay|alright|right|yes|no|sure|exactly|absolutely|definitely|certainly|obviously|clearly|basically|essentially|actually|really|truly|indeed)\b(?=\s+[A-Z])', r'\1.', text, flags=re.IGNORECASE)
    
    # Ensure proper sentence structure - capitalize after periods
    sentences = re.split(r'([.!?]+)', text)
    processed_sentences = []
    
    for i, sentence in enumerate(sentences):
        if i % 2 == 0 and sentence.strip():  # Text parts (not punctuation)
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
        processed_sentences.append(sentence)
    
    result = ''.join(processed_sentences)
    
    # Final cleanup - ensure the text ends with proper punctuation
    if result and not result.endswith(('.', '!', '?')):
        result += '.'
    
    return result



def analyze_emotional_keywords(text):
    """Analyze emotional keywords in the text"""
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 
                     'happy', 'pleased', 'satisfied', 'perfect', 'awesome', 'brilliant', 'outstanding',
                     'beautiful', 'success', 'win', 'gain', 'benefit', 'advantage', 'positive', 'yes']
    
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad', 'angry', 
                     'disappointed', 'frustrated', 'annoyed', 'upset', 'worried', 'concerned',
                     'problem', 'issue', 'fail', 'loss', 'disadvantage', 'negative', 'no', 'never']
    
    neutral_words = ['okay', 'fine', 'normal', 'standard', 'regular', 'usual', 'typical', 'average']
    
    words = text.lower().split()
    
    pos_count = sum(1 for word in words if any(pw in word for pw in positive_words))
    neg_count = sum(1 for word in words if any(nw in word for nw in negative_words))
    neu_count = sum(1 for word in words if any(nw in word for nw in neutral_words))
    
    total_emotional = pos_count + neg_count + neu_count
    
    return {
        'positive_words': pos_count,
        'negative_words': neg_count,
        'neutral_words': neu_count,
        'total_emotional_words': total_emotional,
        'emotional_density': total_emotional / len(words) if words else 0
    }



def calculate_variance(values):
    """Calculate variance without numpy"""
    if not values or len(values) < 2:
        return 0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance

def analyze_sentiment_detailed(text):
    """Comprehensive sentiment analysis of the text with enhanced error handling"""
    if not text.strip():
        return None
    
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Basic sentiment classification
        if polarity > 0.3:
            sentiment = "Strongly Positive"
            sentiment_emoji = "😊"
        elif polarity > 0.1:
            sentiment = "Moderately Positive"
            sentiment_emoji = "🙂"
        elif polarity > -0.1:
            sentiment = "Neutral"
            sentiment_emoji = "😐"
        elif polarity > -0.3:
            sentiment = "Moderately Negative"
            sentiment_emoji = "😕"
        else:
            sentiment = "Strongly Negative"
            sentiment_emoji = "😞"
        
        # Confidence level
        confidence = abs(polarity)
        if confidence > 0.7:
            confidence_level = "Very High"
        elif confidence > 0.5:
            confidence_level = "High"
        elif confidence > 0.3:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"
        
        # Subjectivity classification
        if subjectivity > 0.7:
            subjectivity_level = "Highly Subjective"
        elif subjectivity > 0.5:
            subjectivity_level = "Moderately Subjective"
        elif subjectivity > 0.3:
            subjectivity_level = "Slightly Subjective"
        else:
            subjectivity_level = "Mostly Objective"
        
        # Additional analyses
        emotional_keywords = analyze_emotional_keywords(text)
        
        # Sentence-level sentiment analysis with error handling
        sentence_sentiments = []
        try:
            for sentence in blob.sentences:
                sent_polarity = sentence.sentiment.polarity
                sentence_sentiments.append(sent_polarity)
        except Exception as e:
            st.info("Using simplified sentence analysis due to missing language data.")
            sentence_sentiments = [polarity]  # Use overall polarity as fallback
        
        sentiment_variance = calculate_variance(sentence_sentiments)
        
        return {
            "overall_sentiment": sentiment,
            "sentiment_emoji": sentiment_emoji,
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "confidence_level": confidence_level,
            "subjectivity_level": subjectivity_level,
            "emotional_keywords": emotional_keywords,
            "sentence_sentiments": sentence_sentiments,
            "sentiment_variance": round(sentiment_variance, 3)
        }
    
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        st.info("Please ensure all required packages are installed and NLTK data is downloaded.")
        return None

def main_app():
    """Main application after login"""
    st.title("🎤 Enhanced Audio Transcription & Sentiment Analysis")
    st.markdown("---")
    
    # Logout button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
    
    st.markdown("### Upload an audio file for transcription and sentiment analysis")
    st.info("Supported formats: WAV, MP3, MP4, M4A, FLAC, OGG | Recommended: Clear audio with minimal background noise")
    
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
        
        # Audio quality tips
        if file_size > 10:
            st.warning("⚠️ Large file detected. Processing may take longer. For best results, use clear audio with minimal background noise.")
        
        # Process button
        if st.button("🚀 Process Audio", type="primary"):
            with st.spinner("Converting and optimizing audio file..."):
                # Convert audio to WAV
                wav_file_path = convert_audio_to_wav(uploaded_file)
                
                if wav_file_path:
                    # Transcribe audio with enhanced method
                    with st.spinner("Transcribing audio..."):
                        transcript, status = transcribe_audio(wav_file_path)
                    
                    # Clean up temporary file
                    try:
                        os.unlink(wav_file_path)
                    except:
                        pass
                    
                    if "Success" in status and transcript and len(transcript.strip()) > 0:
                        st.markdown("---")
                        st.success(f"✅ Transcription completed: {status}")
                        
                        # Display word count
                        word_count = len(transcript.split())
                        st.info(f"📊 Total words transcribed: **{word_count}**")
                        
                        # Detailed sentiment analysis FIRST
                        if word_count >= 3:  # Only analyze if we have sufficient text
                            with st.spinner("Performing comprehensive sentiment analysis..."):
                                sentiment_result = analyze_sentiment_detailed(transcript)
                            
                            if sentiment_result:
                                st.markdown("---")
                                st.markdown("""
                                <div style='text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                                           padding: 20px; border-radius: 15px; margin-bottom: 30px;'>
                                    <h1 style='color: white; margin: 0; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                                        📊 Comprehensive Sentiment Analysis
                                    </h1>
                                    <p style='color: #f0f0f0; margin: 10px 0 0 0; font-size: 1.2em;'>
                                        Deep dive into the emotional landscape of your audio
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Main sentiment display with enhanced styling
                                st.markdown("### 🎯 Core Sentiment Metrics")
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                                               padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                                        <h3 style='margin: 0; color: #333;'>Overall Sentiment</h3>
                                        <h1 style='margin: 10px 0; font-size: 2.5em;'>{sentiment_result['sentiment_emoji']}</h1>
                                        <p style='margin: 0; font-weight: bold; color: #444;'>{sentiment_result['overall_sentiment']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    polarity_color = "#28a745" if sentiment_result['polarity'] > 0 else "#dc3545" if sentiment_result['polarity'] < 0 else "#ffc107"
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                                               padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                                        <h3 style='margin: 0; color: #333;'>Polarity Score</h3>
                                        <h1 style='margin: 10px 0; color: {polarity_color}; font-size: 2.2em;'>{sentiment_result['polarity']:.3f}</h1>
                                        <p style='margin: 0; font-size: 0.9em; color: #666;'>Range: -1 to +1</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    subj_color = "#ff6b6b" if sentiment_result['subjectivity'] > 0.7 else "#4ecdc4" if sentiment_result['subjectivity'] < 0.3 else "#45b7d1"
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                                               padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                                        <h3 style='margin: 0; color: #333;'>Subjectivity</h3>
                                        <h1 style='margin: 10px 0; color: {subj_color}; font-size: 2.2em;'>{sentiment_result['subjectivity']:.3f}</h1>
                                        <p style='margin: 0; font-size: 0.9em; color: #666;'>0=Objective, 1=Subjective</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col4:
                                    conf_color = "#28a745" if sentiment_result['confidence_level'] in ["High", "Very High"] else "#ffc107" if sentiment_result['confidence_level'] == "Moderate" else "#dc3545"
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%); 
                                               padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                                        <h3 style='margin: 0; color: #333;'>Confidence</h3>
                                        <h1 style='margin: 10px 0; color: {conf_color}; font-size: 1.8em;'>{sentiment_result['confidence_level']}</h1>
                                        <p style='margin: 0; font-size: 0.9em; color: #666;'>Analysis Certainty</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                                
                                # Visual Charts Section with enhanced styling
                                st.markdown("""
                                <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                                           padding: 15px; border-radius: 10px; margin: 20px 0;'>
                                    <h2 style='color: white; margin: 0; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                                        📈 Interactive Visual Analysis
                                    </h2>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Create sentiment distribution chart
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("#### 🎯 Sentiment Polarity Gauge")
                                    # Create a gauge-like visualization with enhanced styling
                                    import plotly.graph_objects as go
                                    
                                    fig = go.Figure(go.Indicator(
                                        mode = "gauge+number+delta",
                                        value = sentiment_result['polarity'],
                                        domain = {'x': [0, 1], 'y': [0, 1]},
                                        title = {'text': "Sentiment Polarity", 'font': {'size': 18, 'color': '#333'}},
                                        delta = {'reference': 0, 'font': {'size': 16}},
                                        number = {'font': {'size': 24, 'color': '#333'}},
                                        gauge = {
                                            'axis': {'range': [-1, 1], 'tickfont': {'size': 14}},
                                            'bar': {'color': "#667eea", 'thickness': 0.8},
                                            'steps': [
                                                {'range': [-1, -0.5], 'color': "#ff6b6b"},
                                                {'range': [-0.5, -0.1], 'color': "#ffa726"},
                                                {'range': [-0.1, 0.1], 'color': "#ffeb3b"},
                                                {'range': [0.1, 0.5], 'color': "#66bb6a"},
                                                {'range': [0.5, 1], 'color': "#4caf50"}
                                            ],
                                            'threshold': {
                                                'line': {'color': "#333", 'width': 4},
                                                'thickness': 0.75,
                                                'value': sentiment_result['polarity']
                                            }
                                        }
                                    ))
                                    fig.update_layout(
                                        height=350,
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        font=dict(family="Arial", size=14)
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    st.markdown("#### 🎭 Emotional Word Distribution")
                                    emo = sentiment_result['emotional_keywords']
                                    
                                    # Create enhanced pie chart
                                    labels = ['Positive Words', 'Negative Words', 'Neutral Words']
                                    values = [emo['positive_words'], emo['negative_words'], emo['neutral_words']]
                                    colors = ['#4caf50', '#f44336', '#9e9e9e']
                                    
                                    fig = go.Figure(data=[go.Pie(
                                        labels=labels, 
                                        values=values,
                                        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=3)),
                                        hole=0.4,
                                        textfont=dict(size=14, color='white'),
                                        textinfo='percent+label',
                                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                                    )])
                                    fig.update_layout(
                                        title=dict(text="Distribution of Emotional Words", font=dict(size=16, color='#333')),
                                        height=350,
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        showlegend=True,
                                        legend=dict(font=dict(size=12))
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Subjectivity vs Objectivity Chart with better styling
                                st.markdown("#### 📊 Subjectivity Analysis")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Enhanced Subjectivity gauge
                                    fig = go.Figure(go.Indicator(
                                        mode = "gauge+number",
                                        value = sentiment_result['subjectivity'],
                                        domain = {'x': [0, 1], 'y': [0, 1]},
                                        title = {'text': "Subjectivity Level", 'font': {'size': 18, 'color': '#333'}},
                                        number = {'font': {'size': 24, 'color': '#333'}},
                                        gauge = {
                                            'axis': {'range': [0, 1], 'tickfont': {'size': 14}},
                                            'bar': {'color': "#e91e63", 'thickness': 0.8},
                                            'steps': [
                                                {'range': [0, 0.3], 'color': "#e3f2fd"},
                                                {'range': [0.3, 0.7], 'color': "#fff3e0"},
                                                {'range': [0.7, 1], 'color': "#fce4ec"}
                                            ],
                                        }
                                    ))
                                    fig.update_layout(
                                        height=350,
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    # Enhanced Sentiment intensity radar chart
                                    st.markdown("**🔥 Sentiment Intensity Radar**")
                                    
                                    categories = ['Positive Intensity', 'Negative Intensity', 'Emotional Depth', 'Confidence Level']
                                    values = [
                                        max(0, sentiment_result['polarity']),
                                        abs(min(0, sentiment_result['polarity'])),
                                        sentiment_result['subjectivity'],
                                        abs(sentiment_result['polarity'])
                                    ]
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatterpolar(
                                        r=values,
                                        theta=categories,
                                        fill='toself',
                                        fillcolor='rgba(102, 126, 234, 0.3)',
                                        line=dict(color='rgba(102, 126, 234, 1)', width=3),
                                        marker=dict(size=8, color='rgba(102, 126, 234, 1)'),
                                        name='Intensity'
                                    ))
                                    
                                    fig.update_layout(
                                        polar=dict(
                                            radialaxis=dict(
                                                visible=True,
                                                range=[0, 1],
                                                tickfont=dict(size=12)
                                            ),
                                            angularaxis=dict(tickfont=dict(size=12))
                                        ),
                                        height=350,
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        showlegend=False
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Advanced Metrics with beautiful cards
                                st.markdown("""
                                <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                                           padding: 15px; border-radius: 10px; margin: 20px 0;'>
                                    <h2 style='color: white; margin: 0; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                                        🔬 Advanced Sentiment Metrics
                                    </h2>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Calculate additional metrics
                                total_words = len(transcript.split())
                                emotional_ratio = emo['total_emotional_words'] / total_words if total_words > 0 else 0
                                sentiment_strength = abs(sentiment_result['polarity'])
                                emotional_volatility = sentiment_result['sentiment_variance']
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%); 
                                               padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                                        <h4 style='margin: 0; color: #333;'>Emotional Ratio</h4>
                                        <h2 style='margin: 10px 0; color: #d63384; font-size: 2em;'>{emotional_ratio:.1%}</h2>
                                        <p style='margin: 0; font-size: 0.8em; color: #666;'>Emotionally charged words</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                                               padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                                        <h4 style='margin: 0; color: #333;'>Sentiment Strength</h4>
                                        <h2 style='margin: 10px 0; color: #0dcaf0; font-size: 2em;'>{sentiment_strength:.3f}</h2>
                                        <p style='margin: 0; font-size: 0.8em; color: #666;'>Absolute strength (0-1)</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    volatility_color = "#dc3545" if emotional_volatility > 0.3 else "#28a745" if emotional_volatility < 0.1 else "#ffc107"
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                                               padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                                        <h4 style='margin: 0; color: #333;'>Emotional Volatility</h4>
                                        <h2 style='margin: 10px 0; color: {volatility_color}; font-size: 2em;'>{emotional_volatility:.3f}</h2>
                                        <p style='margin: 0; font-size: 0.8em; color: #666;'>Sentiment variation</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col4:
                                    consistency = 1 - emotional_volatility if emotional_volatility <= 1 else 0
                                    consistency_color = "#28a745" if consistency > 0.7 else "#ffc107" if consistency > 0.4 else "#dc3545"
                                    st.markdown(f"""
                                    <div style='background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%); 
                                               padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                                        <h4 style='margin: 0; color: #333;'>Consistency</h4>
                                        <h2 style='margin: 10px 0; color: {consistency_color}; font-size: 2em;'>{consistency:.3f}</h2>
                                        <p style='margin: 0; font-size: 0.8em; color: #666;'>Sentiment stability</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Sentiment Timeline Chart (Enhanced)
                                if len(sentiment_result['sentence_sentiments']) > 1:
                                    st.markdown("#### 📈 Sentiment Journey Throughout Speech")
                                    
                                    # Create more detailed timeline
                                    sentence_nums = list(range(1, len(sentiment_result['sentence_sentiments']) + 1))
                                    sentiments = sentiment_result['sentence_sentiments']
                                    
                                    # Create the timeline chart with additional features
                                    fig = go.Figure()
                                    
                                    # Main sentiment line
                                    fig.add_trace(go.Scatter(
                                        x=sentence_nums,
                                        y=sentiments,
                                        mode='lines+markers',
                                        name='Sentiment Score',
                                        line=dict(color='blue', width=3),
                                        marker=dict(size=8)
                                    ))
                                    
                                    # Add trend line
                                    if len(sentiments) > 2:
                                        z = np.polyfit(sentence_nums, sentiments, 1)
                                        p = np.poly1d(z)
                                        fig.add_trace(go.Scatter(
                                            x=sentence_nums,
                                            y=p(sentence_nums),
                                            mode='lines',
                                            name='Trend Line',
                                            line=dict(color='red', width=2, dash='dash')
                                        ))
                                    
                                    # Add horizontal reference lines
                                    fig.add_hline(y=0, line_dash="dot", line_color="gray", 
                                                 annotation_text="Neutral", annotation_position="bottom right")
                                    fig.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.5,
                                                 annotation_text="Positive Threshold", annotation_position="top right")
                                    fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.5,
                                                 annotation_text="Negative Threshold", annotation_position="bottom right")
                                    
                                    fig.update_layout(
                                        title="Sentiment Evolution Across Sentences",
                                        xaxis_title="Sentence Number",
                                        yaxis_title="Sentiment Score",
                                        height=400,
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Sentiment trend analysis with more detail
                                    if len(sentiments) >= 3:
                                        trend_start = sum(sentiments[:len(sentiments)//3]) / len(sentiments[:len(sentiments)//3])
                                        trend_middle = sum(sentiments[len(sentiments)//3:2*len(sentiments)//3]) / len(sentiments[len(sentiments)//3:2*len(sentiments)//3])
                                        trend_end = sum(sentiments[-len(sentiments)//3:]) / len(sentiments[-len(sentiments)//3:])
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Opening Sentiment", f"{trend_start:.3f}")
                                        with col2:
                                            st.metric("Middle Sentiment", f"{trend_middle:.3f}")
                                        with col3:
                                            st.metric("Closing Sentiment", f"{trend_end:.3f}")
                                        
                                        if trend_end > trend_start + 0.1:
                                            st.success("📈 **Positive Journey** - Sentiment improves throughout the speech")
                                        elif trend_end < trend_start - 0.1:
                                            st.error("📉 **Declining Trend** - Sentiment becomes more negative over time")
                                        else:
                                            st.info("➡️ **Stable Sentiment** - Consistent emotional tone maintained")
                                
                                # Word Cloud Simulation (Text-based)
                                st.markdown("#### ☁️ Key Emotional Expressions")
                                
                                # Extract and display key emotional phrases
                                positive_phrases = []
                                negative_phrases = []
                                
                                # Simple phrase extraction based on emotional words
                                words = transcript.lower().split()
                                positive_indicators = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like', 'happy', 'pleased']
                                negative_indicators = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'disappointed']
                                
                                for i, word in enumerate(words):
                                    if any(pos in word for pos in positive_indicators):
                                        # Get context around the word
                                        start = max(0, i-2)
                                        end = min(len(words), i+3)
                                        phrase = ' '.join(words[start:end])
                                        positive_phrases.append(phrase)
                                    elif any(neg in word for neg in negative_indicators):
                                        start = max(0, i-2)
                                        end = min(len(words), i+3)
                                        phrase = ' '.join(words[start:end])
                                        negative_phrases.append(phrase)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**🟢 Positive Expressions:**")
                                    if positive_phrases:
                                        for phrase in positive_phrases[:5]:  # Show top 5
                                            st.write(f"• *{phrase.title()}*")
                                    else:
                                        st.write("• No strong positive expressions detected")
                                
                                with col2:
                                    st.markdown("**🔴 Critical Expressions:**")
                                    if negative_phrases:
                                        for phrase in negative_phrases[:5]:  # Show top 5
                                            st.write(f"• *{phrase.title()}*")
                                    else:
                                        st.write("• No strong negative expressions detected")
                                
                                # Detailed Analysis Sections
                                st.markdown("### 🔍 Detailed Analysis")
                                
                                # Sentiment Interpretation
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("#### 📈 Sentiment Interpretation")
                                    
                                    polarity_desc = ""
                                    if sentiment_result["polarity"] > 0.5:
                                        polarity_desc = "**Very positive** - Expresses strong positive emotions"
                                    elif sentiment_result["polarity"] > 0.1:
                                        polarity_desc = "**Positive** - Generally favorable tone"
                                    elif sentiment_result["polarity"] > -0.1:
                                        polarity_desc = "**Neutral** - Balanced or factual tone"
                                    elif sentiment_result["polarity"] > -0.5:
                                        polarity_desc = "**Negative** - Generally unfavorable tone"
                                    else:
                                        polarity_desc = "**Very negative** - Expresses strong negative emotions"
                                    
                                    st.write(f"**Polarity ({sentiment_result['polarity']:.3f}):** {polarity_desc}")
                                    st.write(f"**Subjectivity ({sentiment_result['subjectivity']:.3f}):** {sentiment_result['subjectivity_level']}")
                                    
                                    if sentiment_result['subjectivity'] > 0.5:
                                        st.write("💭 This text contains personal opinions, emotions, or subjective statements.")
                                    else:
                                        st.write("📊 This text is mostly factual and objective.")
                                
                                with col2:
                                    st.markdown("#### 🎭 Emotional Keywords Analysis")
                                    emo = sentiment_result['emotional_keywords']
                                    
                                    st.write(f"**Positive words:** {emo['positive_words']}")
                                    st.write(f"**Negative words:** {emo['negative_words']}")
                                    st.write(f"**Neutral words:** {emo['neutral_words']}")
                                    st.write(f"**Emotional density:** {emo['emotional_density']:.2%}")
                                    
                                    if emo['emotional_density'] > 0.1:
                                        st.write("🔥 High emotional content detected")
                                    elif emo['emotional_density'] > 0.05:
                                        st.write("😊 Moderate emotional content")
                                    else:
                                        st.write("😐 Low emotional content")
                                
                                # Speaking Style Analysis (simplified without linguistic features)
                                st.markdown("#### 🗣️ Speaking Style Analysis")
                                
                                style_notes = []
                                
                                if sentiment_result['sentiment_variance'] > 0.2:
                                    style_notes.append("**Variable emotions** - Sentiment changes throughout the speech")
                                
                                # Simple style analysis based on sentiment only
                                if sentiment_result['polarity'] > 0.5:
                                    style_notes.append("**Positive speaker** - Predominantly positive communication style")
                                elif sentiment_result['polarity'] < -0.5:
                                    style_notes.append("**Critical speaker** - Predominantly negative communication style")
                                else:
                                    style_notes.append("**Balanced speaker** - Neutral communication style")
                                
                                if sentiment_result['subjectivity'] > 0.7:
                                    style_notes.append("**Expressive speaker** - Highly subjective and emotional")
                                elif sentiment_result['subjectivity'] < 0.3:
                                    style_notes.append("**Factual speaker** - Objective and matter-of-fact")
                                
                                if style_notes:
                                    for note in style_notes:
                                        st.write(f"• {note}")
                                else:
                                    st.write("• **Balanced speaker** - Neutral speaking style")
                                
                                # Sentiment Progression
                                if len(sentiment_result['sentence_sentiments']) > 1:
                                    # This section is now handled above in the Visual Analysis section
                                    pass
                                
                                # Overall Assessment
                                st.markdown("#### 🎯 Overall Assessment")
                                
                                assessment = f"""
                                **Communication Style:** {sentiment_result['overall_sentiment']} tone with {sentiment_result['confidence_level'].lower()} confidence.
                                
                                **Emotional Pattern:** The speaker demonstrates {sentiment_result['subjectivity_level'].lower()} expression with 
                                {emo['emotional_density']:.1%} emotional word density.
                                
                                **Key Insights:**
                                • Primary sentiment: {sentiment_result['overall_sentiment']} ({sentiment_result['polarity']:.3f})
                                • Objectivity level: {sentiment_result['subjectivity_level']}
                                • Emotional intensity: {sentiment_result['confidence_level']}
                                • Communication style: {"Expressive" if sentiment_result['subjectivity'] > 0.5 else "Objective"}
                                """
                                
                                st.markdown(assessment)
                            else:
                                st.warning("Could not perform detailed sentiment analysis.")
                        else:
                            st.warning("⚠️ Transcript too short for meaningful sentiment analysis. Need at least 3 words.")
                        
                        # Display transcript AFTER sentiment analysis (ONLY ONCE)
                        st.markdown("---")
                        st.markdown("## 📝 Complete Transcription")
                        
                        # Display full transcript directly
                        st.markdown("### Full Audio Transcript:")
                        
                        # Format transcript with better readability
                        formatted_transcript = transcript.replace('. ', '.\n\n')  # Add line breaks after sentences
                        st.markdown(f"**{formatted_transcript}**")
                        
                        # Additional transcript info
                        st.markdown("---")
                        st.markdown("**Transcript Statistics:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Words", word_count)
                        with col2:
                            sentences = len(re.split(r'[.!?]+', transcript))
                            st.metric("Estimated Sentences", max(1, sentences - 1))
                        with col3:
                            avg_words = word_count / max(1, sentences - 1)
                            st.metric("Avg Words/Sentence", f"{avg_words:.1f}")
                        
                        # Option to copy transcript (REMOVED - no duplicate)
                    
                    else:
                        st.error(f"❌ Transcription failed: {status}")
                        st.markdown("### 🔧 Troubleshooting Tips:")
                        st.markdown("""
                        - **Check audio quality:** Ensure clear speech with minimal background noise
                        - **Audio format:** Try converting to WAV format before uploading
                        - **Speaking pace:** Moderate speaking pace works best
                        - **Language:** Currently optimized for English language
                        - **Internet connection:** Google Speech Recognition requires internet access
                        - **File size:** Very large files may timeout - try shorter clips
                        """)
                        
                        # Option to install offline recognition
                        st.info("💡 **Tip:** For offline processing, ensure pocketsphinx is installed: `pip install pocketsphinx`")
                
                else:
                    st.error("❌ Failed to process the audio file. Please try a different format or check the file.")

def main():
    """Main function to control app flow"""
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
