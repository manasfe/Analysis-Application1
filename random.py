import streamlit as st
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
nltk.download('all')
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

def display_sentiment_analysis(sentiment_result, transcript):
    """Display comprehensive sentiment analysis with improved visual layout using Streamlit charts"""
    st.markdown("---")
    st.markdown("# 📊 Comprehensive Sentiment Analysis")
    st.markdown("")
    
    # Header metrics in a clean layout
    st.markdown("## 🎯 Key Sentiment Indicators")
    col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])
    
    with col1:
        st.markdown("**Overall Sentiment**")
        st.markdown(f"<div style='text-align: center; font-size: 1.5em; color: #1f77b4; font-weight: bold; margin: 5px 0;'>{sentiment_result['overall_sentiment']}</div>", 
                   unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size: 2.5em; margin: 5px 0;'>{sentiment_result['sentiment_emoji']}</div>", 
                   unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Polarity Score**")
        st.markdown(f"<div style='text-align: center; font-size: 1.8em; color: #2e8b57; font-weight: bold;'>{sentiment_result['polarity']:.3f}</div>", 
                   unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size: 0.8em; color: #666;'>Range: -1.0 to +1.0</div>", 
                   unsafe_allow_html=True)
    
    with col3:
        st.markdown("**Subjectivity Score**")
        st.markdown(f"<div style='text-align: center; font-size: 1.8em; color: #ff6b35; font-weight: bold;'>{sentiment_result['subjectivity']:.3f}</div>", 
                   unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size: 0.8em; color: #666;'>Range: 0.0 to 1.0</div>", 
                   unsafe_allow_html=True)
    
    with col4:
        st.markdown("**Confidence Level**")
        st.markdown(f"<div style='text-align: center; font-size: 1.5em; color: #9d4edd; font-weight: bold;'>{sentiment_result['confidence_level']}</div>", 
                   unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size: 0.8em; color: #666;'>Assessment Confidence</div>", 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visual Analysis Section
    st.markdown("## 📈 Visual Sentiment Analysis")
    
    # Row 1: Sentiment Polarity and Emotional Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Sentiment Polarity")
        
        # Create polarity visualization using progress bar and color coding
        polarity_normalized = (sentiment_result['polarity'] + 1) / 2  # Convert -1,1 to 0,1
        
        if sentiment_result['polarity'] > 0.1:
            st.success(f"Positive Sentiment: {sentiment_result['polarity']:.3f}")
            st.progress(polarity_normalized)
        elif sentiment_result['polarity'] < -0.1:
            st.error(f"Negative Sentiment: {sentiment_result['polarity']:.3f}")
            st.progress(polarity_normalized)
        else:
            st.info(f"Neutral Sentiment: {sentiment_result['polarity']:.3f}")
            st.progress(polarity_normalized)
        
        # Polarity scale visualization
        st.markdown("**Sentiment Scale:**")
        st.markdown("🔴 Very Negative (-1.0) ←→ 🟡 Neutral (0.0) ←→ 🟢 Very Positive (+1.0)")
    
    with col2:
        st.markdown("### 🎭 Emotional Word Distribution")
        
        emo = sentiment_result['emotional_keywords']
        
        # Create DataFrame for bar chart
        emotion_data = pd.DataFrame({
            'Emotion Type': ['Positive', 'Negative', 'Neutral'],
            'Word Count': [emo['positive_words'], emo['negative_words'], emo['neutral_words']]
        })
        
        # Display bar chart
        st.bar_chart(emotion_data.set_index('Emotion Type'))
        
        # Display the actual numbers
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Positive", emo['positive_words'])
        with col_b:
            st.metric("Negative", emo['negative_words'])
        with col_c:
            st.metric("Neutral", emo['neutral_words'])
    
    st.markdown("---")
    
    # Row 2: Subjectivity Analysis
    st.markdown("## 🔍 Communication Style Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Subjectivity Level")
        
        # Subjectivity visualization
        subjectivity_percent = sentiment_result['subjectivity'] * 100
        st.progress(sentiment_result['subjectivity'])
        
        if sentiment_result['subjectivity'] > 0.7:
            st.success(f"🎭 **Highly Subjective** ({subjectivity_percent:.1f}%)")
            st.write("Contains personal opinions and emotions")
        elif sentiment_result['subjectivity'] > 0.5:
            st.info(f"💭 **Moderately Subjective** ({subjectivity_percent:.1f}%)")
            st.write("Mix of facts and personal opinions")
        elif sentiment_result['subjectivity'] > 0.3:
            st.info(f"📊 **Slightly Subjective** ({subjectivity_percent:.1f}%)")
            st.write("Mostly factual with some opinions")
        else:
            st.success(f"🎯 **Objective** ({subjectivity_percent:.1f}%)")
            st.write("Factual and neutral presentation")
        
        st.markdown("**Objectivity Scale:**")
        st.markdown("🎯 Objective (0%) ←→ 🎭 Subjective (100%)")
    
    with col2:
        st.markdown("### 📈 Advanced Metrics")
        
        # Calculate additional metrics
        total_words = len(transcript.split())
        emotional_ratio = emo['total_emotional_words'] / total_words if total_words > 0 else 0
        sentiment_strength = abs(sentiment_result['polarity'])
        emotional_volatility = sentiment_result['sentiment_variance']
        consistency = max(0, 1 - emotional_volatility) if emotional_volatility <= 1 else 0
        
        # Display metrics in a clean grid
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Emotional Ratio", f"{emotional_ratio:.1%}", help="% of emotionally charged words")
            st.metric("Emotional Volatility", f"{emotional_volatility:.3f}", help="Sentiment variation")
        with col_b:
            st.metric("Sentiment Strength", f"{sentiment_strength:.3f}", help="Absolute strength (0-1)")
            st.metric("Consistency Score", f"{consistency:.3f}", help="How stable the sentiment is")
    
    st.markdown("---")
    
    # Sentiment Timeline (if multiple sentences)
    if len(sentiment_result['sentence_sentiments']) > 1:
        st.markdown("## 📈 Sentiment Journey Throughout Speech")
        
        # Create DataFrame for line chart
        sentiment_timeline = pd.DataFrame({
            'Sentence': range(1, len(sentiment_result['sentence_sentiments']) + 1),
            'Sentiment Score': sentiment_result['sentence_sentiments']
        })
        
        # Display line chart
        st.line_chart(sentiment_timeline.set_index('Sentence'))
        
        # Sentiment progression analysis
        if len(sentiment_result['sentence_sentiments']) >= 3:
            st.markdown("### 📊 Sentiment Progression Analysis")
            
            sentiments = sentiment_result['sentence_sentiments']
            third = len(sentiments) // 3
            trend_start = sum(sentiments[:third]) / third if third > 0 else sentiments[0]
            trend_middle = sum(sentiments[third:2*third]) / third if third > 0 else sentiments[len(sentiments)//2]
            trend_end = sum(sentiments[-third:]) / third if third > 0 else sentiments[-1]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Opening Sentiment", f"{trend_start:.3f}", help="Average sentiment at the beginning")
            with col2:
                st.metric("Middle Sentiment", f"{trend_middle:.3f}", help="Average sentiment in the middle")
            with col3:
                st.metric("Closing Sentiment", f"{trend_end:.3f}", help="Average sentiment at the end")
            
            # Trend interpretation
            if trend_end > trend_start + 0.1:
                st.success("📈 **Positive Journey** - Sentiment improves throughout the speech")
            elif trend_end < trend_start - 0.1:
                st.error("📉 **Declining Trend** - Sentiment becomes more negative over time")
            else:
                st.info("➡️ **Stable Sentiment** - Consistent emotional tone maintained")
    
    st.markdown("---")
    
    # Key Insights Section
    st.markdown("## 🎯 Key Insights & Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Sentiment Breakdown")
        
        # Polarity interpretation with better formatting
        if sentiment_result["polarity"] > 0.5:
            polarity_desc = "**Very Positive** - Expresses strong positive emotions and optimism"
            color = "#28a745"
        elif sentiment_result["polarity"] > 0.1:
            polarity_desc = "**Positive** - Generally favorable and upbeat tone"
            color = "#20c997"
        elif sentiment_result["polarity"] > -0.1:
            polarity_desc = "**Neutral** - Balanced, factual, or matter-of-fact tone"
            color = "#6c757d"
        elif sentiment_result["polarity"] > -0.5:
            polarity_desc = "**Negative** - Generally unfavorable or critical tone"
            color = "#fd7e14"
        else:
            polarity_desc = "**Very Negative** - Expresses strong negative emotions or criticism"
            color = "#dc3545"
        
        st.markdown(f"<div style='padding: 10px; border-left: 4px solid {color}; background-color: #f8f9fa; margin: 10px 0;'><strong>Sentiment:</strong> {polarity_desc}</div>", 
                   unsafe_allow_html=True)
        st.markdown(f"**Objectivity Level:** {sentiment_result['subjectivity_level']}")
        st.markdown(f"**Emotional Density:** {emo['emotional_density']:.1%} of words are emotionally charged")
        
        # Communication style assessment
        style_notes = []
        if sentiment_result['sentiment_variance'] > 0.2:
            style_notes.append("**Emotionally Dynamic** - Sentiment changes throughout the speech")
        else:
            style_notes.append("**Emotionally Consistent** - Stable sentiment throughout")
        
        if sentiment_result['subjectivity'] > 0.7:
            style_notes.append("**Highly Expressive** - Personal and emotional communication")
        elif sentiment_result['subjectivity'] < 0.3:
            style_notes.append("**Factual Communicator** - Objective and informative style")
        else:
            style_notes.append("**Balanced Communicator** - Mix of facts and personal views")
        
        st.markdown("### 🗣️ Communication Style")
        for note in style_notes:
            st.markdown(f"• {note}")
    
    with col2:
        st.markdown("### 📋 Overall Assessment")
        
        # Create comprehensive assessment with better formatting
        st.markdown(f"🎯 **Primary Tone:** {sentiment_result['overall_sentiment']}")
        st.markdown(f"🔍 **Confidence Level:** {sentiment_result['confidence_level']}")
        st.markdown(f"🗣️ **Expression Style:** {sentiment_result['subjectivity_level']} communication")
        st.markdown(f"💭 **Emotional Content:** {emo['emotional_density']:.1%} emotional word density")
        
        # Add specific insights based on the data
        if emo['positive_words'] > emo['negative_words'] * 2:
            st.success("✨ **Positivity Focus:** Strong emphasis on positive language")
        elif emo['negative_words'] > emo['positive_words'] * 2:
            st.warning("⚠️ **Critical Focus:** Emphasis on challenges or concerns")
        
        if sentiment_result['sentiment_variance'] > 0.3:
            st.info("📊 **Emotional Range:** High variability in emotional expression")
        
        # Final recommendation or insight
        st.markdown("### 💡 Key Takeaway")
        if sentiment_result['polarity'] > 0.3:
            st.success("The speaker demonstrates a positive and optimistic communication style.")
        elif sentiment_result['polarity'] < -0.3:
            st.warning("The speaker expresses concerns or negative sentiments that may need attention.")
        else:
            st.info("The speaker maintains a balanced and neutral communication approach.")
    
    # Emotional Keywords Section
    st.markdown("---")
    st.markdown("## ☁️ Key Emotional Expressions")
    
    # Extract key emotional phrases
    positive_phrases = []
    negative_phrases = []
    
    words = transcript.lower().split()
    positive_indicators = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like', 'happy', 'pleased']
    negative_indicators = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'disappointed']
    
    for i, word in enumerate(words):
        if any(pos in word for pos in positive_indicators):
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
    st.info("💡 **Supported formats:** WAV, MP3, MP4, M4A, FLAC, OGG | **Recommended:** Clear audio with minimal background noise")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg'],
        help="Upload an audio file to get its transcription and sentiment analysis"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ File uploaded: **{uploaded_file.name}**")
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
        st.info(f"📊 File size: **{file_size:.2f} MB**")
        
        # Audio quality tips
        if file_size > 10:
            st.warning("⚠️ **Large file detected.** Processing may take longer. For best results, use clear audio with minimal background noise.")
        
        # Process button
        if st.button("🚀 Process Audio", type="primary"):
            with st.spinner("🔄 Converting and optimizing audio file..."):
                # Convert audio to WAV
                wav_file_path = convert_audio_to_wav(uploaded_file)
                
                if wav_file_path:
                    # Transcribe audio with enhanced method
                    with st.spinner("🎯 Transcribing audio... This may take a moment."):
                        transcript, status = transcribe_audio(wav_file_path)
                    
                    # Clean up temporary file
                    try:
                        os.unlink(wav_file_path)
                    except:
                        pass
                    
                    if "Success" in status and transcript and len(transcript.strip()) > 0:
                        st.success(f"✅ **Transcription completed successfully!** {status}")
                        
                        # Display word count
                        word_count = len(transcript.split())
                        st.info(f"📊 **Total words transcribed:** {word_count}")
                        
                        # SENTIMENT ANALYSIS FIRST (above transcript)
                        if word_count >= 3:  # Only analyze if we have sufficient text
                            with st.spinner("🧠 Performing comprehensive sentiment analysis..."):
                                sentiment_result = analyze_sentiment_detailed(transcript)
                            
                            if sentiment_result:
                                # Display comprehensive sentiment analysis
                                display_sentiment_analysis(sentiment_result, transcript)
                            else:
                                st.warning("⚠️ Could not perform detailed sentiment analysis.")
                        else:
                            st.warning("⚠️ **Transcript too short for meaningful sentiment analysis.** Need at least 3 words.")
                        
                        # TRANSCRIPT DISPLAY (after sentiment analysis)
                        st.markdown("---")
                        st.markdown("# 📝 Complete Audio Transcription")
                        st.markdown("")
                        
                        # Display transcript in a clean, readable format
                        st.markdown("### 🎙️ Your Audio Content:")
                        
                        # Create a clean transcript display
                        formatted_transcript = transcript.replace('. ', '.\n\n')  # Add line breaks after sentences
                        
                        # Display transcript in a nice container
                        st.markdown(
                            f"""
                            <div style="
                                background-color: #f8f9fa;
                                padding: 20px;
                                border-radius: 10px;
                                border-left: 4px solid #007bff;
                                margin: 10px 0;
                                font-family: 'Georgia', serif;
                                line-height: 1.6;
                                font-size: 16px;
                            ">
                                {formatted_transcript}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # Transcript statistics
                        st.markdown("---")
                        st.markdown("### 📈 Transcript Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Words", word_count)
                        
                        with col2:
                            sentences = len(re.split(r'[.!?]+', transcript))
                            sentence_count = max(1, sentences - 1)
                            st.metric("Sentences", sentence_count)
                        
                        with col3:
                            avg_words = word_count / sentence_count
                            st.metric("Avg Words/Sentence", f"{avg_words:.1f}")
                        
                        with col4:
                            # Estimate reading time (average 200 words per minute)
                            reading_time = word_count / 200
                            st.metric("Est. Reading Time", f"{reading_time:.1f} min")
                        
                        # Success message
                        st.success("🎉 **Processing completed successfully!** Your audio has been transcribed and analyzed.")
                    
                    else:
                        st.error(f"❌ **Transcription failed:** {status}")
                        
                        # Enhanced troubleshooting section
                        st.markdown("---")
                        st.markdown("### 🔧 Troubleshooting Guide")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🎵 Audio Quality Tips:**")
                            st.markdown("""
                            - Ensure clear speech with minimal background noise
                            - Use a moderate speaking pace (not too fast/slow)
                            - Record in a quiet environment
                            - Avoid echo or reverberation
                            """)
                        
                        with col2:
                            st.markdown("**🔧 Technical Solutions:**")
                            st.markdown("""
                            - Try converting to WAV format before uploading
                            - Check your internet connection (Google Speech API)
                            - For large files, try splitting into smaller segments
                            - Ensure the audio language is English
                            """)
                        
                        st.info("💡 **Pro Tip:** For offline processing, ensure pocketsphinx is installed: `pip install pocketsphinx`")
                
                else:
                    st.error("❌ **Failed to process the audio file.** Please try a different format or check the file integrity.")

def main():
    """Main function to control app flow"""
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
