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
nltk.download("all")
import math

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
        
        if len(audio_chunks) > 1:
            st.info(f"Processing audio in {len(audio_chunks)} chunks for better accuracy...")
        
        # Process each chunk
        full_transcript = []
        successful_chunks = 0
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, chunk_path in enumerate(audio_chunks):
            status_text.text(f"Processing chunk {i+1} of {len(audio_chunks)}...")
            progress_bar.progress((i + 1) / len(audio_chunks))
            
            chunk_text, chunk_status = transcribe_audio_chunk(chunk_path, recognizer, i)
            
            if "Success" in chunk_status and chunk_text.strip():
                full_transcript.append(chunk_text.strip())
                successful_chunks += 1
                st.success(f"Chunk {i+1}: {chunk_status}")
            else:
                st.warning(f"Chunk {i+1}: {chunk_status}")
            
            # Clean up chunk file if it's not the original
            if chunk_path != audio_file_path:
                try:
                    os.unlink(chunk_path)
                except:
                    pass
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
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
    """Post-process the transcript to improve readability"""
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
    
    # Capitalize first letter of sentences
    sentences = re.split(r'([.!?]+)', text)
    processed_sentences = []
    
    for i, sentence in enumerate(sentences):
        if i % 2 == 0 and sentence.strip():  # Text parts (not punctuation)
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
        processed_sentences.append(sentence)
    
    return ''.join(processed_sentences)



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

def analyze_linguistic_features(text):
    """Analyze linguistic features of the text with error handling"""
    try:
        # Try to use TextBlob for sentence detection
        try:
            blob = TextBlob(text)
            sentence_count = len(blob.sentences)
        except Exception:
            # Fallback: count sentences using simple regex
            sentence_count = len(re.split(r'[.!?]+', text))
            sentence_count = max(1, sentence_count - 1)  # Subtract 1 for empty string at end
        
        # Basic statistics
        word_count = len(text.split())
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Punctuation analysis
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Uppercase analysis (indicating emphasis/shouting)
        uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': round(avg_sentence_length, 1),
            'exclamation_marks': exclamation_count,
            'question_marks': question_count,
            'uppercase_ratio': round(uppercase_ratio, 3)
        }
    except Exception as e:
        st.warning(f"Simplified linguistic analysis used due to: {str(e)}")
        # Fallback analysis
        word_count = len(text.split()) if text else 0
        sentence_count = max(1, len(re.split(r'[.!?]+', text)) - 1) if text else 1
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': round(word_count / sentence_count, 1) if sentence_count > 0 else 0,
            'exclamation_marks': text.count('!') if text else 0,
            'question_marks': text.count('?') if text else 0,
            'uppercase_ratio': round(sum(1 for c in text if c.isupper()) / len(text), 3) if text else 0
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
        linguistic_features = analyze_linguistic_features(text)
        
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
            "linguistic_features": linguistic_features,
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
    st.info("✨ **Enhanced Features:** Chunked processing for long audio, multiple recognition engines, improved accuracy")
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
                    with st.spinner("Transcribing audio with enhanced processing..."):
                        st.info("🔄 Using multiple recognition engines and chunked processing for better accuracy...")
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
                        
                        st.markdown("## 📝 Complete Transcription")
                        
                        # Display full transcript directly (not in expander)
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
                        
                        # Option to copy transcript
                        st.markdown("---")
                        st.markdown("**Copy Transcript:**")
                        st.text_area("Copy the text below:", transcript, height=200)
                        
                        # Detailed sentiment analysis
                        if word_count >= 3:  # Only analyze if we have sufficient text
                            with st.spinner("Performing comprehensive sentiment analysis..."):
                                sentiment_result = analyze_sentiment_detailed(transcript)
                            
                            if sentiment_result:
                                st.markdown("---")
                                st.markdown("## 📊 Comprehensive Sentiment Analysis")
                                
                                # Main sentiment display
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Overall Sentiment", 
                                             f"{sentiment_result['sentiment_emoji']} {sentiment_result['overall_sentiment']}")
                                
                                with col2:
                                    st.metric("Polarity Score", f"{sentiment_result['polarity']:.3f}")
                                
                                with col3:
                                    st.metric("Subjectivity Score", f"{sentiment_result['subjectivity']:.3f}")
                                
                                with col4:
                                    st.metric("Confidence Level", sentiment_result['confidence_level'])
                                
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
                                
                                # Linguistic Features
                                st.markdown("#### 📝 Linguistic Features")
                                ling = sentiment_result['linguistic_features']
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Word Count", ling['word_count'])
                                    st.metric("Sentences", ling['sentence_count'])
                                
                                with col2:
                                    st.metric("Avg Sentence Length", f"{ling['avg_sentence_length']} words")
                                    st.metric("Exclamation Marks", ling['exclamation_marks'])
                                
                                with col3:
                                    st.metric("Question Marks", ling['question_marks'])
                                    st.metric("Uppercase Ratio", f"{ling['uppercase_ratio']:.2%}")
                                
                                # Speaking Style Analysis
                                st.markdown("#### 🗣️ Speaking Style Analysis")
                                
                                style_notes = []
                                
                                if ling['exclamation_marks'] > 2:
                                    style_notes.append("**Emphatic speaker** - Uses exclamation marks frequently")
                                
                                if ling['question_marks'] > 2:
                                    style_notes.append("**Inquisitive speaker** - Asks many questions")
                                
                                if ling['avg_sentence_length'] > 20:
                                    style_notes.append("**Complex speaker** - Uses long, detailed sentences")
                                elif ling['avg_sentence_length'] < 8:
                                    style_notes.append("**Concise speaker** - Uses short, direct sentences")
                                
                                if ling['uppercase_ratio'] > 0.05:
                                    style_notes.append("**Animated speaker** - Uses emphasis through capitalization")
                                
                                if sentiment_result['sentiment_variance'] > 0.2:
                                    style_notes.append("**Variable emotions** - Sentiment changes throughout the speech")
                                
                                if style_notes:
                                    for note in style_notes:
                                        st.write(f"• {note}")
                                else:
                                    st.write("• **Balanced speaker** - Neutral speaking style")
                                
                                # Sentiment Progression
                                if len(sentiment_result['sentence_sentiments']) > 1:
                                    st.markdown("#### 📈 Sentiment Progression Throughout Speech")
                                    
                                    try:
                                        sentiment_df = pd.DataFrame({
                                            'Sentence': range(1, len(sentiment_result['sentence_sentiments']) + 1),
                                            'Sentiment Score': sentiment_result['sentence_sentiments']
                                        })
                                        
                                        st.line_chart(sentiment_df.set_index('Sentence'))
                                        
                                        # Sentiment trend analysis
                                        sentiments = sentiment_result['sentence_sentiments']
                                        if len(sentiments) >= 3:
                                            trend_start = sum(sentiments[:len(sentiments)//3]) / len(sentiments[:len(sentiments)//3])
                                            trend_end = sum(sentiments[-len(sentiments)//3:]) / len(sentiments[-len(sentiments)//3:])
                                            
                                            if trend_end > trend_start + 0.1:
                                                st.write("📈 **Positive trend** - Sentiment becomes more positive over time")
                                            elif trend_end < trend_start - 0.1:
                                                st.write("📉 **Negative trend** - Sentiment becomes more negative over time")
                                            else:
                                                st.write("➡️ **Stable sentiment** - Consistent emotional tone throughout")
                                    except Exception as e:
                                        st.warning(f"Could not display sentiment progression chart: {str(e)}")
                                
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
                                • Communication complexity: {"High" if ling['avg_sentence_length'] > 15 else "Moderate" if ling['avg_sentence_length'] > 10 else "Simple"}
                                """
                                
                                st.markdown(assessment)
                            else:
                                st.warning("Could not perform detailed sentiment analysis.")
                        else:
                            st.warning("⚠️ Transcript too short for meaningful sentiment analysis. Need at least 3 words.")
                    
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
