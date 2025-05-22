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
import numpy as np

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

def create_summary(text, max_sentences=4):
    """Create a concise summary of the transcript in 4-5 lines"""
    if not text.strip():
        return "No content available for summary."
    
    # Split text into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= max_sentences:
        return text
    
    # Simple extractive summarization based on sentence length and keyword frequency
    words = text.lower().split()
    word_freq = Counter(words)
    
    # Score sentences based on word frequency
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_words = sentence.lower().split()
        score = sum(word_freq[word] for word in sentence_words if word in word_freq)
        sentence_scores[i] = score / len(sentence_words) if sentence_words else 0
    
    # Select top sentences
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
    top_sentences = sorted([idx for idx, score in top_sentences])
    
    summary = '. '.join([sentences[i] for i in top_sentences if i < len(sentences)])
    return summary + '.' if summary and not summary.endswith('.') else summary

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
    """Analyze linguistic features of the text"""
    blob = TextBlob(text)
    
    # Basic statistics
    word_count = len(text.split())
    sentence_count = len(blob.sentences)
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # Punctuation analysis
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # Uppercase analysis (indicating emphasis/shouting)
    uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length,
        'exclamation_marks': exclamation_count,
        'question_marks': question_count,
        'uppercase_ratio': uppercase_ratio
    }

def analyze_sentiment_detailed(text):
    """Comprehensive sentiment analysis of the text"""
    if not text.strip():
        return None
    
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
    
    # Sentence-level sentiment analysis
    sentence_sentiments = []
    for sentence in blob.sentences:
        sent_polarity = sentence.sentiment.polarity
        sentence_sentiments.append(sent_polarity)
    
    sentiment_variance = np.var(sentence_sentiments) if sentence_sentiments else 0
    
    return {
        "overall_sentiment": sentiment,
        "sentiment_emoji": sentiment_emoji,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "confidence_level": confidence_level,
        "subjectivity_level": subjectivity_level,
        "emotional_keywords": emotional_keywords,
        "linguistic_features": linguistic_features,
        "sentence_sentiments": sentence_sentiments,
        "sentiment_variance": sentiment_variance
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
                        st.markdown("## 📝 Transcription Summary")
                        
                        # Create and display summary
                        summary = create_summary(transcript, max_sentences=4)
                        st.subheader("Summary (4-5 lines):")
                        st.write(summary)
                        
                        # Option to view full transcript
                        with st.expander("📄 View Full Transcript"):
                            st.write(transcript)
                        
                        # Detailed sentiment analysis
                        with st.spinner("Performing detailed sentiment analysis..."):
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
                                st.metric("Avg Sentence Length", f"{ling['avg_sentence_length']:.1f} words")
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
                                
                                sentiment_df = pd.DataFrame({
                                    'Sentence': range(1, len(sentiment_result['sentence_sentiments']) + 1),
                                    'Sentiment Score': sentiment_result['sentence_sentiments']
                                })
                                
                                st.line_chart(sentiment_df.set_index('Sentence'))
                                
                                # Sentiment trend analysis
                                sentiments = sentiment_result['sentence_sentiments']
                                if len(sentiments) >= 3:
                                    trend_start = np.mean(sentiments[:len(sentiments)//3])
                                    trend_end = np.mean(sentiments[-len(sentiments)//3:])
                                    
                                    if trend_end > trend_start + 0.1:
                                        st.write("📈 **Positive trend** - Sentiment becomes more positive over time")
                                    elif trend_end < trend_start - 0.1:
                                        st.write("📉 **Negative trend** - Sentiment becomes more negative over time")
                                    else:
                                        st.write("➡️ **Stable sentiment** - Consistent emotional tone throughout")
                            
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
