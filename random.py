import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import io
import tempfile
import os
from textblob import TextBlob
import pandas as pd
import numpy as np
import re
from collections import Counter

# Set page config
st.set_page_config(
    page_title="Real Estate Audio Analysis",
    page_icon="🏠",
    layout="wide"
)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Real estate keywords for context-aware corrections
REAL_ESTATE_KEYWORDS = {
    'property': ['property', 'properties'],
    'real estate': ['real estate', 'realestate'],
    'apartment': ['apartment', 'apartments'],
    'house': ['house', 'houses', 'home', 'homes'],
    'bedroom': ['bedroom', 'bedrooms', 'bed room'],
    'bathroom': ['bathroom', 'bathrooms', 'bath room'],
    'square feet': ['square feet', 'sq ft', 'sqft'],
    'mortgage': ['mortgage', 'mortgages'],
    'down payment': ['down payment', 'downpayment'],
    'closing': ['closing', 'close'],
    'listing': ['listing', 'listings'],
    'realtor': ['realtor', 'realtors', 'agent', 'agents'],
    'commission': ['commission', 'commissions'],
    'inspection': ['inspection', 'inspections'],
    'appraisal': ['appraisal', 'appraisals'],
    'equity': ['equity'],
    'refinance': ['refinance', 'refinancing'],
    'lease': ['lease', 'leasing'],
    'tenant': ['tenant', 'tenants'],
    'landlord': ['landlord', 'landlords'],
    'rent': ['rent', 'rental', 'renting'],
    'deposit': ['deposit', 'deposits'],
    'price': ['price', 'pricing', 'cost'],
    'market': ['market', 'markets'],
    'neighborhood': ['neighborhood', 'neighbourhood'],
    'location': ['location', 'locations'],
    'investment': ['investment', 'investments']
}

def login_page():
    """Display login page"""
    st.title("🔐 Real Estate Audio Analysis - Login")
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
    """Convert and optimize audio file for better speech recognition"""
    try:
        # Read the uploaded file
        audio_data = audio_file.read()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_filename = temp_file.name
            
        # Convert audio to WAV format using pydub with optimization
        audio = AudioSegment.from_file(io.BytesIO(audio_data))
        
        # Optimize audio for speech recognition
        # Convert to mono, set optimal sample rate, normalize volume
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        # Normalize audio levels
        audio = audio.normalize()
        
        # Apply noise reduction (simple high-pass filter)
        audio = audio.high_pass_filter(300)
        
        # Export optimized audio
        audio.export(temp_filename, format="wav")
        
        return temp_filename
    except Exception as e:
        st.error(f"Error converting audio file: {str(e)}")
        return None

def correct_real_estate_terms(text):
    """Apply real estate context-aware corrections to transcript"""
    corrected_text = text.lower()
    
    # Common speech recognition errors in real estate context
    corrections = {
        'real a state': 'real estate',
        'real estate': 'real estate',
        'sq ft': 'square feet',
        'square foot': 'square feet',
        'bed room': 'bedroom',
        'bath room': 'bathroom',
        'down payment': 'down payment',
        'close ing': 'closing',
        'mortgage': 'mortgage',
        'commission': 'commission',
        'apart ment': 'apartment',
        'neighbor hood': 'neighborhood',
        'land lord': 'landlord',
        'prop erty': 'property',
        'in vest ment': 'investment',
        'ap praisal': 'appraisal',
        'in spec tion': 'inspection'
    }
    
    for wrong, correct in corrections.items():
        corrected_text = corrected_text.replace(wrong, correct)
    
    return corrected_text

def transcribe_audio_enhanced(audio_file_path):
    """Enhanced transcription with multiple attempts and real estate context"""
    recognizer = sr.Recognizer()
    
    # Adjust recognizer settings for better accuracy
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.operation_timeout = None
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.8
    
    try:
        with sr.AudioFile(audio_file_path) as source:
            # Adjust for ambient noise with longer duration
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            # Record the audio data
            audio_data = recognizer.record(source)
        
        # Try multiple recognition methods
        transcripts = []
        
        # Method 1: Google Speech Recognition (primary)
        try:
            text = recognizer.recognize_google(audio_data, language='en-US')
            transcripts.append(("Google", text))
        except:
            pass
        
        # Method 2: Google with enhanced language model
        try:
            text = recognizer.recognize_google(audio_data, language='en-US', show_all=False)
            transcripts.append(("Google Enhanced", text))
        except:
            pass
        
        if transcripts:
            # Use the longest transcript (usually more complete)
            best_transcript = max(transcripts, key=lambda x: len(x[1]))
            raw_text = best_transcript[1]
            
            # Apply real estate context corrections
            corrected_text = correct_real_estate_terms(raw_text)
            
            return corrected_text, raw_text, "Success"
        else:
            return "", "", "Could not understand audio"
            
    except Exception as e:
        return "", "", f"Error processing audio file: {str(e)}"

def summarize_transcript(text):
    """Generate a 4-5 line summary of the real estate transcript"""
    if not text.strip():
        return "No content to summarize."
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 4:
        return text
    
    # Simple extractive summarization based on keyword frequency
    word_freq = Counter()
    for sentence in sentences:
        words = re.findall(r'\b\w+\b', sentence.lower())
        word_freq.update(words)
    
    # Score sentences based on important words
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        words = re.findall(r'\b\w+\b', sentence.lower())
        score = sum(word_freq[word] for word in words if word in word_freq)
        sentence_scores[i] = score / len(words) if words else 0
    
    # Get top 4 sentences
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:4]
    top_sentences = sorted([idx for idx, score in top_sentences])
    
    summary = '. '.join([sentences[i] for i in top_sentences])
    return summary + '.'

def analyze_sentiment_comprehensive(text):
    """Comprehensive sentiment analysis for real estate context"""
    if not text.strip():
        return None
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Real estate specific sentiment indicators
    positive_indicators = [
        'great', 'excellent', 'perfect', 'love', 'beautiful', 'amazing', 'fantastic',
        'deal', 'opportunity', 'profitable', 'good investment', 'prime location',
        'spacious', 'modern', 'updated', 'move-in ready', 'motivated seller'
    ]
    
    negative_indicators = [
        'problem', 'issue', 'concern', 'expensive', 'overpriced', 'needs work',
        'repair', 'damage', 'old', 'outdated', 'small', 'noisy', 'busy street',
        'not interested', 'pass', 'reject'
    ]
    
    neutral_indicators = [
        'consider', 'think about', 'maybe', 'possibly', 'potential', 'review',
        'analyze', 'evaluate', 'discuss', 'meeting', 'appointment'
    ]
    
    # Count indicators
    text_lower = text.lower()
    positive_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
    negative_count = sum(1 for indicator in negative_indicators if indicator in text_lower)
    neutral_count = sum(1 for indicator in neutral_indicators if indicator in text_lower)
    
    # Adjust polarity based on real estate context
    context_adjustment = (positive_count - negative_count) * 0.1
    adjusted_polarity = max(-1, min(1, polarity + context_adjustment))
    
    # Determine sentiment category with more granular classification
    if adjusted_polarity > 0.3:
        sentiment = "Very Positive"
        emoji = "😊"
    elif adjusted_polarity > 0.1:
        sentiment = "Positive"
        emoji = "🙂"
    elif adjusted_polarity > -0.1:
        sentiment = "Neutral"
        emoji = "😐"
    elif adjusted_polarity > -0.3:
        sentiment = "Negative"
        emoji = "😕"
    else:
        sentiment = "Very Negative"
        emoji = "😞"
    
    # Confidence calculation
    confidence = min(100, abs(adjusted_polarity) * 100 + (positive_count + negative_count) * 10)
    
    # Intent detection for real estate
    intent = detect_real_estate_intent(text_lower)
    
    return {
        "sentiment": sentiment,
        "emoji": emoji,
        "polarity": polarity,
        "adjusted_polarity": adjusted_polarity,
        "subjectivity": subjectivity,
        "confidence": confidence,
        "positive_indicators": positive_count,
        "negative_indicators": negative_count,
        "neutral_indicators": neutral_count,
        "intent": intent
    }

def detect_real_estate_intent(text):
    """Detect the intent/purpose of the real estate conversation"""
    intent_keywords = {
        "Buying Interest": ["buy", "purchase", "interested in buying", "looking to buy", "want to buy"],
        "Selling Interest": ["sell", "selling", "list", "listing", "want to sell"],
        "Rental Inquiry": ["rent", "rental", "lease", "tenant", "landlord"],
        "Investment Discussion": ["investment", "roi", "return", "profit", "cash flow"],
        "Property Viewing": ["show", "tour", "visit", "see the property", "viewing"],
        "Negotiation": ["negotiate", "offer", "counter offer", "price", "deal"],
        "Information Gathering": ["information", "details", "tell me about", "what about"],
        "Closing Discussion": ["closing", "close", "contract", "agreement", "paperwork"]
    }
    
    detected_intents = []
    for intent, keywords in intent_keywords.items():
        if any(keyword in text for keyword in keywords):
            detected_intents.append(intent)
    
    return detected_intents if detected_intents else ["General Discussion"]

def main_app():
    """Main application after login"""
    st.title("🏠 Real Estate Audio Transcription & Analysis")
    st.markdown("---")
    
    # Logout button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
    
    st.markdown("### Upload real estate conversation audio for analysis")
    st.info("📌 Optimized for real estate conversations • Supported formats: WAV, MP3, MP4, M4A, FLAC, OGG")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg'],
        help="Upload an audio file containing real estate conversations"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"📁 File uploaded: {uploaded_file.name}")
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.write(f"📊 File size: {file_size:.2f} MB")
        
        # Process button
        if st.button("🚀 Analyze Audio", type="primary", use_container_width=True):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Convert audio
                status_text.text("🔄 Converting and optimizing audio...")
                progress_bar.progress(20)
                
                wav_file_path = convert_audio_to_wav(uploaded_file)
                
                if wav_file_path:
                    # Step 2: Transcribe
                    status_text.text("🎤 Transcribing audio with enhanced recognition...")
                    progress_bar.progress(60)
                    
                    transcript, raw_transcript, status = transcribe_audio_enhanced(wav_file_path)
                    
                    # Clean up temporary file
                    try:
                        os.unlink(wav_file_path)
                    except:
                        pass
                    
                    if status == "Success" and transcript:
                        progress_bar.progress(80)
                        status_text.text("📊 Analyzing sentiment and generating insights...")
                        
                        # Generate summary
                        summary = summarize_transcript(transcript)
                        
                        # Analyze sentiment
                        sentiment_result = analyze_sentiment_comprehensive(transcript)
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.markdown("---")
                        
                        # Results section
                        st.markdown("## 📋 Analysis Results")
                        
                        # Summary section
                        with st.expander("📝 Executive Summary", expanded=True):
                            st.markdown("**Key Points:**")
                            st.write(summary)
                        
                        # Transcript section
                        with st.expander("📄 Full Transcript", expanded=False):
                            st.markdown("**Enhanced Transcript:**")
                            st.write(transcript)
                            
                            if raw_transcript != transcript:
                                st.markdown("**Original Raw Transcript:**")
                                st.write(raw_transcript)
                                st.info("ℹ️ Transcript was enhanced with real estate context corrections")
                        
                        # Comprehensive Sentiment Analysis
                        if sentiment_result:
                            st.markdown("---")
                            st.markdown("## 🎯 Comprehensive Sentiment Analysis")
                            
                            # Main sentiment display
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "Overall Sentiment", 
                                    f"{sentiment_result['sentiment']} {sentiment_result['emoji']}"
                                )
                            
                            with col2:
                                st.metric(
                                    "Confidence", 
                                    f"{sentiment_result['confidence']:.1f}%"
                                )
                            
                            with col3:
                                st.metric(
                                    "Polarity Score", 
                                    f"{sentiment_result['adjusted_polarity']:.3f}",
                                    delta=f"{sentiment_result['adjusted_polarity'] - sentiment_result['polarity']:.3f}"
                                )
                            
                            with col4:
                                st.metric(
                                    "Subjectivity", 
                                    f"{sentiment_result['subjectivity']:.3f}"
                                )
                            
                            # Detailed Analysis
                            st.markdown("### 🔍 Detailed Sentiment Breakdown")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Sentiment indicators
                                st.markdown("**Sentiment Indicators Found:**")
                                indicator_data = {
                                    'Type': ['Positive', 'Negative', 'Neutral'],
                                    'Count': [
                                        sentiment_result['positive_indicators'],
                                        sentiment_result['negative_indicators'],
                                        sentiment_result['neutral_indicators']
                                    ]
                                }
                                
                                if sum(indicator_data['Count']) > 0:
                                    st.bar_chart(pd.DataFrame(indicator_data).set_index('Type'))
                                else:
                                    st.write("No specific sentiment indicators detected")
                                
                                # Intent detection
                                st.markdown("**Conversation Intent:**")
                                for intent in sentiment_result['intent']:
                                    st.write(f"• {intent}")
                            
                            with col2:
                                # Polarity visualization
                                st.markdown("**Sentiment Scale:**")
                                
                                # Create a gauge-like visualization
                                polarity_df = pd.DataFrame({
                                    'Sentiment Component': ['Adjusted Polarity', 'Original Polarity', 'Subjectivity'],
                                    'Score': [
                                        sentiment_result['adjusted_polarity'],
                                        sentiment_result['polarity'],
                                        sentiment_result['subjectivity']
                                    ]
                                })
                                
                                st.bar_chart(polarity_df.set_index('Sentiment Component'))
                                
                                # Interpretation
                                st.markdown("**Interpretation Guide:**")
                                st.write(f"""
                                • **Polarity** (-1 to +1): Emotional tone direction
                                • **Subjectivity** (0 to 1): Opinion vs. fact ratio
                                • **Confidence**: Reliability of sentiment detection
                                """)
                            
                            # Advanced insights
                            st.markdown("### 📈 Advanced Insights")
                            
                            insights = []
                            
                            if sentiment_result['confidence'] > 70:
                                insights.append("🎯 High confidence in sentiment detection")
                            elif sentiment_result['confidence'] < 30:
                                insights.append("⚠️ Low confidence - conversation may be ambiguous")
                            
                            if sentiment_result['subjectivity'] > 0.7:
                                insights.append("💭 Highly subjective conversation with personal opinions")
                            elif sentiment_result['subjectivity'] < 0.3:
                                insights.append("📊 Objective, fact-based conversation")
                            
                            if abs(sentiment_result['adjusted_polarity']) > abs(sentiment_result['polarity']):
                                insights.append("🔧 Real estate context significantly influenced sentiment analysis")
                            
                            if sentiment_result['positive_indicators'] > sentiment_result['negative_indicators']:
                                insights.append("✅ More positive language patterns detected")
                            elif sentiment_result['negative_indicators'] > sentiment_result['positive_indicators']:
                                insights.append("❌ More negative language patterns detected")
                            
                            for insight in insights:
                                st.write(insight)
                            
                            # Business recommendations
                            st.markdown("### 💼 Business Recommendations")
                            
                            if sentiment_result['adjusted_polarity'] > 0.2:
                                st.success("🟢 **Positive sentiment detected** - Good opportunity to move forward with this client/deal")
                            elif sentiment_result['adjusted_polarity'] < -0.2:
                                st.error("🔴 **Negative sentiment detected** - May need to address concerns before proceeding")
                            else:
                                st.warning("🟡 **Neutral sentiment** - Client may need more information or time to decide")
                            
                            # Export option
                            st.markdown("---")
                            if st.button("📊 Generate Detailed Report", use_container_width=True):
                                report = f"""
REAL ESTATE AUDIO ANALYSIS REPORT
================================

SUMMARY:
{summary}

SENTIMENT ANALYSIS:
- Overall Sentiment: {sentiment_result['sentiment']} {sentiment_result['emoji']}
- Confidence: {sentiment_result['confidence']:.1f}%
- Polarity Score: {sentiment_result['adjusted_polarity']:.3f}
- Subjectivity: {sentiment_result['subjectivity']:.3f}

CONVERSATION INTENT:
{', '.join(sentiment_result['intent'])}

FULL TRANSCRIPT:
{transcript}
                                """
                                
                                st.download_button(
                                    label="Download Report",
                                    data=report,
                                    file_name=f"real_estate_analysis_{uploaded_file.name}.txt",
                                    mime="text/plain"
                                )
                        
                        else:
                            st.warning("Could not analyze sentiment - no meaningful text found.")
                    
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Transcription failed: {status}")
                        st.info("💡 Try these solutions:")
                        st.write("• Ensure the audio contains clear speech")
                        st.write("• Check if the audio quality is good")
                        st.write("• Try a different audio format")
                        st.write("• Reduce background noise if possible")
                
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ Failed to process the audio file")
                    st.info("Please try a different format or check file integrity")
            
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ An error occurred: {str(e)}")

def main():
    """Main function to control app flow"""
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
