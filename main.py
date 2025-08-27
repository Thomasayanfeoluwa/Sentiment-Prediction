import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
import os
import glob
import sys

# Windows path compatibility
if sys.platform.startswith('win'):
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath

# Professional page configuration
st.set_page_config(
    page_title="Sentiment Analysis Platform",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_resources():
    """Load model and vocabulary-constrained IMDB word indices."""
    
    # Find model files
    model_files = []
    for pattern in ["*.keras", "*.h5"]:
        model_files.extend(glob.glob(pattern))
    
    if not model_files:
        st.error("Model file not found. Please ensure your trained model is in the current directory.")
        st.stop()
    
    # Load best available model
    model = None
    for model_file in sorted(model_files, key=lambda x: x.endswith('.keras'), reverse=True):
        try:
            model = load_model(model_file)
            break
        except Exception:
            continue
    
    if model is None:
        st.error("Unable to load model. Please check file integrity.")
        st.stop()
    
    # Load IMDB word mappings with vocabulary constraint
    try:
        full_word_index = imdb.get_word_index()
        
        # Create vocabulary-constrained word index (top 10000 words only)
        max_features = 10000
        constrained_word_index = {
            word: idx for word, idx in full_word_index.items() 
            if idx < max_features
        }
        
        return model, constrained_word_index
        
    except Exception:
        st.error("Failed to load IMDB dataset. Please check your internet connection.")
        st.stop()

def preprocess_text(text, word_index, max_len=500):
    """Convert text to model input format with automatic vocabulary handling."""
    words = text.lower().strip().split()
    
    # Convert words to indices (constrained word_index ensures compatibility)
    # The +3 offset is because 0 is for padding, 1 is for start of sequence, and 2 is for unknown words.
    encoded = [word_index.get(word, 2) + 3 for word in words]
    
    return sequence.pad_sequences([encoded], maxlen=max_len)

def analyze_sentiment(model, preprocessed_input):
    """Generate sentiment prediction with error handling and nuanced classification."""
    try:
        prediction = model.predict(preprocessed_input, verbose=0)
        score = float(prediction[0][0])
        
        # Corrected Logic: Lower score indicates positive sentiment, higher score indicates negative.
        if score < 0.2:
            sentiment = 'Strongly Positive'
        elif score < 0.4:
            sentiment = 'Positive'
        elif score < 0.6:
            sentiment = 'Neutral'
        elif score < 0.8:
            sentiment = 'Negative'
        else:
            sentiment = 'Strongly Negative'
            
        # Confidence calculation remains the same, as it measures distance from the neutral 0.5 midpoint.
        confidence = abs(score - 0.5) * 2
        return sentiment, confidence, score
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None, None

def main():
    # Professional header
    st.title("Sentiment Analysis Platform")
    st.markdown("Advanced movie review sentiment classification using deep learning")
    st.divider()
    
    # Load resources
    model, word_index = load_resources()
    
    # Clean input section
    st.subheader("Review Analysis")
    
    # Example selector
    examples = [
        "Select a sample review...",
        "This movie exceeded all my expectations with outstanding performances.",
        "Poorly written script with terrible acting throughout.",
        "Average film with some good moments but overall forgettable.",
        "Brilliant cinematography and compelling storyline make this a masterpiece.",
        "An absolute disaster from start to finish. A waste of time and money.",
    ]
    
    selected = st.selectbox("Sample Reviews", examples, label_visibility="collapsed")
    
    # Main text input
    if selected != examples[0]:
        default_value = selected
    else:
        default_value = ""
    
    review_text = st.text_area(
        "Enter Review Text",
        value=default_value,
        height=120,
        placeholder="Write your movie review here...",
        label_visibility="collapsed"
    )
    
    # Analysis button
    if st.button("Analyze Sentiment", type="primary", use_container_width=True):
        if not review_text.strip():
            st.warning("Please enter a review to analyze.")
            return
        
        if len(review_text.strip().split()) < 3:
            st.warning("Please provide a more detailed review for accurate analysis.")
            return
        
        # Process analysis
        with st.spinner("Processing..."):
            preprocessed = preprocess_text(review_text, word_index)
            sentiment, confidence, score = analyze_sentiment(model, preprocessed)
        
        # Check for prediction errors
        if sentiment is None:
            st.error("Unable to analyze this review. Please try different text.")
            return
        
        # Results section
        st.divider()
        st.subheader("Analysis Results")
        
        # Clean results display
        col1, col2 = st.columns(2)
        
        with col1:
            # Display sentiment with appropriate color
            if 'Positive' in sentiment:
                st.success(f"**{sentiment}**")
            elif 'Negative' in sentiment:
                st.error(f"**{sentiment}**")
            else:
                st.info(f"**{sentiment}**")
        
        with col2:
            st.metric("Confidence", f"{confidence:.0%}")
        
        # Technical metrics (collapsed by default)
        with st.expander("Detailed Metrics"):
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Prediction Score", f"{score:.4f}")
            with metric_col2:
                st.metric("Word Count", len(review_text.split()))
        
        # Confidence interpretation
        if confidence > 0.8:
            st.info("High confidence prediction")
        elif confidence > 0.6:
            st.warning("Moderate confidence prediction")
        else:
            st.warning("Low confidence - the sentiment may be ambiguous.")
    
    # Professional footer
    st.divider()
    
    # Minimal sidebar with technical info
    with st.sidebar:
        st.header("Model Information")
        st.info(f"""
        **Framework**: TensorFlow {tf.__version__}
        **Architecture**: Recurrent Neural Network
        **Training Data**: IMDB Movie Reviews
        **Vocabulary**: 10,000 words
        """)
        
        if st.button("View System Info"):
            st.code(f"""
Python: {sys.version.split()[0]}
Platform: {sys.platform}
Model Input: {model.input_shape if hasattr(model, 'input_shape') else 'N/A'}
            """)

if __name__ == "__main__":
    main()
