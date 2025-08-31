# app.py
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# 1. Model & Word Index Paths
# -------------------------------
MODEL_PATH = "best_model.h5"       # Change to final_model.h5 if needed
WORD_INDEX_PATH = "word_index.pkl"
MAXLEN = 200

@st.cache_resource
def load_sentiment_model():
    model = load_model(MODEL_PATH)
    with open(WORD_INDEX_PATH, "rb") as f:
        word_index = pickle.load(f)
    return model, word_index

model, word_index = load_sentiment_model()

# -------------------------------
# 2. Helper Functions
# -------------------------------
def encode_review(text, word_index, maxlen=MAXLEN):
    words = text.lower().split()
    encoded = [1]  # <START>
    for w in words:
        if w in word_index:
            encoded.append(word_index[w] + 3)
        else:
            encoded.append(2)  # <UNK>
    padded = pad_sequences([encoded], maxlen=maxlen)
    return padded

def predict_sentiment(text):
    padded = encode_review(text, word_index)
    prob = model.predict(padded)[0][0]
    if prob > 0.6:
        sentiment = "😊 Positive"
    elif prob < 0.4:
        sentiment = "😡 Negative"
    else:
        sentiment = "😐 Neutral"
    return sentiment, prob

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.set_page_config(page_title="💬 Sentiment Analysis App", layout="wide")

st.title("💬 Sentiment Analysis App")
st.write("Analyze text sentiment with Deep Learning 🚀")

# Sample reviews
sample_reviews = [
    "Absolutely loved the movie! Amazing performance!",
    "It was okay, not the best but not bad either.",
    "Terrible experience. Will never recommend.",
    "The plot was average but the acting was good.",
    "Best film I've seen this year! Brilliant!",
    "Mediocre movie, could be better.",
    "Horrible acting, very disappointing."
]

st.subheader("🧪 Try a sample or write your own")
option = st.selectbox("Pick a sample review:", ["-- Choose a sample --"] + sample_reviews)

text_input = st.text_area("✍️ Or write your own text:", "")

# If user selects a sample, override text_input
if option != "-- Choose a sample --":
    text_input = option

if st.button("📊 Predict Sentiment") and text_input.strip() != "":
    sentiment, confidence = predict_sentiment(text_input)
    st.markdown(f"**Sentiment:** {sentiment}")
    st.markdown(f"**Confidence Level:** {confidence*100:.2f}%")
    st.info("Tip: Sentiment prediction depends on vocabulary seen during training; unusual words may reduce accuracy.")
