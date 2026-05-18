import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import pickle
import re
import os
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from googletrans import Translator
from collections import Counter
import asyncio

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Multilingual LSTM Sentiment Analyzer",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f8fafc;
        font-weight: 700;
    }
    
    /* Text input area */
    .stTextArea textarea {
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.5);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 10px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
        border: none;
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e293b;
        color: #f8fafc;
        border-radius: 8px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# SAFE FILE PATHS (VERY IMPORTANT FOR CLOUD)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")
model_path = os.path.join(BASE_DIR, "lstm_sentiment_model.h5")

# -------------------------------
# LOAD TOKENIZER & MODEL
# -------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)

    model = tf.keras.models.load_model(model_path)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# -------------------------------
# NLTK STOPWORDS (SAFE DOWNLOAD)
# -------------------------------
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# -------------------------------
# TRANSLATION FUNCTION
# -------------------------------
def translate_text_sync(user_input):
    async def inner():
        async with Translator() as translator:
            return await translator.translate(user_input, dest="en")
    return asyncio.run(inner())

# -------------------------------
# TEXT CLEANING FUNCTION
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words), words

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8636/8636906.png", width=100)
    st.title("About this App")
    st.info(
        "This application uses a deep learning LSTM model to predict the sentiment of your text. "
        "It supports multilingual inputs by automatically translating them to English before analysis!"
    )
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. **Translate** text to English\n2. **Clean** stopwords and punctuation\n3. **Analyze** with Keras LSTM\n4. **Visualize** word frequency")
    st.markdown("---")
    st.caption("Powered by Streamlit, TensorFlow, and NLTK")

# -------------------------------
# MAIN UI
# -------------------------------
st.title("✨ Multilingual Sentiment Analysis")
st.markdown("Analyze the sentiment of your text in **any language** in real-time.")

# Create two columns for the main layout
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("✏️ What's on your mind?", height=150, placeholder="Type something like 'I love this amazing product!' or 'Je suis très triste aujourd'hui...'")

MAX_LEN = 100

with col2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

if analyze_btn:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        with st.spinner("🧠 Analyzing sentiment..."):
            try:
                # Translate
                translation = translate_text_sync(user_input)
                translated_text = translation.text

                # Clean
                cleaned_text, word_list = clean_text(translated_text)

                # Predict
                seq = tokenizer.texts_to_sequences([cleaned_text])
                padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
                prediction = model.predict(padded)[0][0]

                # Label and colors
                is_positive = prediction >= 0.5
                label = "Positive 😊" if is_positive else "Negative 😞"
                color = "green" if is_positive else "red"
                confidence = prediction if is_positive else (1 - prediction)

                st.markdown("---")
                
                # Metrics Row
                m1, m2 = st.columns(2)
                with m1:
                    st.metric(label="Sentiment", value=label)
                with m2:
                    st.metric(label="Confidence", value=f"{confidence:.2%}")

                # Details Expander
                with st.expander("📝 View Detailed Analysis", expanded=True):
                    st.markdown(f"**Original Input:** `{user_input}`")
                    if translated_text.lower() != user_input.lower():
                        st.markdown(f"**Translated to English:** `{translated_text}`")
                    
                    st.markdown("### Top Words Used")
                    # Word frequency chart (Streamlit native)
                    word_freq = Counter(word_list)
                    top_words = word_freq.most_common(5)

                    if len(top_words) > 0:
                        import pandas as pd
                        df = pd.DataFrame(top_words, columns=["Word", "Frequency"]).set_index("Word")
                        st.bar_chart(df, color="#3b82f6")
                    else:
                        st.info("No significant words to display.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
