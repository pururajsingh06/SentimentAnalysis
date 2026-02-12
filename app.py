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
    layout="centered"
)

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

    model = tf.keras.models.load_model("C:\Users\acer\OneDrive\Desktop\sentiment analysis\lstm_sentiment_model.h5")
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
# UI
# -------------------------------
st.title("🌐 Multilingual LSTM Sentiment Analysis App")
st.write("Enter text in **any language**, and we'll analyze its **sentiment**!")

user_input = st.text_area("✏️ Enter your sentence:", height=150)

MAX_LEN = 100

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
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

            label = "Positive 😊" if prediction >= 0.5 else "Negative 😞"

            # Display results
            st.subheader("🧾 Result")
            st.write(f"**Original Input:** {user_input}")
            st.write(f"**Translated (EN):** {translated_text}")
            st.write(f"**Sentiment:** {label}")
            st.write(f"**Confidence:** {prediction:.2%}")

            # Word frequency chart
            word_freq = Counter(word_list)
            top_words = word_freq.most_common(5)

            if len(top_words) > 0:
                words = [word for word, freq in top_words]
                freqs = [freq for word, freq in top_words]

                fig, ax = plt.subplots()
                ax.bar(words, freqs)
                ax.set_title("Top Frequent Words After Cleaning")
                ax.set_xlabel("Words")
                ax.set_ylabel("Frequency")
                ax.set_ylim(0, max(freqs) + 1)
                ax.grid(axis="y", linestyle="--", alpha=0.7)

                st.pyplot(fig)
            else:
                st.info("No words to display.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
