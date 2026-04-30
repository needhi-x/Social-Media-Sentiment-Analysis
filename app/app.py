import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Sentiment function
def predict_sentiment(text):
    score = sia.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

# UI
st.title("📊 Social Media Sentiment Analysis Dashboard")

user_input = st.text_input("Enter your text here")

if st.button("Analyze"):
    if user_input.strip() != "":
        result = predict_sentiment(user_input)
        st.success(f"Sentiment: {result}")
    else:
        st.warning("Please enter some text")