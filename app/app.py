import streamlit as st
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon safely
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize analyzer
sia = SentimentIntensityAnalyzer()

# Session state to store history
if "data" not in st.session_state:
    st.session_state.data = []

# Sentiment function
def predict_sentiment(text):
    score = sia.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# UI
st.title("📊 Social Media Sentiment Analysis Dashboard")

user_input = st.text_input("Enter your text here")

if st.button("Analyze"):
    if user_input.strip() != "":
        sentiment = predict_sentiment(user_input)

        # Save data
        st.session_state.data.append({
            "Text": user_input,
            "Sentiment": sentiment
        })

        st.success(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text")

# Show Data Table
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data)

    st.subheader("📋 Analyzed Data")
    st.dataframe(df)

    # Count sentiments
    sentiment_counts = df["Sentiment"].value_counts()

    st.subheader("📊 Sentiment Distribution")

    # Plot chart
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")

    st.pyplot(fig)