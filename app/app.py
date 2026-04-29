
import streamlit as st

import pickle
import matplotlib.pyplot as plt

# Load model + vectorizer
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
st.sidebar.title("📊 Dashboard Info")
st.sidebar.write("Social Media Sentiment Analysis")
st.sidebar.write("Classifies text into Positive, Negative, Neutral")

# Main Title
st.title("📊 Social Media Sentiment Analysis Dashboard")
st.write("Enter any social media comment and analyze its sentiment 😊")

# Input box
text = st.text_input("Enter text here:")

# Predict button
if st.button("Analyze Sentiment"):

    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        data = vectorizer.transform([text])
        prediction = model.predict(data)[0]

        # Save history
        st.session_state.history.append((text, prediction))

        # Output
        st.success(f"Predicted Sentiment: {prediction}")

# =========================
# HISTORY SECTION
# =========================
st.subheader("📜 Prediction History")

for t, p in st.session_state.history[::-1]:
    st.write(f"**{t} → {p}**")

# =========================
# SIMPLE CHART
# =========================
st.subheader("📊 Sentiment Overview")

if len(st.session_state.history) > 0:

    labels = [x[1] for x in st.session_state.history]
    counts = {
        "positive": labels.count("positive"),
        "negative": labels.count("negative"),
        "neutral": labels.count("neutral")
    }

    fig, ax = plt.subplots()
    ax.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%")
    st.pyplot(fig)

else:
    st.info("No data yet. Start analyzing text.")