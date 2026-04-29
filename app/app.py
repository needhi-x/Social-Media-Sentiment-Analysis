import streamlit as st
import pickle

# Load model
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

st.title("💬 Social Media Sentiment Analyzer")
st.write("Type a message and check if it's Positive or Negative")

# Input box
user_input = st.text_area("Enter your text here:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        text_vec = vectorizer.transform([user_input])
        result = model.predict(text_vec)[0]

        if result == "positive":
            st.success("😊 Positive Sentiment")
        else:
            st.error("😠 Negative Sentiment")