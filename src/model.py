import pandas as pd
import pickle
from nltk.sentiment import SentimentIntensityAnalyzer

# load data (optional, for reference only)
df = pd.read_csv("data/sample_data.csv")

# VADER model
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

# test accuracy (simple check)
df["predicted"] = df["text"].apply(get_sentiment)

print(df[["text", "sentiment", "predicted"]].head())

# save function (for streamlit use)
pickle.dump(sia, open("models/vader.pkl", "wb"))