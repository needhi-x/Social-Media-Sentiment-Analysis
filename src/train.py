import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

from preprocess import clean_text

# Load data
df = pd.read_csv("data/sample_data.csv")

# Fix labels
df['sentiment'] = df['sentiment'].str.lower().str.strip()

# Clean text
df['cleaned'] = df['text'].apply(clean_text)

# Remove empty rows
df = df[df['cleaned'] != ""]

# Features and labels
X = df['cleaned']
y = df['sentiment']

# Convert text → numbers
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split data
model = LogisticRegression()
model.fit(X_vec, y)

y_pred = model.predict(X_vec)

print("\n✅ Accuracy:", accuracy_score(y, y_pred))

# Save model
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

import pickle
import os

os.makedirs("models", exist_ok=True)

pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))