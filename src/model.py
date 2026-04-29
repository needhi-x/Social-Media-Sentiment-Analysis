import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# load dataset
df = pd.read_csv("data/sample_data.csv")

# clean labels
df["sentiment"] = df["sentiment"].str.strip().str.lower()

X = df["text"]
y = df["sentiment"]

# vectorize
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
X = vectorizer.fit_transform(X)

# split (IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# model
model = LogisticRegression(max_iter=2000, C=0.7)

model.fit(X_train, y_train)

# results
print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))

# save
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))