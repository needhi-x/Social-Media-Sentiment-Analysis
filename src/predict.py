import pickle

# Load model and vectorizer
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return prediction


if __name__ == "__main__":
    while True:
        user_input = input("\nEnter text (or type 'exit'): ")

        if user_input.lower() == "exit":
            break

        result = predict_sentiment(user_input)
        print("Sentiment:", result)