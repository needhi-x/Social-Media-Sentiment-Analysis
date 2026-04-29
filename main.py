from src.preprocess import clean_text

test = "This app is AMAZING!!! 😍🔥"
print("Original:", test)
print("Cleaned:", clean_text(test))