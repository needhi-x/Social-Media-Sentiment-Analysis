import re

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.strip()
    return text