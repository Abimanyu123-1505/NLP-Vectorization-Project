# ==============================
# 🧹 TEXT PREPROCESSING MODULE
# ==============================

import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (runs once)
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))


def preprocess(text):
    """
    Clean and preprocess text data
    
    Steps:
    - Lowercasing
    - Remove special characters
    - Tokenization
    - Stopword removal
    - Remove short words
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Tokenization (split into words)
    tokens = text.split()
    
    # Remove stopwords and short words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Join tokens back into string
    return " ".join(tokens)


# ==============================
# 🔍 TEST RUN (optional)
# ==============================

if __name__ == "__main__":
    sample_text = "This is an example sentence! It contains numbers 123 and symbols."
    
    cleaned = preprocess(sample_text)
    
    print("Original:", sample_text)
    print("Processed:", cleaned)