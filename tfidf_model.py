from data_loader import load_data
from preprocessing import preprocess

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

# Load data
df = load_data()

# Preprocess
df['clean_text'] = df['Overview'].apply(preprocess)

# Label
threshold = df['IMDB_Rating'].median()
df['label'] = (df['IMDB_Rating'] > threshold).astype(int)

# Split
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# TF-IDF
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model
model = LinearSVC(class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# Predict
y_pred = model.predict(X_test_tfidf)

print("\n=== TF-IDF Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))