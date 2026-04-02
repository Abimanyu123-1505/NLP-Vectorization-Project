# ==============================
#  TF-IDF PIPELINE 
# ==============================

from data_loader import load_data
from preprocessing import preprocess

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import time

# Load data
df = load_data()

# Preprocess
df['clean_text'] = df['Overview'].apply(preprocess)

# Label creation
threshold = df['IMDB_Rating'].median()
df['label'] = (df['IMDB_Rating'] > threshold).astype(int)

# Split
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==============================
#  TF-IDF
# ==============================

start = time.time()

tfidf = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

time_taken = time.time() - start

# Model
model = LinearSVC(class_weight='balanced', C=1.5)
model.fit(X_train_tfidf, y_train)

# Prediction
y_pred = model.predict(X_test_tfidf)

# ==============================
# 📊 RESULTS
# ==============================

print("\n===== TF-IDF RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Time:", time_taken)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# 🔥 CROSS VALIDATION (EXTRA)
# ==============================

scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
print("\nCross Validation Accuracy:", scores.mean())