from data_loader import load_data
from preprocessing import preprocess

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import time

df = load_data()

df['clean_text'] = df['Overview'].apply(preprocess)

threshold = df['IMDB_Rating'].median()
df['label'] = (df['IMDB_Rating'] > threshold).astype(int)

X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

start = time.time()

bow = CountVectorizer(
    max_features=12000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9
)

X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)

time_taken = time.time() - start

model = LinearSVC(class_weight='balanced', C=1.5)
model.fit(X_train_bow, y_train)

y_pred = model.predict(X_test_bow)

print("\n===== BOW RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Time:", time_taken)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))