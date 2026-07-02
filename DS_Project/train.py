import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv(r"data\Twitter_Data.csv")
df = df.dropna()

mapping = {
    -1: "Negative",
     0: "Neutral",
     1: "Positive"
}

df["category"] = df["category"].map(mapping)

X = df["clean_text"]
y = df["category"]

tfidf = TfidfVectorizer(max_features=5000)

X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

print(classification_report(y_test, pred))

import os

os.makedirs("model", exist_ok=True)

print("Accuracy:", accuracy_score(y_test, pred))

print(classification_report(y_test, pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")  # saves image
plt.show()

# Save Model
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(tfidf, "model/tfidf.pkl")

print("Model Saved Successfully")
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(tfidf, "model/tfidf.pkl")

print("Model Saved Successfully")
