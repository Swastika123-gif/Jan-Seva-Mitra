import pandas as pd
import re
import pickle
from pathlib import Path
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("complaint_dataset.csv")

# ----------------------------
# Clean text
# ----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\\s]", " ", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["cleaned_text"] = df["complaint_text"].apply(clean_text)

# ----------------------------
# Train-test split
# ----------------------------
X = df["cleaned_text"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# Train model
# ----------------------------
model = LinearSVC()
model.fit(X_train_vec, y_train)

# ----------------------------
# Evaluate
# ----------------------------
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# Save model files
# ----------------------------
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

with open(models_dir / "svm_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(models_dir / "tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved successfully in 'models/' folder.")