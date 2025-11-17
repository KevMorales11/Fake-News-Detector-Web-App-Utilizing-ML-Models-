# fake-news-detector.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load and Prepare the Data
# -----------------------------
true_df = pd.read_csv('C:\\Users\\music\\fake-news-detector\\Data\\True.csv')
fake_df = pd.read_csv('C:\\Users\\music\\fake-news-detector\\Data\\Fake.csv')

# Add labels
true_df['label'] = 1  # Real
fake_df['label'] = 0  # Fake 

# Balance both datasets
min_len = min(len(true_df), len(fake_df))
true_df = true_df.sample(min_len, random_state=42)
fake_df = fake_df.sample(min_len, random_state=42)

# Combine and shuffle
df = pd.concat([true_df, fake_df], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset size: {df.shape}")

# -----------------------------
# 2. Combine title + text
# -----------------------------
if 'title' in df.columns:
    df["content"] = df["title"].fillna('') + " " + df["text"].fillna('')
else:
    df["content"] = df["text"].fillna('')

X = df["content"].astype(str)
y = df["label"]

# -----------------------------
# 3. Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4. TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    max_features=10000,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Vectorized data shape:", X_train_vec.shape, X_test_vec.shape)

# -----------------------------
# 5. Train the Model
# -----------------------------
model = LogisticRegression(max_iter=2000, C=2.0)
model.fit(X_train_vec, y_train)

# -----------------------------
# 6. Evaluate the Model
# -----------------------------
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# -----------------------------
# 7. Example Prediction
# -----------------------------
example = ["The economy is expected to grow by 5% next year."]
example_vec = vectorizer.transform(example)
prediction = model.predict(example_vec)
print("\nExample Prediction:", "Real News" if prediction[0] == 1 else "Fake News")

# -----------------------------
# 8. 🔹 User Input Prediction
# -----------------------------
while True:
    user_input = input("\nEnter an article or headline (or type 'quit' to exit):\n> ")

    if user_input.strip().lower() == 'quit':
        print("Goodbye! 👋")
        break

    if not user_input.strip():
        print("Please enter some text.")
        continue

    # Transform and predict
    user_vec = vectorizer.transform([user_input])
    user_pred = model.predict(user_vec)

    # Display result
    print("\n📰 Prediction:", "✅ Real News" if user_pred[0] == 1 else "❌ Fake News")
