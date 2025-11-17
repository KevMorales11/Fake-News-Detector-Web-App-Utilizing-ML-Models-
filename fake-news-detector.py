# fake-news-detector.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

from flask import Flask, request, render_template_string
import webbrowser  

app = Flask(__name__)


# 1. Load and Prepare the Data
true_df = pd.read_csv('C:\\Users\\music\\fake-news-detector\\Data\\True.csv')
fake_df = pd.read_csv('C:\\Users\\music\\fake-news-detector\\Data\\Fake.csv')

true_df['label'] = 1
fake_df['label'] = 0

min_len = min(len(true_df), len(fake_df))
true_df = true_df.sample(min_len, random_state=42)
fake_df = fake_df.sample(min_len, random_state=42)

df = pd.concat([true_df, fake_df], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

if 'title' in df.columns:
    df["content"] = df["title"].fillna('') + " " + df["text"].fillna('')
else:
    df["content"] = df["text"].fillna('')

X = df["content"].astype(str)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    max_features=10000,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=2000, C=2.0)
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==========================
# WEBSITE HTML
# ==========================

HTML_HOME = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
</head>
<body>
    <h1>Fake News Detector</h1>
    <form action="/predict" method="POST">
        <textarea name="article" rows="12" cols="80" placeholder="Paste article here..."></textarea>
        <br><br>
        <button type="submit">Analyze</button>
    </form>
</body>
</html>
"""

HTML_RESULT = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Result</title>
</head>
<body>
    <h1>Prediction</h1>
    <h2>{{ label }}</h2>

    <h3>Your Article:</h3>
    <p>{{ text }}</p>

    <br><br>
    <a href="/">Analyze Another Article</a>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_HOME)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["article"]
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    label = "Real News" if pred == 1 else "Fake News"
    return render_template_string(HTML_RESULT, text=text, label=label)


# ==========================
# AUTO-OPEN WEBSITE ON START
# ==========================
if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    print(f"\nWebsite running at: {url}")

    # 🔥 Automatically open browser
    webbrowser.open(url)

    # Run Flask
    app.run(debug=True)

