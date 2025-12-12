import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Load CSVs
true_df = pd.read_csv('C:\\Users\Derrick\OneDrive\Documents\Fake-news-web\data\True.csv')
fake_df = pd.read_csv('\\Users\Derrick\OneDrive\Documents\Fake-news-web\data\Fake.csv')

# 2. Add labels
true_df['label'] = 1  # Real
fake_df['label'] = 0  # Fake

# 3. Balance datasets
min_len = min(len(true_df), len(fake_df))
true_df = true_df.sample(min_len, random_state=42)
fake_df = fake_df.sample(min_len, random_state=42)

# 4. Combine
df = pd.concat([true_df, fake_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Combine title+text
if 'title' in df.columns:
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
else:
    df['content'] = df['text'].fillna('')

X = df['content']
y = df['label']

# 6. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 7. TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=10000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 8. Train model
model = LogisticRegression(max_iter=2000, C=2.0)
model.fit(X_train_vec, y_train)

# 9. Save artifacts
joblib.dump(vectorizer, 'backend/vectorizer.joblib')
joblib.dump(model, 'backend/fake_news_model.joblib')

print("Model training complete and saved in backend folder!")
