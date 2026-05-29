import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import time

print("Starting retraining process...")
start_time = time.time()

# 1. Download stopwords
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(content):
    if not isinstance(content, str):
        content = str(content)
    # Remove non-alphabet
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    # Stemming and stopword removal
    content = [ps.stem(word) for word in content if word not in stop_words]
    return ' '.join(content)

# 2. Load dataset
print("Loading dataset...")
df = pd.read_csv('fake_news.csv')
print(f"Dataset loaded with {len(df)} rows.")

# Handle missing values
df = df.fillna('')

# 3. Stemming on 'text' instead of 'author + title'
# We will use 'text' column for training so it matches the website's expected input
print("Applying stemming to the 'text' column... (This might take a few minutes)")
df['stemmed_text'] = df['text'].apply(stemming)
print("Stemming complete!")

X = df['stemmed_text'].values
y = df['label'].values

# 4. Vectorization
print("Vectorizing...")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# 5. Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 6. Model Training
print("Training model...")
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Testing Accuracy on 'text': {acc:.4f}")

# 8. Save models
print("Saving new .pkl files...")
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"Done in {time.time() - start_time:.2f} seconds!")

# 9. Verify on unseen custom data
print("\n--- Testing on Unseen Data ---")
test_sentences = [
    "NASA has officially launched a new rover to Mars to search for signs of ancient life.",
    "Breaking: Scientists confirm that the Earth is actually flat and gravity is an illusion created by the government.",
    "The stock market saw a significant increase today following the Federal Reserve's decision to lower interest rates.",
    "Shocking! Aliens have infiltrated the White House and are currently replacing all senators with clones."
]

stemmed_test_sentences = [stemming(s) for s in test_sentences]
transformed_test = vectorizer.transform(stemmed_test_sentences)
test_preds = model.predict(transformed_test)

for s, p in zip(test_sentences, test_preds):
    label = "Fake" if p == 1 else "Real"
    print(f"[{label}] {s}")
