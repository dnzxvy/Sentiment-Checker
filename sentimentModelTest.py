import nltk
from nltk.corpus import movie_reviews
import random

# Download dataset
nltk.download('movie_reviews')

# Load documents (List of words, labels)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

#prepare texts and labels
texts = [" ".join(words) for words, label in documents]
labels = [1 if label == 'pos' else 0 for words, label in documents]


# Step 2: vectorize the text ( convert words into numerical features
#with bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

# Step 3 train test split. Split training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

""""from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

new_texts = [
    "This movie was absolutely wonderful, I loved it!",
    "Worst film ever, don’t waste your time.",
    "Mediocre story but some good acting.",
    "I hated the film, but the soundtrack was great."
]

new_X = vectorizer.transform(new_texts)
print("Custom Predictions:", model.predict(new_X))"""

import joblib
joblib.dump(model, "sentiment_model.pk1")
joblib.dump(vectorizer, "vectorizer.pk1")
