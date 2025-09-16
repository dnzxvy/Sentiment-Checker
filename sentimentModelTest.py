import nltk
import pandas as pd
from nltk.corpus import movie_reviews
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
import kagglehub # To download IMDB Dataset

# Download dataset
nltk.download('movie_reviews')

# Load documents (List of words, labels)

nltk_docs = []
nltk_labels = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        nltk_docs.append(movie_reviews.raw(fileid))
        nltk_labels.append(1 if category == "pos" else 0)

nltk_df = pd.DataFrame({"review": nltk_docs, "label": nltk_labels})
print("NLTK dataset size:",nltk_df.shape)

path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
print("Path to dataset files:", path)

csv_path = path + "/IMDB Dataset.csv"

imdb_dataset = pd.read_csv(csv_path)
print(imdb_dataset.head())

imdb_dataset["label"] = imdb_dataset["sentiment"].apply(lambda x: 1 if x == "positive"
                                                        else 0)
imdb_df = imdb_dataset[["review", "label"]]
print("IMDB dataset size:", imdb_df.shape)

# Combining NLTK + IMDB datasets

df = pd.concat([imdb_df, nltk_df], axis=0).reset_index(drop=True)
print("Combined dataset size:", df.shape)

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["label"], test_size=0.2, random_state=42)

# Vectorize the text ( convert words into numerical features
#with bag of words)

vectorizer = CountVectorizer(stop_words="english")




X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

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
