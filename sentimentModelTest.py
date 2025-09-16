import nltk
import pandas as pd
from nltk.corpus import movie_reviews
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib # To save trained model
import kagglehub # To download IMDB Dataset
from sklearn.metrics import accuracy_score

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

# Downloading IMDB dataset
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
print("Path to dataset files:", path)

# Loading IMDB dataset
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

vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)



pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
joblib.dump(model, "sentiment_model.pk1")
joblib.dump(vectorizer, "vectorizer.pk1")


def predict_sentiment(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return "Positive" if pred == 1 else "Negative"

print(predict_sentiment("I really love this!"))
print(predict_sentiment("This is boring and terrible."))
