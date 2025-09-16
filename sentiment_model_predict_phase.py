import joblib

# Load trained model and vectorizer

model = joblib.load("sentiment_model.pk1")
vectorizer = joblib.load("vectorizer.pk1")

def predict_sentiment(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return "Positive" if pred == 1 else "Negative"

# Example

print(predict_sentiment("I hated this movie, it was terrible!"))
print(predict_sentiment("Absolutely loved it, one of the best films!"))