from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Loading the model/vectorizer

model = joblib.load("sentiment_model.pk1")
vectorizer = joblib.load("vectorizer.pk1")

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Transform text into vector
    X = vectorizer.transform([text])

    # Predicting sentiment using trained model
    pred = model.predict(X)[0]

    sentiment = "Positive" if pred == 1 else "Negative"

    # Sending results back as json
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(port=5000)