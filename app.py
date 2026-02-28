from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text
app = Flask(__name__)
CORS(app)

# Load vectorizer + model
vectorizer, model = joblib.load("spam_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    text = data["text"]        # FIRST get text from request
    text = clean_text(text)    # THEN clean it

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    result = "spam" if prediction == 1 else "ham"

    return jsonify({"result": result})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)