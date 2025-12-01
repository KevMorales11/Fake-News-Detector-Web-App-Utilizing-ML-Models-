from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np

app = Flask(__name__, static_folder='.')

vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("fake_news_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text or not text.strip():
        return jsonify({"error": "Empty text"}), 400

    X_vec = vectorizer.transform([text])
    pred = model.predict(X_vec)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_vec)[0]
        prob = float(np.max(probs))

    label = "Real News" if int(pred) == 1 else "Fake News"
    resp = {"prediction": label}
    if prob is not None:
        resp["probability"] = prob
    return jsonify(resp)

@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
