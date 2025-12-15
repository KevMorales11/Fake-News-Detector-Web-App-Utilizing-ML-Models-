import os
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, '..', 'templates')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


try:
    vectorizer = joblib.load(os.path.join(BASE_DIR, 'vectorizer.joblib'))
    model = joblib.load(os.path.join(BASE_DIR, 'fake_news_model.joblib'))
except Exception as e:
    print("Error loading model files. Please run train.py first.")
    raise e

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    text = ''
    if ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    elif ext == 'pdf':
        try:
            reader = PyPDF2.PdfReader(filepath)
            text = ' '.join([page.extract_text() or '' for page in reader.pages])
        except Exception:
            text = ''
    elif ext in ('doc', 'docx'):
        try:
            doc = docx.Document(filepath)
            text = ' '.join([p.text for p in doc.paragraphs])
        except Exception:
            text = ''
    return text

def predict_text(text):
    X_vec = vectorizer.transform([text])
    pred = model.predict(X_vec)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(np.max(model.predict_proba(X_vec)[0]))
    label = "Real News" if int(pred) == 1 else "Fake News"
    resp = {"prediction": label}
    if prob is not None:
        resp["probability"] = prob
    return resp

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400
    return jsonify(predict_text(text))

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    text = extract_text_from_file(filepath)
    if not text.strip():
        return jsonify({"error": "No readable text in file"}), 400

    return jsonify(predict_text(text))

if __name__ == "__main__":
    # Automatically opens browser
    webbrowser.open("http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=True)
