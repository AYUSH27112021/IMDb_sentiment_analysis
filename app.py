import joblib
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer ,AutoModel
from BERT_model import BERT_Arch

device = torch.device("cpu")

app = Flask(__name__ ,static_folder='static')
# Load ML model (TF-IDF + Linear SVC)
loaded_tfidf = joblib.load('tfidf_vectorizer.joblib')
loaded_ml_model = joblib.load('linear_svc_model.joblib')

# Function to preprocess text for DL model
def preprocess_text(text, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    return encoding["input_ids"], encoding["attention_mask"]

def predict_text(text, model):
    """Preprocesses input text, runs model inference, and returns prediction."""
    input_ids, attention_mask = preprocess_text(text)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds[0]

# Function for ML model prediction
def predict_ml(text):
    text_tfidf = loaded_tfidf.transform([text])
    prediction = loaded_ml_model.predict(text_tfidf)
    if(prediction[0] == 'positive'):
        return "Positive"
    else:
        return "Negative"

# Function for DL model prediction
def predict_dl(text):
    model_path = 'saved_weights.pt'
    bert = AutoModel.from_pretrained('bert-base-uncased')
    model = BERT_Arch(bert)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    prediction = predict_text(text, model)
    return "positive" if prediction == 1 else "negative"

# API Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get("review_text", "")

        if not text:
            return jsonify({"error": "No review_text provided"}), 400

        model_type = data.get("model_type", "ml")  # Default to ML model
        if model_type == "ml":
            sentiment = predict_ml(text)
        elif model_type == "dl":
            sentiment = predict_dl(text)
        else:
            return jsonify({"error": "Invalid model_type. Choose 'ml' or 'dl'"}), 400

        return jsonify({"sentiment_prediction": sentiment})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Web UI to select models
@app.route('/')
def home():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
