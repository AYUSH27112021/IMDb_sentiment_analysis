# Sentiment Analysis on IMDB Reviews  

This repository contains a **sentiment analysis pipeline** trained on the **IMDB movie reviews dataset** using multiple models, including **Linear SVM and BERT**. The project includes **data processing, model training, and a Flask web app with a UI** to showcase predictions.  

## 🚀 Features  
- **Dataset**: IMDB Movie Reviews (~25k labeled reviews)  
- **Models**:  
  - **Linear SVM** (Best ML model - Accuracy: 89.87%)  
  - **BERT Transformer** (Deep Learning model - Accuracy: 83%)  
  - **Multinomial Naïve Bayes** (Baseline ML model - Accuracy: 86.64%)  
- **TF-IDF Vectorizer** (`tfidf_vectorizer.joblib`) for feature extraction in ML models  
- **Flask API** (`app.py`) to serve predictions  
- **UI** to visualize model outputs  
- **Data processing script** (`data_setup.py`) for database loading  
- **Training notebook** (`train_model.ipynb`)  
- **Sample video & screenshots** of the app in action  

---

## 📌 Setup Instructions  

### 1️⃣ Initialize Virtual Environment  
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 2️⃣ Load & Process Data

Run the data_setup.py script to load and preprocess the IMDB dataset into a database.

```bash
python data_setup.py
```
### 3️⃣ Train Models
Model training can be done using `train_model.ipynb`. This notebook contains code for training Linear SVM, Naïve Bayes, and BERT on the IMDB dataset.

### 4️⃣ Run Flask API
Launch the Flask API to serve predictions from Linear SVM and BERT models.
```bash
python app.py
```
### 📂 Repository Structure
```php
├── Database_Files        # Stores a .db file used by created using `data_setup.py` 
├── static/               # UI asset
├── templates/            # Flask UI templates
├── Trained_models/       # Saved models in .joblib and .pt format
├── app.py                # Flask app serving Linear SVM & BERT
├── BERT_model.py         # BERT architecture
├── data_setup.py         # Loads IMDB dataset into a database
├── train_model.ipynb     # Notebook for training models
├── requirements.txt      # Python dependencies
├── .gitignore            
└── README.md             
```

### 5️⃣ Model Serving with Flask
Flask API
A simple Flask app (app.py) with an endpoint:

### 🎯 Additional Notes
All trained models (.pt, .joblib) are available in the Release Section.

### Sample video

https://github.com/user-attachments/assets/5778c1e4-64f6-4e10-873d-cb2cc6413a4d

