# MediPredict AI

A Flask web app for early health risk screening of **Diabetes**, **Heart Disease** and **Stroke**, plus a multi-symptom checker. It uses machine learning models trained on real datasets (BRFSS 2015 and a healthcare stroke dataset).

> **Disclaimer:** This is an educational screening tool, not a medical diagnosis. Always consult a licensed doctor.

## Features

- Secure registration and login (bcrypt, sessions)
- Personal health profile with auto-calculated BMI
- Risk prediction (Low / Moderate / High) with a 0–100 risk score and explainable factors
- Personalized weekly exercise plan
- Dashboard with assessment history, BMI and risk trends
- Symptom checker with emergency red-flag alerts
- AI health chatbot (Google Gemini)
- Nearby hospital finder (Google Places)
- Downloadable PDF reports
- Model performance page (`/model_performance`)
- Dark / light mode

## Tech Stack

Python, Flask, SQLite, Scikit-learn, Pandas, Google Gemini API, ReportLab, Chart.js, HTML/CSS/JS

## Model Performance

| Module | Model | Accuracy | Recall | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| Diabetes | Gradient Boosting | 72.10% | 78.35% | 82.48% |
| Heart Disease | Gradient Boosting | 73.98% | 81.54% | 85.03% |
| Stroke | Logistic Regression | 73.78% | 82.00% | 83.98% |
| Symptom Checker | Random Forest | 94.44% | 94.44% | N/A |

## Project Structure

```
MediPredict_AI/
├── app.py               # Flask routes
├── prediction.py        # Model inference and explainability
├── exercise_plan.py     # Exercise recommendation rules
├── chatbot.py           # Gemini chatbot
├── report_generator.py  # PDF reports
├── create_db.py         # Database setup and migration
├── ml/train_all.py      # Model training and evaluation
├── models/              # Saved models and metrics.json
├── datasets/            # CSV datasets
├── templates/, static/  # Frontend
└── tests/               # Unit tests
```

## Setup

```bash
pip install -r requirements.txt
python create_db.py
python app.py
```

Open `http://127.0.0.1:5000`.

Create a `.env` file (do not commit it):

```
SECRET_KEY=your-secret-key
GEMINI_API_KEY=your-gemini-key
GOOGLE_PLACES_API_KEY=your-places-key
```

Optional: retrain models with `python ml/train_all.py`.

## Tests

```bash
python -m unittest tests/test_app.py
```
