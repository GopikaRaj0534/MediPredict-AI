# MediPredict AI — Comprehensive AI/ML Health Risk Screening Platform

MediPredict AI is an internship-level AI/ML health risk screening web application built with Flask, Python, Scikit-learn, SQLite, Google Gemini AI, ReportLab, and Chart.js.

It provides early statistical health risk assessments for **Diabetes**, **Heart Disease**, **Stroke**, and **Multi-Symptom Pattern Matching**, supported by empirical evaluation metrics on real medical population datasets, Explainable AI (XAI) feature weightings, personalized exercise plans, and downloadable PDF reports.

---

## Key Features

* **User Authentication & Authorization**: Secure registration, login, session security, and bcrypt password hashing.
* **Empirical Machine Learning Models**: Trained on real datasets (BRFSS 2015 & Healthcare Stroke Dataset) with cross-validation and class imbalance handling.
* **Model Performance Page (`/model_performance`)**: Transparent evaluation metrics for evaluated algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) including Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
* **Explainable AI (XAI)**: Feature contribution breakdown showing key drivers behind each model prediction.
* **Model-Derived Risk Score**: Scale (0–100) indicating calculated screening risk level alongside statistical model probability.
* **Enhanced Symptom Checker**: Ranked differential diagnosis candidates with probabilities, symptom descriptions, and precautions.
* **Safety & Emergency Layer**: Automated detection of high-risk red-flag symptoms (severe chest pain, shortness of breath, sudden weakness/numbness) triggering immediate urgent care alerts.
* **Interactive Dashboard**: Real-time user stats, Chart.js visualizations for assessment type distribution and risk level breakdown, and recent history previews.
* **Traceable Prediction History**: SQLite schema with migration support storing risk levels, confidence probabilities, risk scores, model versions (`v1.0`), and input JSON snapshots. Detail view page (`/history/<id>`) for past screenings.
* **AI Health Chatbot**: Context-aware Google Gemini assistant for health inquiries, personalized using the user's latest assessment result.
* **Personalized Exercise Plans**: Tailored weekly exercise schedules, intensity levels, and safety precautions.
* **Nearby Hospital Finder**: Google Places API integration for locating nearby hospitals with fallback Google Maps links.
* **Automated PDF Reports**: High-quality ReportLab PDF generation for saving and sharing screening results with healthcare providers.
* **Theme Customization**: Dark/Light mode toggle with persistence across sessions.

---

## Technology Stack

* **Backend**: Python 3, Flask, Flask-Bcrypt, SQLite3
* **Machine Learning**: Scikit-learn, Pandas, NumPy, Joblib
* **AI Integration**: Google Gemini API (generateContent)
* **Frontend**: HTML5, CSS3, JavaScript (ES6+), Chart.js
* **PDF Generation**: ReportLab
* **Testing**: Python `unittest`

---

## Machine Learning Workflow & Architecture

```
Real Dataset (BRFSS 2015 / Stroke CSV)
  └── Data Cleaning & Missing Value Imputation
      └── Feature Extraction & Normalization
          └── Train/Test Split (80/20 Stratified)
              └── Model Comparisons (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
                  └── Class Imbalance Handling (class_weight='balanced')
                      └── Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
                          └── Model Selection & Export (.pkl + metrics.json)
                              └── Flask Prediction Pipeline + Explainable AI (XAI)
```

---

## Datasets Used

1. **Diabetes**: `diabetes_012_health_indicators_BRFSS2015.csv` (253,680 samples, 21 health features)
2. **Heart Disease**: `heart_disease_health_indicators_BRFSS2015.csv` (253,680 samples, 21 health features)
3. **Stroke**: `healthcare-dataset-stroke-data (1).csv` (5,110 samples, 6 demographic & clinical features)
4. **Symptom Checker**: `dataset.csv` (720 samples across 15 symptoms & multi-class disease labels)

---

## Empirical Model Performance Summary

| Disease Module | Active Model | Accuracy | Recall | F1-Score | ROC-AUC | Dataset |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Diabetes** | Gradient Boosting | 72.10% | 78.35% | 46.95% | 82.48% | BRFSS 2015 Diabetes (253k) |
| **Heart Disease** | Gradient Boosting | 73.98% | 81.54% | 37.12% | 85.03% | BRFSS 2015 Heart (253k) |
| **Stroke Risk** | Logistic Regression | 73.78% | 82.00% | 23.43% | 83.98% | Healthcare Stroke Dataset (5.1k) |
| **Symptom Checker**| Random Forest | 94.44% | 94.44% | 94.44% | N/A | Multi-Symptom Dataset (720) |

*Full algorithm comparisons, precision/recall trade-offs, and confusion matrices are available interactively in the application under `/model_performance`.*

---

## Project Structure

```
MediPredict_AI/
├── app.py                  # Main Flask web application routes & controllers
├── prediction.py           # Model loading, XAI feature ranking, & inference logic
├── create_db.py             # SQLite database setup & migration script
├── report_generator.py     # ReportLab PDF report generation
├── exercise_plan.py        # Exercise recommendation rules engine
├── hospital_finder.py      # Google Places hospital search module
├── chatbot.py              # Google Gemini AI chatbot integration
├── ml/
│   └── train_all.py        # ML pipeline script: cleaning, training, evaluation & metrics export
├── models/                 # Saved scikit-learn model pickles & metrics.json
│   ├── diabetes_model.pkl, scaler.pkl
│   ├── heart_model.pkl, heart_scaler.pkl
│   ├── stroke_model.pkl, stroke_scaler.pkl
│   ├── symptom_model.pkl, symptom_label_encoder.pkl, all_symptoms.pkl
│   └── metrics.json
├── datasets/               # Real CSV datasets and reference tables
├── templates/              # Jinja2 HTML templates
├── static/                 # CSS styles, JavaScript, and asset images
├── tests/
│   └── test_app.py         # Automated unittest test suite
└── requirements.txt        # Python dependency list
```

---

## Installation & Running

### 1. Prerequisites & Environment Setup

Ensure Python 3.9+ is installed. Clone the repository and install dependencies:

```bash
cd MediPredict_AI
pip install -r requirements.txt
```

### 2. Environment Variables (Optional)

Set environment variables for production security and Gemini AI features:

```bash
# Windows PowerShell
$env:SECRET_KEY="your-secure-session-secret"
$env:GEMINI_API_KEY="your-gemini-api-key"
$env:GOOGLE_PLACES_API_KEY="your-google-places-key"
```

### 3. Initialize & Migrate Database

```bash
python create_db.py
```

### 4. Train Models & Generate Metrics (Optional / Verification)

```bash
python ml/train_all.py
```

### 5. Run the Application

```bash
python app.py
```

Navigate to `http://127.0.0.1:5000` in your web browser.

---

## Running Automated Tests

Run the automated test suite to verify authentication, predictions, database operations, ML model loading, and PDF generation:

```bash
python -m unittest tests/test_app.py
```

---

## Medical Disclaimer & Limitations

> **IMPORTANT**: MediPredict AI is an educational and screening demonstration tool. Predictions are generated by statistical machine-learning models trained on population datasets and do **NOT** constitute medical diagnoses or clinical advice. Users should always consult a licensed physician for any healthcare concerns. In case of emergency, contact local emergency services immediately.
