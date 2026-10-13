"""
train_dummy_models.py
----------------------
BONUS SCRIPT -- not part of the original file list, but included so
the site works the moment you clone it, before you plug in your own
real trained models and datasets.

It generates:
  - datasets/dataset.csv                (synthetic symptom data)
  - datasets/symptom_Description.csv
  - datasets/symptom_precaution.csv
  - datasets/Symptom-severity.csv
  - models/*.pkl                        (small RandomForest models
                                          trained on synthetic data)

Replace these with your real datasets/models later -- app.py and
prediction.py don't need to change, as long as file names and the
feature order documented in prediction.py stay the same.

Usage:
    python train_dummy_models.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

rng = np.random.default_rng(42)


# ==================================================================
# 1. SYMPTOM CHECKER MODEL
# ==================================================================
SYMPTOMS = [
    "fever", "cough", "fatigue", "headache", "vomiting", "nausea",
    "chest_pain", "shortness_of_breath", "joint_pain", "skin_rash",
    "abdominal_pain", "dizziness", "sore_throat", "chills", "sweating",
]

DISEASES = ["Common Cold", "Allergy", "Migraine", "Diabetes",
            "Heart Disease", "Gastroenteritis"]

DISEASE_SYMPTOM_MAP = {
    "Common Cold": ["fever", "cough", "sore_throat", "fatigue"],
    "Allergy": ["skin_rash", "sore_throat", "cough"],
    "Migraine": ["headache", "dizziness", "nausea"],
    "Diabetes": ["fatigue", "dizziness", "sweating"],
    "Heart Disease": ["chest_pain", "shortness_of_breath", "sweating", "dizziness"],
    "Gastroenteritis": ["vomiting", "nausea", "abdominal_pain", "fever"],
}

rows = []
for disease, core_symptoms in DISEASE_SYMPTOM_MAP.items():
    for _ in range(120):
        row = {s: 0 for s in SYMPTOMS}
        for s in core_symptoms:
            if rng.random() < 0.85:
                row[s] = 1
        # small chance of a random extra symptom (noise)
        for s in SYMPTOMS:
            if rng.random() < 0.05:
                row[s] = 1
        row["Disease"] = disease
        rows.append(row)

symptom_df = pd.DataFrame(rows)
symptom_df.to_csv(os.path.join(DATASETS_DIR, "dataset.csv"), index=False)

encoder = LabelEncoder()
y = encoder.fit_transform(symptom_df["Disease"])
X = symptom_df[SYMPTOMS]

symptom_model = RandomForestClassifier(n_estimators=150, random_state=42)
symptom_model.fit(X, y)

joblib.dump(symptom_model, os.path.join(MODELS_DIR, "symptom_model.pkl"))
joblib.dump(encoder, os.path.join(MODELS_DIR, "symptom_label_encoder.pkl"))
joblib.dump(SYMPTOMS, os.path.join(MODELS_DIR, "all_symptoms.pkl"))

# ---- reference tables -------------------------------------------------
description_df = pd.DataFrame({
    "Disease": DISEASES,
    "Description": [
        "A mild viral infection of the nose and throat.",
        "An immune reaction to a normally harmless substance.",
        "A neurological condition causing intense, recurring headaches.",
        "A metabolic condition marked by high blood sugar levels.",
        "A range of conditions affecting the heart's structure or function.",
        "Inflammation of the stomach and intestines, often from infection.",
    ],
})
description_df.to_csv(os.path.join(DATASETS_DIR, "symptom_Description.csv"), index=False)

precaution_df = pd.DataFrame({
    "Disease": DISEASES,
    "Precaution_1": ["Rest well", "Avoid the allergen", "Rest in a dark room",
                     "Monitor blood sugar", "Seek medical attention", "Stay hydrated"],
    "Precaution_2": ["Stay hydrated", "Take antihistamines", "Stay hydrated",
                     "Eat a balanced diet", "Avoid strenuous activity", "Eat bland foods"],
    "Precaution_3": ["Use over-the-counter relief", "See an allergist", "Avoid bright light",
                     "Exercise regularly", "Take prescribed medication", "Rest"],
    "Precaution_4": ["See a doctor if symptoms persist", "Carry emergency medication if severe",
                     "See a neurologist if frequent", "Consult a doctor regularly",
                     "Follow up with a cardiologist", "See a doctor if symptoms worsen"],
})
precaution_df.to_csv(os.path.join(DATASETS_DIR, "symptom_precaution.csv"), index=False)

severity_df = pd.DataFrame({
    "Symptom": SYMPTOMS,
    "Weight": rng.integers(1, 7, size=len(SYMPTOMS)),
})
severity_df.to_csv(os.path.join(DATASETS_DIR, "Symptom-severity.csv"), index=False)


# ==================================================================
# Helper to build + save a simple binary classifier
# ==================================================================
def make_binary_model(n_features, filename, informative_idx):
    """Creates a small, plausible synthetic binary-classification dataset
    and trains a RandomForest, so the demo produces sensible-looking,
    non-random predictions (e.g. higher glucose -> higher diabetes risk)."""
    n_samples = 600
    X = rng.normal(loc=0.5, scale=0.25, size=(n_samples, n_features))
    X = np.clip(X, 0, 1)

    # risk score driven mainly by the "informative" columns
    risk_score = X[:, informative_idx].mean(axis=1)
    y = (risk_score + rng.normal(0, 0.08, n_samples) > 0.55).astype(int)

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    joblib.dump(model, os.path.join(MODELS_DIR, filename))
    return model


# 2. Diabetes: pregnancies, glucose, bp, skin, insulin, bmi, dpf, age (8 features)
make_binary_model(8, "diabetes_model.pkl", informative_idx=[1, 5, 6])

# 3. Heart: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
#           oldpeak, slope, ca, thal (13 features)
make_binary_model(13, "heart_model.pkl", informative_idx=[2, 4, 8, 9, 11])

# 4. Kidney: 14 features
make_binary_model(14, "kidney_model.pkl", informative_idx=[6, 7, 10])

# 5. Liver: 10 features
make_binary_model(10, "liver_model.pkl", informative_idx=[2, 3, 5, 6])

# 6. Stroke: gender, age, hypertension, heart_disease, ever_married,
#            work_type, residence_type, avg_glucose_level, bmi, smoking (10 features)
make_binary_model(10, "stroke_model.pkl", informative_idx=[1, 2, 7])

print("Done. Synthetic datasets written to datasets/, demo models written to models/.")
print("Swap these out with your real trained models whenever they're ready --")
print("prediction.py and app.py will keep working as long as filenames match.")
