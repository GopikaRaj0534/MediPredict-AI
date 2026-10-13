"""
prediction.py
-------------
Prediction logic and Explainable AI pipeline for MediPredict AI.
Handles model loading, preprocessor synchronization, prediction, risk scoring,
feature importance extraction, and safety/emergency symptom detection.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


def _load(name):
    path = os.path.join(MODELS_DIR, name)
    return joblib.load(path) if os.path.exists(path) else None


# Models & Scalers
symptom_model = _load("symptom_model.pkl")
symptom_label_encoder = _load("symptom_label_encoder.pkl")
all_symptoms = _load("all_symptoms.pkl") or []

diabetes_model = _load("diabetes_model.pkl")
diabetes_scaler = _load("scaler.pkl")

heart_model = _load("heart_model.pkl")
heart_scaler = _load("heart_scaler.pkl")

stroke_model = _load("stroke_model.pkl")
stroke_scaler = _load("stroke_scaler.pkl")

# Load evaluation metrics if available
metrics_data = {}
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
    except Exception:
        metrics_data = {}

# Reference descriptions & precautions
try:
    description_df = pd.read_csv(os.path.join(DATASETS_DIR, "symptom_Description.csv"))
except Exception:
    description_df = pd.DataFrame(columns=["Disease", "Description"])

try:
    precaution_df = pd.read_csv(os.path.join(DATASETS_DIR, "symptom_precaution.csv"))
except Exception:
    precaution_df = pd.DataFrame()

# Emergency symptoms requiring urgent medical warning
EMERGENCY_SYMPTOMS = {
    "chest_pain", "shortness_of_breath", "severe_chest_pain",
    "sudden_weakness", "sudden_numbness", "unconsciousness",
    "severe_breathing_difficulty"
}

FEATURE_LABELS = {
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "CholCheck": "Cholesterol Check (last 5 yrs)",
    "BMI": "Body Mass Index (BMI)",
    "bmi": "Body Mass Index (BMI)",
    "Smoker": "Smoking History",
    "Stroke": "Prior Stroke",
    "HeartDiseaseorAttack": "Heart Disease / Heart Attack",
    "Diabetes": "Diabetes History",
    "PhysActivity": "Physical Activity",
    "Fruits": "Fruit Intake",
    "Veggies": "Vegetable Intake",
    "HvyAlcoholConsump": "Heavy Alcohol Consumption",
    "AnyHealthcare": "Healthcare Coverage",
    "NoDocbcCost": "Skipped Doctor Due to Cost",
    "GenHlth": "General Health Rating",
    "MentHlth": "Poor Mental Health Days",
    "PhysHlth": "Poor Physical Health Days",
    "DiffWalk": "Difficulty Walking",
    "Sex": "Sex",
    "Age": "Age Group",
    "age": "Age (Years)",
    "Education": "Education Level",
    "Income": "Income Level",
    "gender": "Gender",
    "hypertension": "Hypertension",
    "heart_disease": "Heart Disease History",
    "smoking_status": "Smoking Status"
}


def _lookup_description(d):
    try:
        return description_df.loc[description_df["Disease"].str.strip() == d.strip(), "Description"].iloc[0]
    except Exception:
        return "No specific description available."


def _lookup_precautions(d):
    try:
        row = precaution_df.loc[precaution_df["Disease"].str.strip() == d.strip()].iloc[0]
        return [row[c] for c in precaution_df.columns if c.startswith("Precaution") and pd.notna(row[c])]
    except Exception:
        return []


def _extract_top_contributing_factors(feature_names, raw_inputs, model_key):
    """
    Computes XAI feature importance contribution ranking for a user's input.
    """
    feat_imps = {}
    if model_key in metrics_data:
        sel_name = metrics_data[model_key].get("selected_model")
        algos = metrics_data[model_key].get("algorithms", {})
        if sel_name in algos:
            feat_imps = algos[sel_name].get("feature_importances", {})

    contributions = []
    for idx, name in enumerate(feature_names):
        val = float(raw_inputs[idx])
        base_imp = feat_imps.get(name, 0.05)
        # Weight contribution higher if the factor is elevated (e.g. HighBP=1, High BMI, etc.)
        score = base_imp * (1.0 + abs(val))
        label = FEATURE_LABELS.get(name, name)
        
        impact = "High Impact" if score > 0.15 else ("Moderate Impact" if score > 0.05 else "Low Impact")
        contributions.append({
            "feature": name,
            "label": label,
            "val": val,
            "score": round(score, 4),
            "impact": impact
        })

    # Sort descending by contribution score
    contributions.sort(key=lambda x: x["score"], reverse=True)
    return contributions[:5]


def predict_symptom_disease(selected):
    """
    Symptom Checker prediction with differential diagnoses and emergency detection.
    """
    emergency_flag = False
    for s in selected:
        s_clean = s.lower().replace(" ", "_")
        if s_clean in EMERGENCY_SYMPTOMS:
            emergency_flag = True
            break

    if not symptom_model or not symptom_label_encoder or not all_symptoms:
        return {
            "disease": "Unknown",
            "possible_conditions": [],
            "description": "Symptom checker model not loaded.",
            "precautions": [],
            "confidence": 0.0,
            "emergency_warning": emergency_flag
        }

    vec = [1 if s in selected else 0 for s in all_symptoms]
    x = pd.DataFrame([vec], columns=all_symptoms)

    if hasattr(symptom_model, "predict_proba"):
        probas = symptom_model.predict_proba(x)[0]
        top_indices = np.argsort(probas)[::-1][:3]
        top_diseases = symptom_label_encoder.inverse_transform(top_indices)
        top_confs = probas[top_indices]

        possible_conditions = []
        for d, c in zip(top_diseases, top_confs):
            possible_conditions.append({
                "disease": d,
                "confidence": round(float(c) * 100, 1),
                "description": _lookup_description(d),
                "precautions": _lookup_precautions(d)
            })

        primary = possible_conditions[0]
        disease_name = primary["disease"]
        conf = primary["confidence"]
    else:
        pred = symptom_model.predict(x)[0]
        disease_name = symptom_label_encoder.inverse_transform([pred])[0]
        conf = 100.0
        possible_conditions = [{
            "disease": disease_name,
            "confidence": conf,
            "description": _lookup_description(disease_name),
            "precautions": _lookup_precautions(disease_name)
        }]

    return {
        "disease": disease_name,
        "possible_conditions": possible_conditions,
        "description": _lookup_description(disease_name),
        "precautions": _lookup_precautions(disease_name),
        "confidence": conf,
        "emergency_warning": emergency_flag,
        "model_version": "v1.0"
    }


def age_to_brfss_category(age):
    """
    Converts real age in years to BRFSS age category (1-13).
    Mapping:
      18-24 -> 1
      25-29 -> 2
      30-34 -> 3
      35-39 -> 4
      40-44 -> 5
      45-49 -> 6
      50-54 -> 7
      55-59 -> 8
      60-64 -> 9
      65-69 -> 10
      70-74 -> 11
      75-79 -> 12
      80 and above -> 13
    """
    if age is None or (isinstance(age, str) and age.strip() == ""):
        raise ValueError("Age is required and cannot be empty.")
    try:
        val = float(age)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid age: {age}. Age must be a number.")

    if val < 18 or val > 120:
        raise ValueError(f"Age {val} is outside supported adult screening range (18 to 120).")

    if val <= 24:
        return 1
    elif val <= 29:
        return 2
    elif val <= 34:
        return 3
    elif val <= 39:
        return 4
    elif val <= 44:
        return 5
    elif val <= 49:
        return 6
    elif val <= 54:
        return 7
    elif val <= 59:
        return 8
    elif val <= 64:
        return 9
    elif val <= 69:
        return 10
    elif val <= 74:
        return 11
    elif val <= 79:
        return 12
    else:
        return 13


def classify_risk(val):
    """
    Centralized application-defined screening risk classification.
    Thresholds:
      - Low Risk: score below 20% (< 20)
      - Moderate Risk: score from 20% up to, but not including, 40% (20 <= score < 40)
      - High Risk: score of 40% or higher (score >= 40)

    Returns:
        tuple: (risk_level, result_text)
    """
    if val is None:
        raise ValueError("Risk value cannot be None.")
    try:
        num = float(val)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid risk value: {val}. Must be numeric.")

    # Scale probability [0.0, 1.0] to [0.0, 100.0] if passed as fractional probability
    if 0.0 < num < 1.0 and isinstance(val, float):
        score = num * 100.0
    else:
        score = num

    if score < 0.0 or score > 100.0:
        raise ValueError(f"Risk score {val} is outside valid range (0 to 100).")

    if score < 20.0:
        return "Low Risk", "Low Risk Detected"
    elif score < 40.0:
        return "Moderate Risk", "Moderate Risk Detected"
    else:
        return "High Risk", "Elevated Risk Detected"


def _format_prediction(disease_name, proba, raw_inputs, feature_names, model_key, real_age=None):
    risk_score = int(round(proba * 100))
    risk_level, result_text = classify_risk(risk_score)

    contributing_factors = _extract_top_contributing_factors(feature_names, raw_inputs, model_key)
    model_version = metrics_data.get(model_key, {}).get("model_version", "v1.0")
    algorithm_used = metrics_data.get(model_key, {}).get("selected_model", "Machine Learning Model")

    return {
        "disease": disease_name,
        "result": result_text,
        "risk": risk_level,
        "risk_score": risk_score,
        "confidence": round(proba * 100, 1),
        "probability": round(proba, 4),
        "real_age": real_age,
        "contributing_factors": contributing_factors,
        "model_version": model_version,
        "algorithm_used": algorithm_used,
        "disclaimer": "This result is an AI-based risk screening estimate and is NOT a clinical diagnosis."
    }


def predict_diabetes(form):
    feature_cols = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
        "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
        "Sex", "Age", "Education", "Income"
    ]
    raw_age = form.get("Age") if "Age" in form else form.get("age")
    brfss_age = age_to_brfss_category(raw_age)
    real_age = float(raw_age)

    raw_vals = []
    for k in feature_cols:
        if k == "Age":
            raw_vals.append(float(brfss_age))
        else:
            raw_vals.append(float(form[k]))

    feats = np.array([raw_vals])
    
    if diabetes_scaler:
        feats_scaled = diabetes_scaler.transform(feats)
    else:
        feats_scaled = feats

    if hasattr(diabetes_model, "predict_proba"):
        proba = float(diabetes_model.predict_proba(feats_scaled)[0][1])
    else:
        proba = float(diabetes_model.predict(feats_scaled)[0])

    return _format_prediction("Diabetes", proba, raw_vals, feature_cols, "diabetes", real_age=real_age)


def predict_heart(form):
    feature_cols = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "Diabetes", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
        "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
        "Sex", "Age", "Education", "Income"
    ]
    raw_age = form.get("Age") if "Age" in form else form.get("age")
    brfss_age = age_to_brfss_category(raw_age)
    real_age = float(raw_age)

    raw_vals = []
    for k in feature_cols:
        if k == "Age":
            raw_vals.append(float(brfss_age))
        else:
            raw_vals.append(float(form[k]))

    feats = np.array([raw_vals])

    if heart_scaler:
        feats_scaled = heart_scaler.transform(feats)
    else:
        feats_scaled = feats

    if hasattr(heart_model, "predict_proba"):
        proba = float(heart_model.predict_proba(feats_scaled)[0][1])
    else:
        proba = float(heart_model.predict(feats_scaled)[0])

    return _format_prediction("Heart Disease", proba, raw_vals, feature_cols, "heart", real_age=real_age)


def predict_stroke(form):
    feature_cols = ["gender", "age", "hypertension", "heart_disease", "bmi", "smoking_status"]
    raw_age = form.get("age") if "age" in form else form.get("Age")
    if raw_age is None or (isinstance(raw_age, str) and raw_age.strip() == ""):
        raise ValueError("Age is required and cannot be empty.")
    try:
        real_age = float(raw_age)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid age: {raw_age}. Age must be a number.")

    if real_age < 1 or real_age > 120:
        raise ValueError(f"Age {real_age} is outside supported screening range (1 to 120).")

    raw_vals = []
    for k in feature_cols:
        if k == "age":
            raw_vals.append(real_age)
        else:
            raw_vals.append(float(form[k]))

    feats = np.array([raw_vals])

    if stroke_scaler:
        feats_scaled = stroke_scaler.transform(feats)
    else:
        feats_scaled = feats

    if hasattr(stroke_model, "predict_proba"):
        proba = float(stroke_model.predict_proba(feats_scaled)[0][1])
    else:
        proba = float(stroke_model.predict(feats_scaled)[0])

    return _format_prediction("Stroke", proba, raw_vals, feature_cols, "stroke", real_age=real_age)

