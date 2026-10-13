"""
ml/train_all.py
---------------
Training script for MediPredict AI health risk models.
Trains on real datasets included in the project:
1. Diabetes (BRFSS 2015)
2. Heart Disease (BRFSS 2015)
3. Stroke Risk (Healthcare Stroke Dataset)
4. Symptom Checker (Dataset.csv)

Evaluates multiple algorithms with class imbalance handling, computes real metrics,
selects the optimal model based on ROC-AUC / F1, extracts feature importances, and exports models & metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_VERSION = "v1.0"


def evaluate_classifiers(X_train, X_test, y_train, y_test, feature_names):
    """
    Evaluates Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.
    Uses class_weight='balanced' to handle medical outcome class imbalance effectively.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calculate class weights for Gradient Boosting sample weights
    classes = np.unique(y_train)
    if len(classes) == 2:
        n_samples = len(y_train)
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        weight_pos = n_samples / (2.0 * n_pos) if n_pos > 0 else 1.0
        weight_neg = n_samples / (2.0 * n_neg) if n_neg > 0 else 1.0
        sample_weights = np.where(y_train == 1, weight_pos, weight_neg)
    else:
        sample_weights = None

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    }

    eval_results = {}
    best_name = None
    best_auc = -1.0
    best_model = None

    for name, clf in classifiers.items():
        if name == "Gradient Boosting" and sample_weights is not None:
            clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        else:
            clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)

        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test_scaled)[:, 1]
            auc = float(roc_auc_score(y_test, y_proba))
        else:
            auc = 0.0

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_.tolist()
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0]).tolist()
        else:
            importances = [1.0 / len(feature_names)] * len(feature_names)

        # Normalize feature importances so they sum to 1.0
        tot_imp = sum(importances) if sum(importances) > 0 else 1.0
        norm_importances = [round(val / tot_imp, 4) for val in importances]
        feat_imp_dict = dict(zip(feature_names, norm_importances))

        eval_results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
            "feature_importances": feat_imp_dict
        }

        # Select model based on ROC-AUC (standard metric for medical risk ranking)
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = clf

    return eval_results, best_name, best_model, scaler


def train_diabetes():
    print("\n--- Training Diabetes Model ---")
    path = os.path.join(DATASETS_DIR, "diabetes_012_health_indicators_BRFSS2015.csv")
    df = pd.read_csv(path)

    y = (df["Diabetes_012"] > 0).astype(int)

    feature_cols = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
        "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
        "Sex", "Age", "Education", "Income"
    ]

    X = df[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    eval_results, best_name, best_model, scaler = evaluate_classifiers(
        X_train, X_test, y_train, y_test, feature_cols
    )

    print(f"Diabetes Best Model: {best_name}")
    print(f"Metrics: {json.dumps(eval_results[best_name], indent=2)}")

    joblib.dump(best_model, os.path.join(MODELS_DIR, "diabetes_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    return {
        "dataset_name": "BRFSS 2015 Diabetes Health Indicators",
        "samples": len(df),
        "features_count": len(feature_cols),
        "feature_names": feature_cols,
        "selected_model": best_name,
        "model_version": MODEL_VERSION,
        "algorithms": eval_results
    }


def train_heart():
    print("\n--- Training Heart Disease Model ---")
    path = os.path.join(DATASETS_DIR, "heart_disease_health_indicators_BRFSS2015.csv")
    df = pd.read_csv(path)

    y = df["HeartDiseaseorAttack"].astype(int)

    feature_cols = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "Diabetes", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
        "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
        "Sex", "Age", "Education", "Income"
    ]

    X = df[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    eval_results, best_name, best_model, scaler = evaluate_classifiers(
        X_train, X_test, y_train, y_test, feature_cols
    )

    print(f"Heart Disease Best Model: {best_name}")
    print(f"Metrics: {json.dumps(eval_results[best_name], indent=2)}")

    joblib.dump(best_model, os.path.join(MODELS_DIR, "heart_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "heart_scaler.pkl"))

    return {
        "dataset_name": "BRFSS 2015 Heart Disease Health Indicators",
        "samples": len(df),
        "features_count": len(feature_cols),
        "feature_names": feature_cols,
        "selected_model": best_name,
        "model_version": MODEL_VERSION,
        "algorithms": eval_results
    }


def train_stroke():
    print("\n--- Training Stroke Model ---")
    path = os.path.join(DATASETS_DIR, "healthcare-dataset-stroke-data (1).csv")
    df = pd.read_csv(path)

    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    gender_map = {"Male": 1, "Female": 0, "Other": 0}
    df["gender_num"] = df["gender"].map(gender_map).fillna(0)

    smoke_map = {
        "never smoked": 0,
        "formerly smoked": 1,
        "smokes": 2,
        "Unknown": 3
    }
    df["smoking_num"] = df["smoking_status"].map(smoke_map).fillna(3)

    feature_cols = ["gender", "age", "hypertension", "heart_disease", "bmi", "smoking_status"]
    X = pd.DataFrame({
        "gender": df["gender_num"],
        "age": df["age"],
        "hypertension": df["hypertension"],
        "heart_disease": df["heart_disease"],
        "bmi": df["bmi"],
        "smoking_status": df["smoking_num"]
    })

    y = df["stroke"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    eval_results, best_name, best_model, scaler = evaluate_classifiers(
        X_train, X_test, y_train, y_test, feature_cols
    )

    print(f"Stroke Best Model: {best_name}")
    print(f"Metrics: {json.dumps(eval_results[best_name], indent=2)}")

    joblib.dump(best_model, os.path.join(MODELS_DIR, "stroke_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "stroke_scaler.pkl"))

    return {
        "dataset_name": "Healthcare Stroke Dataset",
        "samples": len(df),
        "features_count": len(feature_cols),
        "feature_names": feature_cols,
        "selected_model": best_name,
        "model_version": MODEL_VERSION,
        "algorithms": eval_results
    }


def train_symptom():
    print("\n--- Training Symptom Checker Model ---")
    path = os.path.join(DATASETS_DIR, "dataset.csv")
    df = pd.read_csv(path)

    symptom_cols = [c for c in df.columns if c != "Disease"]
    X = df[symptom_cols]

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Disease"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    joblib.dump(clf, os.path.join(MODELS_DIR, "symptom_model.pkl"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "symptom_label_encoder.pkl"))
    joblib.dump(symptom_cols, os.path.join(MODELS_DIR, "all_symptoms.pkl"))

    print(f"Symptom Model Accuracy: {acc:.4f}")

    return {
        "dataset_name": "Multi-Symptom Disease Dataset",
        "samples": len(df),
        "features_count": len(symptom_cols),
        "accuracy": round(acc, 4),
        "model_version": MODEL_VERSION
    }


def main():
    metrics = {
        "diabetes": train_diabetes(),
        "heart": train_heart(),
        "stroke": train_stroke(),
        "symptom": train_symptom()
    }

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining completed! Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
