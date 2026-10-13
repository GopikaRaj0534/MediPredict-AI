"""
tests/test_app.py
------------------
Comprehensive unit and integration test suite for MediPredict AI.
Covers:
1. Authentication (Register, Login, Invalid Login, Logout)
2. Predictions (Diabetes, Heart Disease, Stroke, Invalid Inputs)
3. Symptom Checker & Emergency Flags
4. History & Detail Retrieval
5. ML Models & Scalers Integrity
6. Health PDF Report Generation
7. AI Health Chatbot Endpoint & Validation Tests
"""

import os
import unittest
import tempfile
import sqlite3
import json
from unittest.mock import patch
from datetime import datetime

# Configure test environment
os.environ["SECRET_KEY"] = "test-secret-key-12345"

import app
import prediction
import report_generator
import create_db
import chatbot


class MediPredictTestCase(unittest.TestCase):

    def setUp(self):
        self.db_fd, app.DB_PATH = tempfile.mkstemp()
        app.app.config["TESTING"] = True
        app.app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.app.test_client()

        # Initialize fresh test database schema
        create_db.DB_PATH = app.DB_PATH
        create_db.create_or_migrate_database()

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(app.DB_PATH):
            os.unlink(app.DB_PATH)

    # ----------------------------------------------------------------
    # 1. AUTHENTICATION TESTS
    # ----------------------------------------------------------------

    def test_user_registration_success(self):
        response = self.client.post("/register", data={
            "name": "Test User",
            "email": "testuser@example.com",
            "password": "password123"
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Registration Successful", response.data)

    def test_user_registration_duplicate_email(self):
        self.client.post("/register", data={
            "name": "Test User",
            "email": "testuser@example.com",
            "password": "password123"
        })
        response = self.client.post("/register", data={
            "name": "Test User 2",
            "email": "testuser@example.com",
            "password": "password123"
        }, follow_redirects=True)
        self.assertIn(b"Email already exists", response.data)

    def test_user_login_valid_and_invalid(self):
        self.client.post("/register", data={
            "name": "Test User",
            "email": "login@example.com",
            "password": "securepassword123"
        })

        resp_invalid = self.client.post("/login", data={
            "email": "login@example.com",
            "password": "wrongpassword"
        }, follow_redirects=True)
        self.assertIn(b"Invalid Email or Password", resp_invalid.data)

        resp_valid = self.client.post("/login", data={
            "email": "login@example.com",
            "password": "securepassword123"
        }, follow_redirects=True)
        self.assertIn(b"Login Successful", resp_valid.data)

        resp_logout = self.client.get("/logout", follow_redirects=True)
        self.assertIn(b"Logged Out Successfully", resp_logout.data)

    # ----------------------------------------------------------------
    # 2. ML MODELS & SCALERS INTEGRITY TESTS
    # ----------------------------------------------------------------

    def test_ml_models_loaded(self):
        self.assertIsNotNone(prediction.diabetes_model)
        self.assertIsNotNone(prediction.diabetes_scaler)
        self.assertIsNotNone(prediction.heart_model)
        self.assertIsNotNone(prediction.heart_scaler)
        self.assertIsNotNone(prediction.stroke_model)
        self.assertIsNotNone(prediction.stroke_scaler)
        self.assertIsNotNone(prediction.symptom_model)

    def test_diabetes_prediction_format_and_range(self):
        sample_form = {
            "HighBP": "1", "HighChol": "1", "CholCheck": "1", "BMI": "28.5",
            "Smoker": "0", "Stroke": "0", "HeartDiseaseorAttack": "0",
            "PhysActivity": "1", "Fruits": "1", "Veggies": "1",
            "HvyAlcoholConsump": "0", "AnyHealthcare": "1", "NoDocbcCost": "0",
            "GenHlth": "3", "MentHlth": "2", "PhysHlth": "1", "DiffWalk": "0",
            "Sex": "1", "Age": "57", "Education": "5", "Income": "6"
        }
        res = prediction.predict_diabetes(sample_form)
        self.assertEqual(res["disease"], "Diabetes")
        self.assertIn(res["risk"], ["High Risk", "Moderate Risk", "Low Risk"])
        self.assertTrue(0.0 <= res["probability"] <= 1.0)
        self.assertTrue(0 <= res["risk_score"] <= 100)
        self.assertGreater(len(res["contributing_factors"]), 0)

    def test_heart_prediction_format_and_range(self):
        sample_form = {
            "HighBP": "1", "HighChol": "1", "CholCheck": "1", "BMI": "31.2",
            "Smoker": "1", "Stroke": "0", "Diabetes": "1",
            "PhysActivity": "0", "Fruits": "0", "Veggies": "1",
            "HvyAlcoholConsump": "0", "AnyHealthcare": "1", "NoDocbcCost": "0",
            "GenHlth": "4", "MentHlth": "5", "PhysHlth": "4", "DiffWalk": "1",
            "Sex": "1", "Age": "67", "Education": "4", "Income": "5"
        }
        res = prediction.predict_heart(sample_form)
        self.assertEqual(res["disease"], "Heart Disease")
        self.assertIn(res["risk"], ["High Risk", "Moderate Risk", "Low Risk"])
        self.assertTrue(0.0 <= res["probability"] <= 1.0)
        self.assertTrue(0 <= res["risk_score"] <= 100)

    def test_stroke_prediction_format_and_range(self):
        sample_form = {
            "gender": "1", "age": "65", "hypertension": "1",
            "heart_disease": "0", "bmi": "29.4", "smoking_status": "2"
        }
        res = prediction.predict_stroke(sample_form)
        self.assertEqual(res["disease"], "Stroke")
        self.assertTrue(0.0 <= res["probability"] <= 1.0)

    def test_symptom_checker_emergency_flag(self):
        symptoms = ["fever", "chest_pain", "shortness_of_breath"]
        res = prediction.predict_symptom_disease(symptoms)
        self.assertTrue(res["emergency_warning"])
        self.assertGreater(len(res["possible_conditions"]), 0)

    # ----------------------------------------------------------------
    # 3. END-TO-END FLASK PREDICTION & HISTORY TESTS
    # ----------------------------------------------------------------

    def _login_test_user(self):
        self.client.post("/register", data={
            "name": "Predictor User",
            "email": "predictor@example.com",
            "password": "password123"
        })
        self.client.post("/login", data={
            "email": "predictor@example.com",
            "password": "password123"
        })

    def test_predict_diabetes_route_saves_history(self):
        self._login_test_user()
        form_data = {
            "HighBP": "1", "HighChol": "0", "CholCheck": "1", "BMI": "25.0",
            "Smoker": "0", "Stroke": "0", "HeartDiseaseorAttack": "0",
            "PhysActivity": "1", "Fruits": "1", "Veggies": "1",
            "HvyAlcoholConsump": "0", "AnyHealthcare": "1", "NoDocbcCost": "0",
            "GenHlth": "2", "MentHlth": "0", "PhysHlth": "0", "DiffWalk": "0",
            "Sex": "0", "Age": "42", "Education": "6", "Income": "8"
        }
        response = self.client.post("/predict_diabetes", data=form_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Model Health Risk Indicator", response.data)

        conn = sqlite3.connect(app.DB_PATH)
        conn.row_factory = sqlite3.Row
        rec = conn.execute("SELECT * FROM history WHERE user_email='predictor@example.com'").fetchone()
        conn.close()
        self.assertIsNotNone(rec)
        self.assertEqual(rec["prediction_type"], "Diabetes")
        self.assertEqual(rec["disease"], "Diabetes")

    def test_history_detail_view(self):
        self._login_test_user()
        form_data = {
            "gender": "1", "age": "50", "hypertension": "0",
            "heart_disease": "0", "bmi": "24.0", "smoking_status": "0"
        }
        self.client.post("/predict_stroke", data=form_data)

        resp_hist = self.client.get("/history")
        self.assertEqual(resp_hist.status_code, 200)
        self.assertIn(b"Stroke", resp_hist.data)

        conn = sqlite3.connect(app.DB_PATH)
        conn.row_factory = sqlite3.Row
        rec = conn.execute("SELECT id FROM history ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()

        resp_detail = self.client.get(f"/history/{rec['id']}")
        self.assertEqual(resp_detail.status_code, 200)
        self.assertIn(b"Assessment Input Snapshot", resp_detail.data)

    # ----------------------------------------------------------------
    # 4. REPORT & MODEL PERFORMANCE TESTS
    # ----------------------------------------------------------------

    def test_pdf_report_generation(self):
        self._login_test_user()
        form_data = {
            "gender": "1", "age": "55", "hypertension": "1",
            "heart_disease": "0", "bmi": "30.0", "smoking_status": "1"
        }
        self.client.post("/predict_stroke", data=form_data)

        resp_pdf = self.client.get("/download_report")
        self.assertEqual(resp_pdf.status_code, 200)
        self.assertEqual(resp_pdf.mimetype, "application/pdf")
        self.assertTrue(len(resp_pdf.data) > 1000)

    def test_model_performance_route(self):
        resp = self.client.get("/model_performance")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Model Evaluation & Performance", resp.data)

    # ----------------------------------------------------------------
    # 5. CHATBOT ENDPOINT & VALIDATION TESTS
    # ----------------------------------------------------------------

    def test_chatbot_unauthenticated_request(self):
        response = self.client.post("/api/chat", json={"message": "Hello"})
        self.assertEqual(response.status_code, 401)
        data = json.loads(response.data)
        self.assertFalse(data["success"])
        self.assertIn("Authentication required", data["error"])

    def test_chatbot_empty_message_validation(self):
        self._login_test_user()
        response = self.client.post("/api/chat", json={"message": "   "})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data["success"])
        self.assertIn("valid message", data["error"])

    def test_chatbot_message_too_long_validation(self):
        self._login_test_user()
        long_msg = "A" * 1005
        response = self.client.post("/api/chat", json={"message": long_msg})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data["success"])
        self.assertIn("too long", data["error"])

    def test_chatbot_unconfigured_api_key_returns_error(self):
        self._login_test_user()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=True):
            response = self.client.post("/api/chat", json={"message": "Hello"})
            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertFalse(data["success"])
            self.assertIn("Gemini API is not configured", data["error"])

    @patch("chatbot.ask_gemini")
    def test_chatbot_successful_response_with_assessment_context(self, mock_ask_gemini):
        mock_ask_gemini.return_value = ("Diabetes risk refers to the likelihood of developing high blood sugar.", None)
        self._login_test_user()

        # Run a prediction first to create assessment context in DB/session
        form_data = {
            "HighBP": "1", "HighChol": "1", "CholCheck": "1", "BMI": "30.0",
            "Smoker": "0", "Stroke": "0", "HeartDiseaseorAttack": "0",
            "PhysActivity": "0", "Fruits": "1", "Veggies": "1",
            "HvyAlcoholConsump": "0", "AnyHealthcare": "1", "NoDocbcCost": "0",
            "GenHlth": "3", "MentHlth": "0", "PhysHlth": "0", "DiffWalk": "0",
            "Sex": "1", "Age": "57", "Education": "5", "Income": "6"
        }
        self.client.post("/predict_diabetes", data=form_data)

        response = self.client.post("/api/chat", json={"message": "What does my diabetes result mean?"})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data["success"])
        self.assertIn("Diabetes risk refers to", data["response"])
        mock_ask_gemini.assert_called_once()


if __name__ == "__main__":
    unittest.main()
