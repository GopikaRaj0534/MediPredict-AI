"""
app.py
------
Main Flask application for MediPredict AI.

Supports:
- User Authentication (Bcrypt hashing)
- Interactive Dashboard with Analytics
- Symptom Checker (Differential diagnosis & safety alerts)
- Diabetes Prediction
- Heart Disease Prediction
- Stroke Prediction
- Explainable AI (XAI) & Risk Score Indicators
- Model Performance & Metrics Transparency Page
- Traceable Prediction History with Detail Views
- AI Health Chatbot (Google Gemini Integration)
- Personalized Exercise Plans
- Hospital Finder
- Automated PDF Health Assessment Report
"""

import os
import json
import sqlite3
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request,
    redirect, url_for, session, flash,
    send_file, jsonify
)

from flask_bcrypt import Bcrypt
import config
import prediction

# Feature modules
import report_generator
import exercise_plan as exercise_plan_module
import hospital_finder
import chatbot

# ----------------------------------------------------
# Flask Configuration
# ----------------------------------------------------

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

bcrypt = Bcrypt(app)

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "database.db")


# ----------------------------------------------------
# Lifestyle Recommendations
# ----------------------------------------------------

LIFESTYLE_TIPS = {
    ("Diabetes", "High Risk"): [
        ("🥗", "Cut back on refined sugar and processed carbs — swap soda/sweets for water and whole fruit."),
        ("🚶", "Aim for at least 30 minutes of walking or light activity most days."),
        ("🩺", "Book a fasting blood glucose or HbA1c test with a doctor to confirm this result."),
        ("⚖️", "Even a 5-7% reduction in body weight meaningfully lowers diabetes risk."),
    ],
    ("Diabetes", "Low Risk"): [
        ("🥗", "Keep up balanced meals with vegetables, whole grains, and lean protein."),
        ("🚶", "Regular movement helps keep this risk low — no special routine required."),
        ("🩺", "A yearly check-up is still a good habit, even with a low-risk result."),
    ],
    ("Heart Disease", "High Risk"): [
        ("🧂", "Reduce salt and saturated fat — they're key dietary drivers of heart risk."),
        ("🚭", "If you smoke, quitting is the single most impactful change you can make."),
        ("🩺", "Ask a doctor about a blood pressure and cholesterol panel to confirm this result."),
        ("🧘", "Chronic stress and poor sleep raise cardiovascular risk — prioritize both."),
    ],
    ("Heart Disease", "Low Risk"): [
        ("🏃", "Keep up regular activity — it's one of the strongest protective factors for the heart."),
        ("🧂", "Watching sodium intake now helps keep blood pressure in a healthy range long-term."),
        ("🩺", "Routine check-ups still matter, even with a low-risk result."),
    ],
    ("Stroke", "High Risk"): [
        ("🩸", "Uncontrolled blood pressure is a top stroke risk factor — get it checked."),
        ("🚭", "Smoking roughly doubles stroke risk; quitting lowers it significantly over time."),
        ("🧠", "Learn the F.A.S.T. warning signs (Face drooping, Arm weakness, Speech difficulty, Time to call emergency services)."),
        ("🩺", "Discuss this result with a doctor — stroke risk factors are highly manageable when caught early."),
    ],
    ("Stroke", "Low Risk"): [
        ("🩸", "Keep monitoring blood pressure periodically, even without symptoms."),
        ("🚶", "Regular activity and a balanced diet help keep stroke risk low over time."),
    ],
}


def get_lifestyle_tips(disease, risk):
    return LIFESTYLE_TIPS.get((disease, risk), [])


BRFSS_AGE_MIDPOINT = {
    1: 21, 2: 27, 3: 32, 4: 37, 5: 42, 6: 47, 7: 52,
    8: 57, 9: 62, 10: 67, 11: 72, 12: 77, 13: 80,
}


def _real_age_years(form, brfss_code_field="Age", direct_field="age"):
    if direct_field in form:
        try:
            return float(form[direct_field])
        except (TypeError, ValueError):
            return None
    if brfss_code_field in form:
        try:
            return BRFSS_AGE_MIDPOINT.get(int(float(form[brfss_code_field])))
        except (TypeError, ValueError):
            return None
    return None


def store_last_prediction(prediction_type, result, inputs=None, description=None,
                           precautions=None, plan=None):
    session["last_prediction"] = {
        "patient_name": session.get("user_name"),
        "generated_at": datetime.now().isoformat(),
        "prediction_type": prediction_type,
        "disease": result.get("disease"),
        "result": result.get("result"),
        "risk": result.get("risk"),
        "risk_score": result.get("risk_score"),
        "confidence": result.get("confidence"),
        "contributing_factors": result.get("contributing_factors", []),
        "model_version": result.get("model_version", "v1.0"),
        "algorithm_used": result.get("algorithm_used", "AI Model"),
        "inputs": inputs or {},
        "description": description,
        "precautions": precautions,
        "exercise_plan": plan,
    }


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user_email" not in session:
            flash("Please login first.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapper


def save_history(prediction_type, disease, result_text, risk_level=None,
                 confidence=None, risk_score=None, model_version="v1.0", inputs_dict=None):
    conn = get_db()
    inputs_json = json.dumps(inputs_dict) if inputs_dict else None

    conn.execute(
        """
        INSERT INTO history
        (user_email, prediction_type, disease, result, risk_level, confidence, risk_score, model_version, inputs_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session["user_email"],
            prediction_type,
            disease,
            result_text,
            risk_level,
            confidence,
            risk_score,
            model_version,
            inputs_json
        ),
    )
    conn.commit()
    conn.close()


# ====================================================
# LANDING & FAQ
# ====================================================

@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/faq")
def faq():
    return render_template("faq.html")


@app.route("/model_performance")
def model_performance():
    metrics = prediction.metrics_data
    return render_template("model_performance.html", metrics=metrics)


# ====================================================
# REGISTER & LOGIN
# ====================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name:
            flash("Please enter your name.", "danger")
            return render_template("register.html")

        if "@" not in email or "." not in email.split("@")[-1]:
            flash("Please enter a valid email address.", "danger")
            return render_template("register.html")

        if len(password) < 8:
            flash("Password must be at least 8 characters long.", "danger")
            return render_template("register.html")

        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
        conn = get_db()

        try:
            conn.execute(
                "INSERT INTO users(name,email,password) VALUES(?,?,?)",
                (name, email, hashed_password),
            )
            conn.commit()
            flash("Registration Successful. Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already exists.", "danger")
        finally:
            conn.close()

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        conn.close()

        if user and bcrypt.check_password_hash(user["password"], password):
            session["user_email"] = user["email"]
            session["user_name"] = user["name"]
            flash("Login Successful", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid Email or Password", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged Out Successfully", "success")
    return redirect(url_for("landing"))


# ====================================================
# DASHBOARD
# ====================================================

@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db()
    email = session["user_email"]

    history = conn.execute(
        "SELECT * FROM history WHERE user_email=? ORDER BY created_at DESC LIMIT 5",
        (email,),
    ).fetchall()

    total_count = conn.execute(
        "SELECT COUNT(*) FROM history WHERE user_email=?", (email,)
    ).fetchone()[0]

    high_risk_count = conn.execute(
        "SELECT COUNT(*) FROM history WHERE user_email=? AND risk_level='High Risk'", (email,)
    ).fetchone()[0]

    low_risk_count = conn.execute(
        "SELECT COUNT(*) FROM history WHERE user_email=? AND risk_level='Low Risk'", (email,)
    ).fetchone()[0]

    type_rows = conn.execute(
        "SELECT prediction_type, COUNT(*) as cnt FROM history WHERE user_email=? GROUP BY prediction_type",
        (email,)
    ).fetchall()

    conn.close()

    type_labels = [r["prediction_type"] for r in type_rows]
    type_counts = [r["cnt"] for r in type_rows]

    return render_template(
        "dashboard.html",
        history=history,
        total_count=total_count,
        high_risk_count=high_risk_count,
        low_risk_count=low_risk_count,
        type_labels=type_labels,
        type_counts=type_counts
    )


# ====================================================
# SYMPTOM CHECKER
# ====================================================

@app.route("/symptom-checker")
@login_required
def symptom_checker():
    return render_template(
        "symptom_checker.html",
        all_symptoms=prediction.all_symptoms,
    )


@app.route("/predict_symptom", methods=["POST"])
@login_required
def predict_symptom():
    symptoms = request.form.getlist("symptoms")

    if not symptoms:
        flash("Please select at least one symptom before submitting.", "warning")
        return redirect(url_for("symptom_checker"))

    try:
        result = prediction.predict_symptom_disease(symptoms)
    except Exception:
        flash("Something went wrong while checking your symptoms. Please try again.", "danger")
        return redirect(url_for("symptom_checker"))

    save_history(
        "Symptom Checker",
        result["disease"],
        f"Match: {result['confidence']}%",
        risk_level="Screening Match",
        confidence=result["confidence"],
        model_version=result.get("model_version", "v1.0"),
        inputs_dict={"selected_symptoms": symptoms}
    )

    store_last_prediction(
        "Symptom Checker",
        result,
        inputs={"Selected Symptoms": ", ".join(symptoms)},
        description=result.get("description"),
        precautions=result.get("precautions"),
    )

    return render_template(
        "result.html",
        result=result,
    )


# ====================================================
# DIABETES
# ====================================================

@app.route("/diabetes")
@login_required
def diabetes():
    return render_template("diabetes.html")


@app.route("/predict_diabetes", methods=["POST"])
@login_required
def predict_diabetes():
    try:
        result = prediction.predict_diabetes(request.form)
    except (KeyError, ValueError, TypeError):
        flash(
            "We couldn't process that submission — please check every field is "
            "filled in with a valid number and try again.",
            "danger",
        )
        return redirect(url_for("diabetes"))

    save_history(
        "Diabetes",
        result["disease"],
        result["result"],
        risk_level=result["risk"],
        confidence=result["confidence"],
        risk_score=result["risk_score"],
        model_version=result["model_version"],
        inputs_dict=dict(request.form)
    )

    plan = exercise_plan_module.get_exercise_plan(
        result["disease"],
        age=_real_age_years(request.form),
        bmi=request.form.get("BMI"),
        risk=result["risk"],
    )

    store_last_prediction("Diabetes", result, inputs=dict(request.form), plan=plan)

    return render_template(
        "disease_result.html",
        result=result,
        tips=get_lifestyle_tips(result["disease"], result["risk"]),
        exercise_plan=plan,
    )


# ====================================================
# HEART DISEASE
# ====================================================

@app.route("/heart")
@login_required
def heart():
    return render_template("heart.html")


@app.route("/predict_heart", methods=["POST"])
@login_required
def predict_heart():
    try:
        result = prediction.predict_heart(request.form)
    except (KeyError, ValueError, TypeError):
        flash(
            "We couldn't process that submission — please check every field is "
            "filled in with a valid number and try again.",
            "danger",
        )
        return redirect(url_for("heart"))

    save_history(
        "Heart Disease",
        result["disease"],
        result["result"],
        risk_level=result["risk"],
        confidence=result["confidence"],
        risk_score=result["risk_score"],
        model_version=result["model_version"],
        inputs_dict=dict(request.form)
    )

    plan = exercise_plan_module.get_exercise_plan(
        result["disease"],
        age=_real_age_years(request.form),
        bmi=request.form.get("BMI"),
        risk=result["risk"],
    )

    store_last_prediction("Heart Disease", result, inputs=dict(request.form), plan=plan)

    return render_template(
        "disease_result.html",
        result=result,
        tips=get_lifestyle_tips(result["disease"], result["risk"]),
        exercise_plan=plan,
    )


# ====================================================
# STROKE
# ====================================================

@app.route("/stroke")
@login_required
def stroke():
    return render_template("stroke.html")


@app.route("/predict_stroke", methods=["POST"])
@login_required
def predict_stroke():
    try:
        result = prediction.predict_stroke(request.form)
    except (KeyError, ValueError, TypeError):
        flash(
            "We couldn't process that submission — please check every field is "
            "filled in with a valid number and try again.",
            "danger",
        )
        return redirect(url_for("stroke"))

    save_history(
        "Stroke",
        result["disease"],
        result["result"],
        risk_level=result["risk"],
        confidence=result["confidence"],
        risk_score=result["risk_score"],
        model_version=result["model_version"],
        inputs_dict=dict(request.form)
    )

    plan = exercise_plan_module.get_exercise_plan(
        result["disease"],
        age=_real_age_years(request.form, direct_field="age"),
        bmi=request.form.get("bmi"),
        risk=result["risk"],
    )

    store_last_prediction("Stroke", result, inputs=dict(request.form), plan=plan)

    return render_template(
        "disease_result.html",
        result=result,
        tips=get_lifestyle_tips(result["disease"], result["risk"]),
        exercise_plan=plan,
    )


# ====================================================
# HISTORY & DETAILS
# ====================================================

@app.route("/history")
@login_required
def history():
    conn = get_db()
    records = conn.execute(
        "SELECT * FROM history WHERE user_email=? ORDER BY created_at DESC",
        (session["user_email"],),
    ).fetchall()
    conn.close()

    return render_template("history.html", history=records)


@app.route("/history/<int:record_id>")
@login_required
def view_history_detail(record_id):
    conn = get_db()
    record = conn.execute(
        "SELECT * FROM history WHERE id=? AND user_email=?",
        (record_id, session["user_email"]),
    ).fetchone()
    conn.close()

    if not record:
        flash("Historical prediction record not found.", "warning")
        return redirect(url_for("history"))

    inputs = {}
    if record["inputs_json"]:
        try:
            inputs = json.loads(record["inputs_json"])
        except Exception:
            inputs = {}

    return render_template("prediction_detail.html", record=record, inputs=inputs)


# ====================================================
# FEATURE 1 — DOWNLOAD PDF HEALTH REPORT
# ====================================================

@app.route("/download_report")
@login_required
def download_report():
    data = dict(session.get("last_prediction") or {})

    if not data:
        flash("Run a prediction first to generate a report.", "warning")
        return redirect(url_for("dashboard"))

    if data.get("generated_at"):
        try:
            data["generated_at"] = datetime.fromisoformat(data["generated_at"])
        except ValueError:
            data["generated_at"] = datetime.now()

    if data.get("risk"):
        data["lifestyle_tips"] = get_lifestyle_tips(data.get("disease"), data.get("risk"))

    pdf_buffer = report_generator.build_health_report(data)
    filename = f"MediPredict_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


# ====================================================
# FEATURE 2 — NEARBY HOSPITALS
# ====================================================

@app.route("/nearby_hospitals")
@login_required
def nearby_hospitals():
    try:
        lat = float(request.args.get("lat"))
        lng = float(request.args.get("lng"))
    except (TypeError, ValueError):
        return jsonify({"configured": hospital_finder.is_configured(), "hospitals": []}), 400

    hospitals = hospital_finder.find_nearby_hospitals(lat, lng)

    return jsonify({
        "configured": hospital_finder.is_configured(),
        "hospitals": hospitals,
    })


def get_user_latest_assessment(email):
    """Retrieves user's most recent prediction context from session or DB."""
    ctx = session.get("last_prediction")
    if ctx and ctx.get("disease"):
        return ctx

    try:
        conn = get_db()
        rec = conn.execute(
            "SELECT * FROM history WHERE user_email=? ORDER BY created_at DESC LIMIT 1",
            (email,)
        ).fetchone()
        conn.close()

        if rec:
            return {
                "prediction_type": rec["prediction_type"],
                "disease": rec["disease"],
                "result": rec["result"],
                "risk": rec["risk_level"],
                "risk_score": rec["risk_score"],
                "confidence": rec["confidence"],
                "model_version": rec["model_version"] or "v1.0"
            }
    except Exception:
        pass
    return None


# ====================================================
# FEATURE 4 — AI HEALTH CHATBOT
# ====================================================

@app.route("/api/chat", methods=["POST"])
@app.route("/chatbot", methods=["POST"])
def chatbot_reply():
    if "user_email" not in session:
        return jsonify({"success": False, "error": "Authentication required. Please log in first."}), 401

    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()

    if not message:
        return jsonify({"success": False, "error": "Please enter a valid message."}), 400

    if len(message) > 1000:
        return jsonify({"success": False, "error": "Message is too long (maximum 1000 characters)."}), 400

    context = get_user_latest_assessment(session["user_email"])

    reply, error = chatbot.ask_gemini(
        message,
        prediction_context=context,
        history=payload.get("history"),
    )

    if error:
        return jsonify({
            "success": False,
            "error": error
        }), 400 if "configured" in error.lower() else 503

    return jsonify({
        "success": True,
        "response": reply,
        "reply": reply
    })


# ====================================================
# ERROR HANDLERS
# ====================================================

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="That page doesn't exist."), 404


@app.errorhandler(500)
def server_error(e):
    app.logger.error("Unhandled server error: %s", e)
    return render_template("error.html", code=500, message="Something went wrong on our end. Please try again."), 500


if __name__ == "__main__":
    config.log_startup_diagnostics()

    if not os.path.exists(DB_PATH):
        print("database.db not found. Creating...")
        import create_db
        create_db.create_or_migrate_database()

    app.run(debug=True)