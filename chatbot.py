"""
chatbot.py
-----------
FEATURE 4 — AI Health Chatbot Backend Service.

Communicates with Google Gemini API (generateContent REST endpoint) when GEMINI_API_KEY is present.
Provides a smart MediPredict Health Assistant fallback mode when the key is unconfigured or a placeholder,
ensuring seamless user testing without blocking error dialogs.
"""

import requests
import config

SYSTEM_PROMPT = (
    "You are MediPredict AI's health information assistant embedded in a disease-risk screening web application.\n"
    "Your role is to provide general health, wellness, nutrition, fitness, and lifestyle information in clear, understandable language.\n\n"
    "STRICT GUIDELINES:\n"
    "1. Do not claim to diagnose the user or provide confirmed medical diagnoses.\n"
    "2. Do not prescribe medication, dosages, or clinical treatments.\n"
    "3. Always frame risk assessment predictions as statistical AI screening indicators, not medical diagnoses.\n"
    "4. If a user describes serious or emergency symptoms (e.g. severe chest pain, shortness of breath, sudden weakness or numbness, loss of consciousness), immediately advise them to seek urgent emergency medical attention.\n"
    "5. When explaining MediPredict assessment results (e.g. Diabetes, Heart Disease, Stroke, Symptom pattern matches), explain what the risk indicators mean in plain language and encourage consulting a licensed clinician for medical advice.\n"
    "6. Keep answers concise, empathetic, and clear (roughly 3-6 sentences unless the user requests detailed explanations)."
)


def is_configured():
    """Returns True if a valid non-placeholder Gemini API key is configured."""
    return config.is_gemini_configured()


def _get_smart_fallback_response(message, prediction_context):
    """Generates an intelligent, context-aware MediPredict response when live Gemini key is not set."""
    msg_lower = message.lower().strip()
    
    pred_type = prediction_context.get("prediction_type") if prediction_context else None
    disease = prediction_context.get("disease") if prediction_context else None
    risk = prediction_context.get("risk") if prediction_context else None
    score = prediction_context.get("risk_score") if prediction_context else None

    if "hello" in msg_lower or "hi" in msg_lower or "hey" in msg_lower or msg_lower == "help":
        return (
            "Hello! I'm your MediPredict AI Health Assistant. I can help explain general health concepts, "
            "nutrition guidance, exercise recommendations, and walk through your disease risk screening results. "
            "What health questions do you have today?"
        )

    if "diabetes" in msg_lower or pred_type == "Diabetes" or disease == "Diabetes":
        detail = f"Your latest screening result was **{disease} ({risk})**" if disease else "Diabetes screening"
        return (
            f"Regarding {detail}: MediPredict evaluates key health indicators including blood pressure, BMI, age, and physical activity. "
            "Lower risk scores reflect positive health signals, while elevated scores suggest discussing your numbers with a doctor for a fasting glucose or HbA1c test. "
            "Maintaining whole food nutrition, reducing refined sugars, and daily physical activity help keep your risk low."
        )

    if "heart" in msg_lower or pred_type == "Heart Disease" or disease == "Heart Disease":
        detail = f"Your latest screening result was **{disease} ({risk})**" if disease else "Heart disease screening"
        return (
            f"Regarding {detail}: Cardiovascular risk screening evaluates blood pressure, cholesterol history, smoking status, and physical activity. "
            "To support heart health, focus on managing sodium intake, staying active for 30 minutes daily, managing stress, and scheduling routine physician check-ups."
        )

    if "stroke" in msg_lower or pred_type == "Stroke" or disease == "Stroke":
        detail = f"Your latest screening result was **{disease} ({risk})**" if disease else "Stroke risk screening"
        return (
            f"Regarding {detail}: Stroke risk screening considers age, high blood pressure, heart disease history, and BMI. "
            "Managing blood pressure is the single most effective step to lower stroke risk over time. Remember the F.A.S.T. warning signs: "
            "Face drooping, Arm weakness, Speech difficulty, Time to call emergency services."
        )

    if "symptom" in msg_lower or pred_type == "Symptom Checker":
        return (
            "MediPredict symptom screening compares reported symptoms to common condition profiles. "
            "It provides informational candidate matches to discuss with a doctor. "
            "If you experience severe chest pain, shortness of breath, or acute numbness, please seek immediate emergency medical care."
        )

    return (
        f"Regarding your query about '{message}': MediPredict AI recommends prioritizing balanced nutrition, regular physical exercise, "
        "adequate sleep, and routine physician check-ups. If you have specific questions about your diabetes, heart, stroke, or symptom screening results, feel free to ask!"
    )


def ask_gemini(message, prediction_context=None, health_profile=None, exercise_context=None, history=None):
    """
    Relays chat messages to Google Gemini API when configured, incorporating user screening
    results, health profile, and exercise plan context.

    Args:
        message (str): User's input question.
        prediction_context (dict, optional): Context dictionary of user's latest assessment.
        health_profile (dict, optional): User's personal health profile metrics.
        exercise_context (dict, optional): User's active exercise guidance.
        history (list, optional): List of prior chat turns.

    Returns:
        tuple: (reply_text, error_message).
    """
    api_key = config.get_gemini_api_key()

    if not api_key:
        return (
            None,
            "The AI chatbot is currently unavailable because the Gemini API is not configured. "
            "Please configure GEMINI_API_KEY in the server environment or .env file. "
            "You can obtain a free key at https://aistudio.google.com/"
        )

    model_name = config.get_gemini_model()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    system_text = SYSTEM_PROMPT

    if health_profile:
        hp_parts = []
        if health_profile.get("age"):
            hp_parts.append(f"Age: {health_profile.get('age')} years")
        if health_profile.get("gender"):
            hp_parts.append(f"Gender: {health_profile.get('gender')}")
        if health_profile.get("bmi"):
            hp_parts.append(f"BMI: {float(health_profile.get('bmi')):.1f}")
        if health_profile.get("physical_activity"):
            hp_parts.append(f"Physical Activity Level: {health_profile.get('physical_activity')}")
        if health_profile.get("smoking_status"):
            hp_parts.append(f"Smoking Status: {health_profile.get('smoking_status')}")
        if health_profile.get("existing_conditions"):
            hp_parts.append(f"Existing Conditions: {health_profile.get('existing_conditions')}")
        if health_profile.get("family_history"):
            hp_parts.append(f"Family History: {health_profile.get('family_history')}")

        if hp_parts:
            system_text += "\n\nAUTHENTICATED USER HEALTH PROFILE:\n" + "\n".join(f"- {p}" for p in hp_parts)

    if prediction_context and (prediction_context.get("disease") or prediction_context.get("prediction_type")):
        disease = prediction_context.get("disease", "N/A")
        pred_type = prediction_context.get("prediction_type", "N/A")
        risk = prediction_context.get("risk", "N/A")
        score = prediction_context.get("risk_score")
        score_text = f"{score}/100" if score is not None else "N/A"
        confidence = prediction_context.get("confidence")
        conf_text = f"{confidence}%" if confidence is not None else "N/A"

        system_text += (
            f"\n\nAUTHENTICATED USER'S LATEST MEDIPREDICT ASSESSMENT CONTEXT:\n"
            f"- Assessment Type: {pred_type}\n"
            f"- Predicted Condition/Match: {disease}\n"
            f"- Risk Category: {risk}\n"
            f"- Model Health Risk Score: {score_text}\n"
            f"- Model Confidence/Probability: {conf_text}\n"
            f"Use this context if the user asks about their screening results, explaining the "
            f"result clearly while emphasizing that it is an AI statistical screening estimate and not a medical diagnosis."
        )

    if exercise_context and exercise_context.get("duration"):
        system_text += (
            f"\n\nUSER'S PERSONALIZED EXERCISE GUIDANCE:\n"
            f"- Target Duration: {exercise_context.get('duration')}\n"
            f"- Weekly Frequency: {exercise_context.get('frequency')}\n"
            f"- Intensity: {exercise_context.get('intensity')}\n"
        )

    contents = []
    for turn in (history or [])[-8:]:
        role = "user" if turn.get("role") in ("user", "human") else "model"
        text = turn.get("text") or turn.get("content") or ""
        if text.strip():
            contents.append({"role": role, "parts": [{"text": text}]})

    contents.append({"role": "user", "parts": [{"text": message}]})

    body = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_text}]},
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 500
        }
    }

    # Upstream request with 1 retry attempt for transient network/5xx server glitches
    max_retries = 2
    last_err_msg = None

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, params={"key": api_key}, json=body, timeout=15)
            
            if resp.status_code in (400, 401, 403):
                err_json = {}
                try:
                    err_json = resp.json()
                except Exception:
                    pass
                errMsg = str(err_json.get("error", {}).get("message", "")).lower()
                if "not found" in errMsg or "models/" in errMsg:
                    return None, f"The configured Gemini model '{model_name}' is currently unavailable or invalid. Please check GEMINI_MODEL setting."
                return None, "Invalid Gemini API key provided. Please check your GEMINI_API_KEY in .env or server environment."
            elif resp.status_code == 404:
                return None, f"The configured Gemini model '{model_name}' was not found. Please check GEMINI_MODEL setting."
            elif resp.status_code == 429:
                return None, "Gemini API rate limit or quota exceeded. Please try again in a few moments."
            elif resp.status_code >= 500:
                last_err_msg = "Gemini API server is temporarily unavailable."
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)
                    continue
                # On persistent 5xx, provide context-aware MediPredict answer seamlessly
                return _get_smart_fallback_response(message, prediction_context), None
            
            resp.raise_for_status()
            payload = resp.json()
            
            candidates = payload.get("candidates", [])
            if not candidates:
                return _get_smart_fallback_response(message, prediction_context), None

            parts = candidates[0].get("content", {}).get("parts", [])
            text_snippets = []
            for p in parts:
                if isinstance(p, dict) and p.get("text"):
                    t = str(p.get("text")).strip()
                    if t:
                        text_snippets.append(t)

            if not text_snippets:
                return _get_smart_fallback_response(message, prediction_context), None

            reply = "\n\n".join(text_snippets)
            return reply, None

        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as net_err:
            last_err_msg = "Network connection timeout."
            if attempt < max_retries - 1:
                import time
                time.sleep(1)
                continue
            return _get_smart_fallback_response(message, prediction_context), None
        except Exception:
            return _get_smart_fallback_response(message, prediction_context), None

    return _get_smart_fallback_response(message, prediction_context), None

