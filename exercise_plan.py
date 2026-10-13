"""
exercise_plan.py
------------------
Personalized Exercise Recommendation Engine for MediPredict AI.

Generates a structured seven-day weekly physical activity plan tailored to:
1. Disease / screening context (Diabetes, Heart Disease, Stroke, or General)
2. Application-defined risk category (High Risk, Moderate Risk, Low Risk)
3. Real age (in years)
4. Body Mass Index (BMI)
5. Current physical activity level from health profile:
   (Sedentary, Lightly Active, Moderately Active, Very Active)

Transparently distinguishes personalized recommendations from general activity
guidelines when profile or assessment data is incomplete (no silent fallback to age 40).
"""

EXERCISE_LIBRARY = {
    "Diabetes": [
        "Brisk Post-Meal Walking",
        "Low-Impact Cycling / Stationary Bike",
        "Gentle Yoga & Core Stability",
        "Light Resistance / Bodyweight Training",
        "Water Aerobics / Swimming",
    ],
    "Heart Disease": [
        "Brisk Walking at a Controlled Pace",
        "Deep Diaphragmatic Breathing & Mobility",
        "Gentle Restorative Yoga",
        "Light Stationary Cycling",
        "Low-Intensity Arm & Leg Mobility Circuits",
    ],
    "Stroke": [
        "Balance & Stability Drills (Heel-to-Toe / Standing)",
        "Assisted / Supervised Walking",
        "Seated Range-of-Motion & Stretching",
        "Targeted Neuromuscular Coordination Drills",
        "Low-Resistance Seated Stepping",
    ],
    "General": [
        "Brisk Walking / Jogging",
        "Dynamic Stretching & Mobility",
        "Bodyweight Strength Training",
        "Cycling or Swimming",
        "Yoga / Pilates for Flexibility",
    ]
}

SAFETY_TIPS = {
    "Diabetes": [
        "Check your blood glucose before and after exercise, especially if using insulin or sulfonylureas.",
        "Keep a fast-acting carbohydrate source (e.g. glucose tablets or fruit juice) accessible during workouts.",
        "Wear proper supportive footwear and inspect your feet daily for blisters, redness, or skin breaks.",
        "Stay consistently hydrated throughout each session, particularly in warmer weather.",
    ],
    "Heart Disease": [
        "Stop activity immediately and rest if you experience chest tightness, dizziness, lightheadedness, or shortness of breath.",
        "Always include 5-10 minutes of gentle warm-up and cool-down to gradually adapt heart rate and blood pressure.",
        "Keep exertion at a comfortable conversational pace; avoid isometric straining or holding your breath (Valsalva maneuver).",
        "Do not exercise outdoors in extreme cold or excessive heat without clinical clearance.",
    ],
    "Stroke": [
        "Perform balance exercises near a sturdy support bar or with a companion nearby to prevent falls.",
        "Prioritize smooth, controlled movements over rapid or jerky repetitions.",
        "Incorporate seated exercises if feeling unsteady, fatigued, or recovering neuromuscular control.",
        "Stay hydrated and avoid high-intensity exercise that could cause sudden blood pressure spikes.",
    ],
    "General": [
        "Listen to your body — stop if you experience acute joint pain, dizziness, or chest discomfort.",
        "Warm up prior to exertion and finish each session with gentle static stretching.",
        "Progress gradually by increasing workout duration before increasing intensity.",
    ]
}


def _determine_intensity(age, bmi, risk, activity_level):
    """
    Computes recommended target intensity considering risk, age, BMI, and current fitness.
    """
    # High risk or older age or higher BMI requires conservative pacing
    if risk == "High Risk":
        if (age and age >= 65) or (bmi and bmi >= 32) or activity_level == "Sedentary":
            return "Light"
        return "Light to Moderate"

    if risk == "Moderate Risk":
        if activity_level == "Sedentary":
            return "Light to Moderate"
        if (age and age >= 65) or (bmi and bmi >= 30):
            return "Moderate (Low-Impact)"
        return "Moderate"

    # Low Risk or General
    if activity_level == "Sedentary":
        return "Light to Moderate"
    elif activity_level in ("Moderately Active", "Very Active"):
        if (age and age >= 65) or (bmi and bmi >= 32):
            return "Moderate (Low-Impact)"
        return "Moderate to Vigorous"
    return "Moderate"


def _determine_duration_and_frequency(age, bmi, risk, activity_level):
    """
    Computes session duration and weekly frequency based on multi-factor profile.
    """
    if activity_level == "Sedentary":
        return "15-20 minutes", "3-4 days a week, with alternating rest days"

    if risk == "High Risk":
        return "15-25 minutes", "4-5 days a week, interspersed with active recovery"

    if (age and age >= 65) or (bmi and bmi >= 32):
        return "20-30 minutes", "4-5 days a week"

    if activity_level == "Very Active" and risk == "Low Risk":
        return "35-50 minutes", "5-6 days a week"

    if activity_level == "Moderately Active":
        return "30-40 minutes", "5 days a week"

    return "25-35 minutes", "4-5 days a week"


def _generate_schedule(disease, intensity, duration, activity_level):
    """
    Constructs a detailed day-by-day 7-day schedule with activity, focus, duration, and rationale.
    """
    key = disease if disease in EXERCISE_LIBRARY else "General"
    lib = EXERCISE_LIBRARY[key]

    # Day configurations
    is_beginner = activity_level == "Sedentary"
    
    if is_beginner:
        return [
            {
                "day": "Monday",
                "focus": lib[0],
                "activity": lib[0],
                "duration": duration,
                "intensity": intensity,
                "notes": "Start at an easy pace. Maintain steady breathing and comfortable posture."
            },
            {
                "day": "Tuesday",
                "focus": "Active Rest & Gentle Stretching",
                "activity": "Gentle Mobility & Full-Body Stretching",
                "duration": "10-15 minutes",
                "intensity": "Light",
                "notes": "Promotes recovery, reduces stiffness, and prepares joints for activity."
            },
            {
                "day": "Wednesday",
                "focus": lib[1] if len(lib) > 1 else lib[0],
                "activity": lib[1] if len(lib) > 1 else lib[0],
                "duration": duration,
                "intensity": intensity,
                "notes": "Low-impact movement to build cardiovascular tolerance gradually."
            },
            {
                "day": "Thursday",
                "focus": "Full Rest Day",
                "activity": "Rest & Hydration",
                "duration": "0 minutes",
                "intensity": "Rest",
                "notes": "Adequate rest is crucial for muscular and cardiovascular adaptation."
            },
            {
                "day": "Friday",
                "focus": lib[2] if len(lib) > 2 else lib[0],
                "activity": lib[2] if len(lib) > 2 else lib[0],
                "duration": duration,
                "intensity": intensity,
                "notes": "Focus on controlled movements, balance, and core stability."
            },
            {
                "day": "Saturday",
                "focus": "Comfortable Leisure Walk",
                "activity": "Low-Pace Walking / Recreational Movement",
                "duration": "15-20 minutes",
                "intensity": "Light",
                "notes": "Enjoyable low-stress outdoor movement or household activity."
            },
            {
                "day": "Sunday",
                "focus": "Rest & Weekly Reflection",
                "activity": "Rest / Light Breathing Exercises",
                "duration": "5-10 minutes",
                "intensity": "Rest",
                "notes": "Reflect on weekly movement and prepare for the upcoming week."
            }
        ]

    # Active / Standard schedule
    return [
        {
            "day": "Monday",
            "focus": lib[0],
            "activity": lib[0],
            "duration": duration,
            "intensity": intensity,
            "notes": "Cardiovascular aerobic base training to kickstart the weekly cycle."
        },
        {
            "day": "Tuesday",
            "focus": lib[1] if len(lib) > 1 else lib[0],
            "activity": lib[1] if len(lib) > 1 else lib[0],
            "duration": duration,
            "intensity": intensity,
            "notes": "Cross-training with low-impact joint movements to prevent overuse fatigue."
        },
        {
            "day": "Wednesday",
            "focus": "Mid-Week Active Recovery & Mobility",
            "activity": "Yoga, Pilates or Targeted Joint Mobility",
            "duration": "20 minutes",
            "intensity": "Light",
            "notes": "Enhances flexibility, reduces muscle tension, and supports arterial elasticity."
        },
        {
            "day": "Thursday",
            "focus": lib[2] if len(lib) > 2 else lib[0],
            "activity": lib[2] if len(lib) > 2 else lib[0],
            "duration": duration,
            "intensity": intensity,
            "notes": "Neuromuscular balance, stability, or light functional resistance."
        },
        {
            "day": "Friday",
            "focus": lib[0],
            "activity": lib[0],
            "duration": duration,
            "intensity": intensity,
            "notes": "Consistent aerobic conditioning to sustain metabolic health benefits."
        },
        {
            "day": "Saturday",
            "focus": lib[3] if len(lib) > 3 else (lib[1] if len(lib) > 1 else lib[0]),
            "activity": lib[3] if len(lib) > 3 else "Recreational Sport or Cycling",
            "duration": duration,
            "intensity": intensity,
            "notes": "Diverse physical activity to maintain enthusiasm and cardiovascular health."
        },
        {
            "day": "Sunday",
            "focus": "Rest & Gentle Restorative Recovery",
            "activity": "Gentle Nature Walk or Rest",
            "duration": "15-20 minutes",
            "intensity": "Light / Rest",
            "notes": "Complete rest and recovery before starting the next week's schedule."
        }
    ]


def _build_rationale(disease, risk, age, bmi, activity_level, is_personalized):
    """
    Produces a clear, readable explanation of why this plan was tailored for the user.
    """
    parts = []
    if is_personalized:
        cond_text = f"{disease} screening" if disease else "general health assessment"
        risk_text = f" ({risk})" if risk else ""
        parts.append(f"This plan is tailored to your {cond_text}{risk_text}.")

        if activity_level:
            parts.append(f"Because your baseline physical activity is listed as '{activity_level}', the initial volume is structured to build endurance safely without risking exhaustion.")

        if age and age >= 65:
            parts.append("Low-impact movements and stability exercises are prioritized to preserve joint comfort and support balance.")
        elif bmi and bmi >= 30:
            parts.append("Joint-friendly aerobic options (such as walking and cycling) were selected to maximize calorie expenditure while minimizing impact on knees and ankles.")

        if disease == "Diabetes":
            parts.append("Regular moderate aerobic exercise and post-meal walking help muscles absorb blood glucose directly and improve insulin sensitivity.")
        elif disease == "Heart Disease":
            parts.append("Steady-state aerobic exercises encourage healthy myocardial blood flow and blood pressure regulation without excessive cardiovascular strain.")
        elif disease == "Stroke":
            parts.append("Dedicated balance, coordination, and seated mobility elements assist in neuro-motor function and help prevent slips and falls.")
    else:
        parts.append(
            "This structured 7-day program follows standard evidence-based physical activity guidelines for general adult cardiovascular wellness. "
            "To receive a fully personalized regimen, complete your Health Profile and run an assessment module."
        )

    return " ".join(parts)


def get_exercise_plan(disease=None, age=None, bmi=None, risk=None, physical_activity=None):
    """
    Builds a personalized or transparently general weekly exercise plan.

    Args:
        disease (str, optional): Disease or screening module (e.g. "Diabetes", "Heart Disease", "Stroke").
        age (float/int/str, optional): User's real age in years.
        bmi (float/int/str, optional): User's Body Mass Index.
        risk (str, optional): Screening risk category ("High Risk", "Moderate Risk", "Low Risk").
        physical_activity (str, optional): "Sedentary", "Lightly Active", "Moderately Active", "Very Active".

    Returns:
        dict: Complete structured exercise guidance with schedule, safety tips, duration, intensity, and rationale.
    """
    # Clean inputs
    cleaned_age = None
    if age not in (None, ""):
        try:
            cleaned_age = float(age)
        except (ValueError, TypeError):
            cleaned_age = None

    cleaned_bmi = None
    if bmi not in (None, ""):
        try:
            cleaned_bmi = float(bmi)
        except (ValueError, TypeError):
            cleaned_bmi = None

    cleaned_risk = risk if risk in ("High Risk", "Moderate Risk", "Low Risk") else None
    cleaned_disease = disease if disease in ("Diabetes", "Heart Disease", "Stroke") else None
    cleaned_activity = physical_activity if physical_activity in (
        "Sedentary", "Lightly Active", "Moderately Active", "Very Active"
    ) else "Moderately Active"

    # Determine personalization status
    has_profile_data = bool(cleaned_age or cleaned_bmi or physical_activity)
    has_assessment_data = bool(cleaned_disease or cleaned_risk)
    is_personalized = has_profile_data or has_assessment_data

    # Multi-factor intensity & duration
    intensity = _determine_intensity(cleaned_age, cleaned_bmi, cleaned_risk, cleaned_activity)
    duration, frequency = _determine_duration_and_frequency(cleaned_age, cleaned_bmi, cleaned_risk, cleaned_activity)

    disease_key = cleaned_disease if cleaned_disease else "General"
    exercises = EXERCISE_LIBRARY.get(disease_key, EXERCISE_LIBRARY["General"])
    schedule = _generate_schedule(disease_key, intensity, duration, cleaned_activity)
    safety_tips = SAFETY_TIPS.get(disease_key, SAFETY_TIPS["General"])
    rationale = _build_rationale(cleaned_disease, cleaned_risk, cleaned_age, cleaned_bmi, physical_activity, is_personalized)

    # Explanation of data sources used
    personalization_notes = []
    if cleaned_disease:
        personalization_notes.append(f"Screening Context: {cleaned_disease}")
    if cleaned_risk:
        personalization_notes.append(f"Risk Level: {cleaned_risk}")
    if cleaned_age is not None:
        personalization_notes.append(f"Age: {int(cleaned_age)} yrs")
    if cleaned_bmi is not None:
        personalization_notes.append(f"BMI: {cleaned_bmi:.1f}")
    if physical_activity:
        personalization_notes.append(f"Activity Level: {physical_activity}")

    if not is_personalized:
        personalization_notes.append("No profile or screening data available. Using general cardiovascular activity guidelines.")

    return {
        "disease": cleaned_disease or "General Health",
        "risk": cleaned_risk or "General",
        "exercises": exercises,
        "schedule": schedule,
        "duration": duration,
        "frequency": frequency,
        "intensity": intensity,
        "safety_tips": safety_tips,
        "rationale": rationale,
        "is_personalized": is_personalized,
        "personalization_notes": personalization_notes,
        "profile_summary": {
            "age": cleaned_age,
            "bmi": cleaned_bmi,
            "risk": cleaned_risk,
            "activity_level": physical_activity or "Not specified",
        },
        "disclaimer": (
            "Physical activity recommendations are generated for educational and screening guidance only. "
            "They do not replace clinical advice. If you experience shortness of breath, chest pain, "
            "or severe dizziness, stop immediately and seek medical attention."
        )
    }
