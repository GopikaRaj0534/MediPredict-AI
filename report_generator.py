"""
report_generator.py
---------------------
FEATURE 1 — AI Health Report Generation.

Builds a downloadable "Health Assessment Report" PDF (via ReportLab) that
summarizes a single prediction: patient info, prediction result, risk score,
contributing factors, AI analysis, personalized exercise plan, general
recommendations, model versioning, and medical disclaimers.
"""

import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

INK = colors.HexColor("#10241F")
PULSE = colors.HexColor("#E1523C")
MUTED = colors.HexColor("#5C6B65")
PAPER_DIM = colors.HexColor("#F1EFE4")
MIST = colors.HexColor("#DCE6DF")

EXCLUDED_INPUT_KEYS = {"Education", "Income", "education", "income"}

FIELD_LABELS = {
    "HighBP": "High Blood Pressure", "HighChol": "High Cholesterol",
    "CholCheck": "Cholesterol Check (last 5 yrs)", "BMI": "BMI", "bmi": "BMI",
    "Smoker": "Smoker", "Stroke": "Prior Stroke",
    "HeartDiseaseorAttack": "Heart Disease / Attack History",
    "Diabetes": "Diabetes History", "PhysActivity": "Physical Activity",
    "Fruits": "Eats Fruit Regularly", "Veggies": "Eats Vegetables Regularly",
    "HvyAlcoholConsump": "Heavy Alcohol Use", "AnyHealthcare": "Has Healthcare Coverage",
    "NoDocbcCost": "Skipped Doctor Due to Cost", "GenHlth": "General Health (1-5)",
    "MentHlth": "Poor Mental Health Days (last 30)", "PhysHlth": "Poor Physical Health Days (last 30)",
    "DiffWalk": "Difficulty Walking", "Sex": "Sex", "Age": "Age", "age": "Age",
    "gender": "Gender", "hypertension": "Hypertension",
    "heart_disease": "Heart Disease History", "smoking_status": "Smoking Status",
}


def _friendly_label(key):
    return FIELD_LABELS.get(key, key.replace("_", " ").title())


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontSize=22, leading=26,
                               textColor=INK, spaceAfter=2, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="ReportSubtitle", fontSize=10, textColor=MUTED, spaceAfter=14))
    styles.add(ParagraphStyle(name="SectionHeading", fontSize=13, leading=16,
                               textColor=PULSE, spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="Body", fontSize=10, leading=15, textColor=INK))
    styles.add(ParagraphStyle(name="Disclaimer", fontSize=8.5, leading=12, textColor=MUTED))
    return styles


def generate_ai_analysis(prediction_type, disease, risk, confidence):
    if prediction_type == "Symptom Checker":
        conf = f" with {confidence}% pattern match confidence" if confidence is not None else ""
        return (
            f"Based on the symptoms selected, the model's closest candidate match was "
            f"\"{disease}\"{conf}. This is a statistical pattern estimate derived from reported "
            f"symptoms only — it does not replace physical examination, lab tests, "
            f"or clinical evaluation by a physician."
        )

    conf = f" The model reported {confidence}% probability." if confidence is not None else ""
    if risk == "High Risk":
        return (
            f"Your submitted values land in the pattern associated with elevated risk for {disease.lower()}.{conf} "
            f"This is an automated statistical screening signal, not a clinical diagnosis. "
            f"We recommend sharing this summary with a qualified physician for further diagnostic testing."
        )
    elif risk == "Moderate Risk":
        return (
            f"Your submitted values reflect a moderate screening risk level for {disease.lower()}.{conf} "
            f"This indicates that key health indicators show early risk signals. Discussing these "
            f"findings with a physician and adopting positive lifestyle modifications can help prevent progression."
        )
    return (
        f"Your submitted values land in the lower risk pattern for {disease.lower()}.{conf} "
        f"This is a reassuring screening indicator, but routine medical check-ups remain "
        f"important for overall health."
    )


def build_health_report(data):
    styles = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=22 * mm, bottomMargin=18 * mm, leftMargin=20 * mm, rightMargin=20 * mm,
        title="MediPredict AI - Health Assessment Report",
    )

    story = []

    story.append(Paragraph("MediPredict <font color='#E1523C'>AI</font>", styles["ReportTitle"]))
    story.append(Paragraph("AI Health Assessment Report &mdash; Screening Aid &amp; Informational Guidance",
                            styles["ReportSubtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=MIST, spaceAfter=10))

    generated_at = data.get("generated_at") or datetime.now()
    summary_rows = [
        ["Patient Name", data.get("patient_name") or "—"],
        ["Report Generated", generated_at.strftime("%d %b %Y, %I:%M %p")],
        ["Assessment Module", data.get("prediction_type") or "—"],
        ["Predicted Condition", data.get("disease") or "—"],
        ["Model Version", data.get("model_version") or "v1.0"],
    ]

    profile = data.get("profile") or {}
    real_age = data.get("real_age") or profile.get("age") or data.get("patient_age")
    if real_age:
        try:
            summary_rows.append(["Patient Age", f"{int(float(real_age))} years"])
        except (ValueError, TypeError):
            pass

    gender = profile.get("gender") or data.get("patient_gender")
    if gender:
        summary_rows.append(["Gender", str(gender)])

    height = profile.get("height_cm")
    weight = profile.get("weight_kg")
    if height and weight:
        summary_rows.append(["Height / Weight", f"{height} cm / {weight} kg"])

    bmi = data.get("bmi") or profile.get("bmi")
    if bmi:
        try:
            bmi_float = float(bmi)
            bmi_cat = "Underweight" if bmi_float < 18.5 else ("Normal weight" if bmi_float < 25.0 else ("Overweight" if bmi_float < 30.0 else "Obese range"))
            summary_rows.append(["Body Mass Index (BMI)", f"{bmi_float:.1f} ({bmi_cat})"])
        except (ValueError, TypeError):
            pass

    if data.get("risk"):
        summary_rows.append(["Risk Category", data["risk"]])
    if data.get("risk_score") is not None:
        summary_rows.append(["Health Risk Score", f"{data['risk_score']} / 100"])
    if data.get("result"):
        summary_rows.append(["Prediction Outcome", data["result"]])
    if data.get("confidence") is not None:
        summary_rows.append(["Model Probability", f"{data['confidence']}%"])

    t = Table(summary_rows, colWidths=[55 * mm, 105 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, MIST),
        ("BACKGROUND", (0, 0), (-1, -1), PAPER_DIM),
    ]))
    story.append(t)

    # Contributing Factors section
    factors = data.get("contributing_factors") or []
    if factors:
        story.append(Paragraph("Key Contributing Factors (XAI Weightings)", styles["SectionHeading"]))
        f_rows = [[f["label"], f"Submitted Value: {f['val']}", f["impact"]] for f in factors]
        ft = Table(f_rows, colWidths=[65 * mm, 55 * mm, 40 * mm])
        ft.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, -1), INK),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, MIST),
        ]))
        story.append(ft)

    inputs = {k: v for k, v in (data.get("inputs") or {}).items() if k not in EXCLUDED_INPUT_KEYS}
    if inputs:
        story.append(Paragraph("Submitted Assessment Values", styles["SectionHeading"]))
        rows = [[_friendly_label(k), str(v)] for k, v in inputs.items()]
        it = Table(rows, colWidths=[75 * mm, 85 * mm])
        it.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, -1), INK),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, MIST),
        ]))
        story.append(it)

    if data.get("description"):
        story.append(Paragraph("About This Condition", styles["SectionHeading"]))
        story.append(Paragraph(data["description"], styles["Body"]))

    if data.get("precautions"):
        story.append(Paragraph("Precautions & Guidelines", styles["SectionHeading"]))
        for p in data["precautions"]:
            story.append(Paragraph(f"• {p}", styles["Body"]))

    story.append(Paragraph("AI Screening Analysis", styles["SectionHeading"]))
    story.append(Paragraph(
        generate_ai_analysis(data.get("prediction_type"), data.get("disease"),
                              data.get("risk"), data.get("confidence")),
        styles["Body"],
    ))

    plan = data.get("exercise_plan")
    if plan:
        story.append(Paragraph("Personalized Exercise Guidance", styles["SectionHeading"]))
        story.append(Paragraph(f"<b>Suggested activities:</b> {', '.join(plan['exercises'])}", styles["Body"]))
        story.append(Paragraph(
            f"<b>Duration:</b> {plan['duration']} per session &nbsp;&nbsp; "
            f"<b>Frequency:</b> {plan['frequency']} &nbsp;&nbsp; "
            f"<b>Intensity:</b> {plan['intensity']}", styles["Body"]))
        sched_rows = [[d["day"], d["focus"]] for d in plan["schedule"]]
        st = Table(sched_rows, colWidths=[35 * mm, 125 * mm])
        st.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (-1, -1), INK),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, MIST),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ]))
        story.append(Spacer(1, 4))
        story.append(st)
        if plan.get("safety_tips"):
            story.append(Spacer(1, 6))
            story.append(Paragraph("<b>Exercise Safety Precautions:</b>", styles["Body"]))
            for tip in plan["safety_tips"]:
                story.append(Paragraph(f"• {tip}", styles["Body"]))

    tips = data.get("lifestyle_tips")
    if tips:
        story.append(Paragraph("General Wellness Recommendations", styles["SectionHeading"]))
        for icon, tip in tips:
            story.append(Paragraph(f"{icon} {tip}", styles["Body"]))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.75, color=MIST, spaceAfter=8))
    story.append(Paragraph(
        "<b>Medical & Legal Disclaimer:</b> This report is generated by an automated machine-learning "
        "screening tool for educational and health risk assessment purposes only. It is NOT a medical "
        "diagnosis and must not replace consultation with a qualified clinician. "
        "If you are experiencing a medical emergency, call local emergency services immediately.",
        styles["Disclaimer"],
    ))

    doc.build(story)
    buf.seek(0)
    return buf
