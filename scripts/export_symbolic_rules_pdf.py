"""Write the symbolic rule base to NSF_Symbolic_Rules.pdf in the project root.

The point totals in the worked examples are computed by the same rule
functions the framework uses at inference time.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpdf import FPDF
from fpdf.fonts import FontFace

from nsf.symbolic.rules import (
    BANKING_FRAUD_THRESHOLD,
    BANKING_RULES,
    HEALTHCARE_RULES,
    HEALTHCARE_THRESHOLDS,
)

OUT = ROOT / "NSF_Symbolic_Rules.pdf"

INK = (26, 35, 50)
MUTED = (80, 90, 102)
TEAL = (12, 92, 86)
TEAL_SOFT = (229, 243, 241)
RULE = (210, 200, 186)


def _fire(rules, features):
    fired = []
    score = 0.0
    for rule in rules:
        hit, message, points = rule["fn"](features)
        if hit:
            fired.append({"rule": rule["name"], "points": points, "message": message})
            score += points
    return fired, round(score, 3)


def _healthcare_label(score: float) -> str:
    if score >= HEALTHCARE_THRESHOLDS["high"]:
        return "high_risk"
    if score >= HEALTHCARE_THRESHOLDS["medium"]:
        return "medium_risk"
    return "low_risk"


def _banking_label(score: float) -> str:
    return "fraudulent" if score >= BANKING_FRAUD_THRESHOLD else "legitimate"


P00000 = {
    "age": 52,
    "bmi": 16.0,
    "systolic_bp": 144.6,
    "diastolic_bp": 74.1,
    "glucose_mg_dl": 117.1,
    "cholesterol_mg_dl": 148.0,
    "heart_rate": 83.6,
    "oxygen_sat": 97.2,
    "smoker": False,
    "family_history": False,
    "exercise_hours_week": 4.31,
    "sleep_hours": 6.69,
}
P00002 = {
    "age": 58,
    "bmi": 21.49,
    "systolic_bp": 144.2,
    "diastolic_bp": 80.0,
    "glucose_mg_dl": 109.8,
    "cholesterol_mg_dl": 170.5,
    "heart_rate": 82.3,
    "oxygen_sat": 96.3,
    "smoker": True,
    "family_history": False,
    "exercise_hours_week": 3.29,
    "sleep_hours": 6.32,
}
T000000 = {
    "amount": 145.98,
    "hour_of_day": 12,
    "day_of_week": 0,
    "merchant_risk": 0.01,
    "distance_from_home_km": 204.96,
    "distance_from_last_txn_km": 3.67,
    "txn_count_1h": 4,
    "txn_count_24h": 4,
    "avg_amount_30d": 56.28,
    "amount_vs_avg_ratio": 2.594,
    "is_international": True,
    "is_night": False,
    "card_present": True,
    "failed_pin_attempts": 0,
}


class RulesPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, "NSF symbolic rules  |  Healthcare and Banking", align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 8, f"Page {self.page_no()}  |  Rules implemented in nsf/symbolic/rules.py and nsf/symbolic/engine.py", align="C")


def heading(pdf: FPDF, text: str) -> None:
    if pdf.get_y() > 255:
        pdf.add_page()
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*TEAL)
    pdf.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_text_color(*INK)


def para(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 5.3, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.6)


def bullets(pdf: FPDF, items: list[str]) -> None:
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*INK)
    for item in items:
        pdf.multi_cell(0, 5.3, f"-  {item}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(0.6)
    pdf.ln(1.2)


def ensure_room(pdf: FPDF, millimeters: float) -> None:
    if pdf.get_y() + millimeters > pdf.page_break_trigger:
        pdf.add_page()


def ensure_room(pdf: FPDF, millimeters: float) -> None:
    if pdf.get_y() + millimeters > pdf.page_break_trigger:
        pdf.add_page()


def add_table(pdf: FPDF, headers: list[str], rows: list[list[str]], widths: tuple[int, ...]) -> None:
    pdf.set_font("Helvetica", "", 8.5)
    style = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=TEAL)
    with pdf.table(
        width=sum(widths),
        col_widths=widths,
        headings_style=style,
        text_align="LEFT",
        line_height=4.4,
        padding=1.4,
        borders_layout="ALL",
        cell_fill_color=TEAL_SOFT,
        cell_fill_mode="ROWS",
        repeat_headings=1,
    ) as table:
        head = table.row()
        for name in headers:
            head.cell(name)
        for values in rows:
            row = table.row()
            for value in values:
                row.cell(value)
    pdf.ln(2)


def points(value: float) -> str:
    return f"{value:+.1f}"


def build() -> None:
    hc00, score00 = _fire(HEALTHCARE_RULES, P00000)
    hc02, score02 = _fire(HEALTHCARE_RULES, P00002)
    bk00, score_b = _fire(BANKING_RULES, T000000)
    if abs(score00 - 1.5) > 1e-9 or _healthcare_label(score00) != "low_risk":
        raise RuntimeError(f"Unexpected P00000 symbolic result: {score00}")
    if abs(score02 - 2.7) > 1e-9 or _healthcare_label(score02) != "medium_risk":
        raise RuntimeError(f"Unexpected P00002 symbolic result: {score02}")
    if abs(score_b - 3.5) > 1e-9 or _banking_label(score_b) != "fraudulent":
        raise RuntimeError(f"Unexpected T000000 symbolic result: {score_b}")

    pdf = RulesPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(15, 16, 15)
    pdf.set_title("NSF symbolic rules")
    pdf.add_page()

    pdf.set_fill_color(*TEAL)
    pdf.rect(0, 0, 210, 8, "F")
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 8, "Symbolic rules and how a conclusion is reached", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(
        0,
        5.5,
        "Domain-agnostic federated neuro-symbolic framework\nHealthcare disease risk and banking fraud",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2)
    pdf.set_draw_color(*TEAL)
    pdf.set_line_width(0.4)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    para(
        pdf,
        "This note lists the symbolic rules the framework actually runs. They are fixed Python checks, not text generated by the neural model. For every case the system adds the points of the rules that fire, maps that total to a class, and only then decides whether the symbolic class replaces the neural class. The written explanation is allowed to mention only the rules that fired.",
    )

    heading(pdf, "1. From features to a conclusion")
    bullets(
        pdf,
        [
            "The saved federated neural model proposes a class and a confidence. That proposal is not the explanation.",
            "Every symbolic rule is tested against the case features. A rule that does not meet its condition adds 0. A rule that fires adds its fixed points. Negative points lower the score.",
            "The sum of the fired points is the symbolic score. Thresholds turn that score into a symbolic class.",
            "If the symbolic class and the neural class are the same, they agree. The conclusion is that class.",
            "If they differ, the neural class is kept unless an override condition below is met. An override replaces the neural class with the symbolic class. A disagreement that is not overridden is stored as a conflict, and the explanation must say so.",
            "The explanation is built only from the entity id, the final class, the fired rule names, and the evidence sentences those rules produced.",
        ],
    )

    heading(pdf, "2. Healthcare rules")
    para(
        pdf,
        "Task: classify a patient as low_risk, medium_risk, or high_risk. Ten rules. Protective rules carry negative points.",
    )
    add_table(
        pdf,
        ["Rule", "Fires when", "Points", "Evidence recorded"],
        [
            ["hypertension", "Systolic BP >= 140 or diastolic BP >= 90", "+1.5", "Blood pressure in mmHg"],
            ["hyperglycemia", "Glucose >= 126 mg/dL", "+1.4", "Glucose value"],
            ["obesity", "BMI >= 30", "+1.0", "BMI value"],
            ["smoker", "smoker is true", "+1.2", "Current smoker status is positive"],
            ["family_history", "family_history is true", "+1.0", "Positive family history"],
            ["hypoxia", "Oxygen saturation < 94%", "+1.6", "Oxygen saturation"],
            ["sedentary", "Exercise < 1.5 hours/week", "+0.6", "Weekly exercise hours"],
            ["advanced_age", "Age >= 60", "+0.8", "Age"],
            ["protective_exercise", "Exercise >= 5 hours/week", "-0.7", "Protective exercise level"],
            ["protective_sleep", "Sleep is from 7 to 9 hours", "-0.4", "Sleep duration"],
        ],
        (38, 62, 18, 62),
    )

    ensure_room(pdf, 52)
    heading(pdf, "3. Healthcare score to class")
    add_table(
        pdf,
        ["Symbolic score", "Symbolic class"],
        [
            ["score < 2.0", "low_risk"],
            ["2.0 <= score < 4.0", "medium_risk"],
            ["score >= 4.0", "high_risk"],
        ],
        (80, 100),
    )
    para(
        pdf,
        "Firing one risk rule does not by itself make the class medium or high. The class is the total. A single hypertension rule is +1.5, which is still low_risk.",
    )

    heading(pdf, "4. When healthcare replaces the neural class")
    para(pdf, "The neural class is replaced only in these situations. Otherwise it stands.")
    bullets(
        pdf,
        [
            "Hard constraint, and the neural class is low_risk. This applies when oxygen saturation < 92, or systolic BP >= 180, or glucose >= 250, or the score is at least 4.0 and at least four rules fired. Oxygen below 92, or a score of at least 4.0 with four or more rules, forces high_risk. The other critical readings force medium_risk.",
            "Medium floor, and the neural class is low_risk: score >= 3.0 and at least four rules fired. The conclusion becomes medium_risk.",
            "Any disagreement where the neural confidence is below 0.55. The conclusion becomes the symbolic class. This is the override used for patient P00002.",
            "Any other disagreement: the neural class is kept, and the conflict is written into the explanation.",
        ],
    )

    heading(pdf, "5. Worked healthcare conclusions")
    para(
        pdf,
        "Both patients are from the held-out healthcare test set. The neural class is the saved FedAdagrad model. The symbolic score below was produced by running HEALTHCARE_RULES on these features.",
    )

    para(pdf, "Patient P00000. Neural class: low_risk (confidence 0.972). Features used by the rules: age 52, BMI 16.0, BP 144.6/74.1, glucose 117.1, oxygen 97.2%, smoker no, family history no, exercise 4.31 h/week, sleep 6.69 h.")
    ensure_room(pdf, 28)
    add_table(
        pdf,
        ["Fired rule", "Why it fired", "Points"],
        [[item["rule"], item["message"], points(item["points"])] for item in hc00]
        + [["Total symbolic score", "score < 2.0, so the symbolic class is low_risk", points(score00)]],
        (42, 118, 20),
    )
    para(
        pdf,
        "Conclusion: symbolic class low_risk and neural class low_risk. They agree, so the final class is low_risk. The explanation may cite only hypertension. No other healthcare rule fired, so none of those names can appear as applied rules.",
    )

    para(pdf, "Patient P00002. Neural class: low_risk (confidence 0.516). Features: age 58, BMI 21.49, BP 144.2/80.0, glucose 109.8, oxygen 96.3%, smoker yes, family history no, exercise 3.29 h/week, sleep 6.32 h. The test label is medium_risk.")
    ensure_room(pdf, 36)
    add_table(
        pdf,
        ["Fired rule", "Why it fired", "Points"],
        [[item["rule"], item["message"], points(item["points"])] for item in hc02]
        + [["Total symbolic score", "2.0 <= 2.7 < 4.0, so the symbolic class is medium_risk", points(score02)]],
        (42, 118, 20),
    )
    para(
        pdf,
        "Conclusion: the classes differ, and the neural confidence 0.516 is below 0.55, so the override rule applies. The neural suggestion low_risk is replaced. The final class is medium_risk. The explanation cites hypertension and smoker, states that the neural suggestion was overridden, and does not name any rule that scored 0.",
    )

    heading(pdf, "6. Banking rules")
    para(pdf, "Task: classify a transaction as legitimate or fraudulent. Ten rules. There are no negative-point rules in this domain.")
    add_table(
        pdf,
        ["Rule", "Fires when", "Points", "Evidence recorded"],
        [
            ["high_amount", "Amount >= 900", "+1.2", "Transaction amount"],
            ["amount_spike", "Amount / 30-day average >= 3.5", "+1.5", "Spike ratio"],
            ["far_from_home", "Distance from home >= 80 km", "+1.2", "Distance in km"],
            ["geo_jump", "Distance from last transaction >= 60 km", "+1.0", "Geo jump in km"],
            ["txn_burst", "Transactions in 1 hour >= 4", "+1.3", "Burst count"],
            ["international", "is_international is true", "+1.0", "International transaction flag"],
            ["night_txn", "is_night is true (hours 00:00-05:59)", "+0.8", "Night-time transaction"],
            ["failed_pin", "Failed PIN attempts >= 2", "+1.6", "PIN attempt count"],
            ["high_risk_merchant", "Merchant risk >= 0.65", "+1.1", "Merchant risk score"],
            ["card_not_present", "card_present is false", "+0.7", "Card-not-present transaction"],
        ],
        (40, 68, 18, 54),
    )

    heading(pdf, "7. Banking score to class")
    ensure_room(pdf, 28)
    add_table(
        pdf,
        ["Symbolic score", "Symbolic class"],
        [
            ["score < 3.0", "legitimate"],
            ["score >= 3.0", "fraudulent"],
        ],
        (80, 100),
    )
    para(pdf, "The neural class is replaced only in these situations. Otherwise it stands.")
    bullets(
        pdf,
        [
            "Clear fraud pattern: score >= 4.5, the neural class is legitimate, and the neural confidence is below 0.75. The conclusion becomes fraudulent.",
            "PIN and amount together: failed PIN attempts >= 3, amount-versus-average ratio >= 4.0, and the neural class is legitimate. The conclusion becomes fraudulent.",
            "Any disagreement where the neural confidence is below 0.55. The conclusion becomes the symbolic class.",
            "Any other disagreement: the neural class is kept, and the conflict is written into the explanation.",
        ],
    )

    ensure_room(pdf, 78)
    heading(pdf, "8. Worked banking conclusion")
    para(
        pdf,
        "Transaction T000000 is the first held-out banking test case. The neural class is the saved FedAvg model: fraudulent, confidence 0.999. The test label is fraudulent. The symbolic score was produced by running BANKING_RULES on these features: amount 145.98, 204.96 km from home, 3.67 km from the last transaction, 4 transactions in one hour, international yes, night no, card present yes, merchant risk 0.01, PIN failures 0.",
    )
    add_table(
        pdf,
        ["Fired rule", "Why it fired", "Points"],
        [[item["rule"], item["message"], points(item["points"])] for item in bk00]
        + [["Total symbolic score", "score >= 3.0, so the symbolic class is fraudulent", points(score_b)]],
        (42, 118, 20),
    )
    para(
        pdf,
        "Conclusion: symbolic class fraudulent and neural class fraudulent. They agree, so the final class is fraudulent. No override is required. The explanation cites far_from_home, txn_burst, and international. high_amount, amount_spike, geo_jump, night_txn, failed_pin, high_risk_merchant, and card_not_present did not fire, so they are not part of the conclusion.",
    )

    heading(pdf, "9. What this shows")
    bullets(
        pdf,
        [
            "The class from the symbolic layer is a sum of named rules with published thresholds, not a second neural guess.",
            "A fired rule is evidence. It becomes the class only when the total crosses 2.0 or 4.0 in healthcare, or 3.0 in banking.",
            "The neural class is overturned only by the override conditions in sections 4 and 7. P00002 is an overturn. P00000 and T000000 are agreements.",
            "The explanation cannot introduce a rule that scored zero. That is the check a teacher can trace from the tables above to the sentence the system prints.",
        ],
    )

    pdf.output(OUT)
    print(f"Wrote {OUT}")
    print(f"P00000 score={score00} class={_healthcare_label(score00)} rules={[r['rule'] for r in hc00]}")
    print(f"P00002 score={score02} class={_healthcare_label(score02)} rules={[r['rule'] for r in hc02]}")
    print(f"T000000 score={score_b} class={_banking_label(score_b)} rules={[r['rule'] for r in bk00]}")


if __name__ == "__main__":
    build()
