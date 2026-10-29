"""Write NSF_Threshold_Evidence.pdf into the mail/ folder.

This document justifies every numeric threshold used by the symbolic rule
base in nsf/symbolic/rules.py. Healthcare thresholds are mapped to
internationally recognised clinical diagnostic / classification cut-points.
Banking thresholds are presented honestly as engineering decision boundaries
that (a) are calibrated to match the causal data generator and (b) use the
feature set and directional logic established in peer-reviewed fraud-detection
literature.

Only core (Helvetica) fonts are used, so text is restricted to ASCII. The
symbol ">=" is written literally rather than the Unicode glyph.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpdf import FPDF
from fpdf.fonts import FontFace

OUT_DIR = ROOT / "mail"
OUT = OUT_DIR / "NSF_Threshold_Evidence.pdf"

INK = (26, 35, 50)
MUTED = (80, 90, 102)
TEAL = (12, 92, 86)
TEAL_SOFT = (229, 243, 241)
AMBER_SOFT = (252, 243, 224)
RULE = (210, 200, 186)


class EvidencePDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MUTED)
        self.cell(
            0, 6,
            "Evidence for symbolic thresholds  |  Healthcare and Banking",
            align="L", new_x="LMARGIN", new_y="NEXT",
        )
        self.ln(2)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(
            0, 8,
            f"Page {self.page_no()}  |  Thresholds implemented in nsf/symbolic/rules.py  |  Sources attached in mail/sources/",
            align="C",
        )


def heading(pdf: FPDF, text: str) -> None:
    if pdf.get_y() > 252:
        pdf.add_page()
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*TEAL)
    pdf.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_text_color(*INK)


def subheading(pdf: FPDF, text: str) -> None:
    if pdf.get_y() > 258:
        pdf.add_page()
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.5)


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


def add_table(pdf: FPDF, headers, rows, widths, fill=TEAL_SOFT) -> None:
    pdf.set_font("Helvetica", "", 8.3)
    style = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=TEAL)
    with pdf.table(
        width=sum(widths),
        col_widths=widths,
        headings_style=style,
        text_align="LEFT",
        line_height=4.3,
        padding=1.4,
        borders_layout="ALL",
        cell_fill_color=fill,
        cell_fill_mode="ROWS",
        repeat_headings=1,
    ) as table:
        head = table.row()
        for name in headers:
            head.cell(name)
        for values in rows:
            row = table.row()
            for value in values:
                row.cell(str(value))
    pdf.ln(2)


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = EvidencePDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(15, 16, 15)
    pdf.set_title("Evidence for NSF symbolic thresholds")
    pdf.add_page()

    # ---- Title block
    pdf.set_fill_color(*TEAL)
    pdf.rect(0, 0, 210, 8, "F")
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 8, "Evidence for the numeric thresholds in the symbolic rule base",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(
        0, 5.5,
        "Domain-agnostic federated neuro-symbolic framework\n"
        "Why each cut-point was chosen for Healthcare disease risk and Banking fraud",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(2)
    pdf.set_draw_color(*TEAL)
    pdf.set_line_width(0.4)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # ---- 1. Summary / how thresholds were fixed
    heading(pdf, "1. How the thresholds were chosen")
    para(
        pdf,
        "Every symbolic rule in nsf/symbolic/rules.py fires on a fixed numeric condition (for example, "
        "systolic BP >= 140 or diastolic BP >= 90). None of these numbers were invented. They were fixed in two ways, "
        "and this document gives the source for each one so the choice can be checked independently.",
    )
    bullets(
        pdf,
        [
            "Healthcare: each threshold is a standard diagnostic or classification cut-point published by a recognised "
            "medical authority (WHO, the American Diabetes Association, NCEP ATP III, and respiratory-oxygen guidelines). "
            "These are the same numbers a clinician would use, so a rule firing has a defensible clinical meaning.",
            "Banking: there are no universal regulatory cut-points for fraud. The rules use the feature set and the "
            "directional logic that are well established in peer-reviewed fraud-detection research (amount, spend-vs-average "
            "ratio, geographic distance, transaction velocity, card-not-present, time-of-day, merchant risk). The specific "
            "numeric constants are engineering decision boundaries calibrated so that the symbolic layer agrees with the "
            "causal data generator that produced the dataset.",
            "In both domains the symbolic thresholds are aligned with the ground-truth generator in nsf/data/generators.py. "
            "That is why symbolic verification is meaningful on this data and not random: the rules test the same breakpoints "
            "that actually created the labels.",
        ],
    )
    para(
        pdf,
        "Honesty note for review: the healthcare numbers are externally grounded clinical standards. The banking numbers are "
        "literature-informed heuristics, not fixed values quoted from a paper. Section 4 states this explicitly so the "
        "distinction is clear.",
    )

    # ---- 2. Healthcare thresholds
    heading(pdf, "2. Healthcare thresholds and their clinical sources")
    para(
        pdf,
        "Task: classify a patient as low_risk, medium_risk, or high_risk. The table maps each rule's numeric condition "
        "to the published cut-point it reproduces, with the source key from Section 5.",
    )
    add_table(
        pdf,
        ["Rule (code)", "Condition in code", "Clinical cut-point it reproduces", "Src"],
        [
            ["hypertension", "systolic >= 140 or diastolic >= 90 mmHg",
             "Hypertension / Grade 1 HTN is office BP >= 140/90 mmHg (WHO 2021; ISH 2020).", "H1, H2"],
            ["hyperglycemia", "glucose >= 126 mg/dL",
             "Fasting plasma glucose >= 126 mg/dL (7.0 mmol/L) diagnoses diabetes (ADA 2026, Table 2.1).", "H3"],
            ["obesity", "BMI >= 30",
             "WHO defines obesity in adults as BMI >= 30 kg/m2 (overweight >= 25).", "H4"],
            ["hypoxia", "oxygen_sat < 94 %",
             "Normal target SpO2 is 94-98 %; below it, supplemental O2 is initiated and hypoxaemia considered.", "H6"],
            ["advanced_age", "age >= 60",
             "WHO treats 60+ as older adult; ATP III counts age as a CVD risk factor (M>=45/F>=55).", "H5, H9"],
            ["sedentary", "exercise < 1.5 h/week",
             "Well below the WHO minimum of 150 min (2.5 h)/week moderate activity.", "H7"],
            ["protective_exercise", "exercise >= 5 h/week",
             "At/above the upper WHO band of 150-300 min/week (5 h) linked to extra benefit.", "H7"],
            ["protective_sleep", "sleep 7-9 h",
             "National Sleep Foundation recommends 7-9 h/night for adults.", "H8"],
            ["smoker", "smoker is true",
             "Current smoking is an established, non-numeric CVD risk factor (ATP III Table 3).", "H5"],
            ["family_history", "family_history is true",
             "Family history of premature CHD is an established CVD risk factor (ATP III Table 3).", "H5"],
        ],
        (26, 46, 92, 16),
    )
    para(
        pdf,
        "Note on cholesterol: the dataset also records total cholesterol and the generator raises risk above 200 mg/dL, which "
        "matches the NCEP ATP III classification (< 200 desirable, 200-239 borderline high, >= 240 high). Cholesterol is used "
        "by the neural model rather than as its own firing rule, but its breakpoint is from the same ATP III source (H5).",
    )

    # ---- 3. Worked healthcare example for the two BP values in the first PDF
    subheading(pdf, "2a. Why exactly 140 and 90 (the question raised)")
    bullets(
        pdf,
        [
            "140 mmHg systolic and 90 mmHg diastolic are the office blood-pressure values at which WHO (2021) recommends "
            "starting treatment for confirmed hypertension, and at which the 2020 International Society of Hypertension "
            "classifies Grade 1 hypertension (140-159 / 90-99 mmHg). The rule uses an OR because either an elevated systolic "
            "or an elevated diastolic reading meets the clinical definition.",
            "These are not arbitrary: the attached ISH 2020 guideline (Table 1) and the WHO 2021 recommendation both state "
            "the >= 140/90 boundary verbatim. The NCEP ATP III documents (attached) also list 'BP >= 140/90 mmHg' as a "
            "cardiovascular risk factor, which is why the same number appears across independent authorities.",
        ],
    )

    # ---- 4. Banking thresholds
    heading(pdf, "3. Banking thresholds and their literature basis")
    para(
        pdf,
        "Task: classify a transaction as legitimate or fraudulent. Unlike medicine, fraud detection has no universal numeric "
        "standard; thresholds are set per system. The table shows, for each rule, the feature and directional logic that is "
        "supported by the research literature, and marks the numeric value as a calibrated design choice.",
    )
    add_table(
        pdf,
        ["Rule (code)", "Condition", "Feature / logic supported by literature", "Src"],
        [
            ["high_amount", "amount >= 900",
             "Raw transaction amount; large amounts are a core fraud feature.", "B1, B3"],
            ["amount_spike", "amount / 30-day avg >= 3.5",
             "Spend-vs-customer-baseline ratio; behavioural aggregation of normal spend.", "B1, B2"],
            ["far_from_home", "dist from home >= 80 km",
             "Geolocation distance from the customer's usual location.", "B2"],
            ["geo_jump", "dist from last txn >= 60 km",
             "Distance between consecutive transactions / implausible travel speed.", "B2"],
            ["txn_burst", "txns in 1 h >= 4",
             "Transaction velocity: counts aggregated over a short recent window.", "B1, B3"],
            ["international", "is_international is true",
             "Country grouping is an established aggregation dimension.", "B1"],
            ["night_txn", "hour in 00:00-05:59",
             "Periodic time-of-day feature (von Mises periodic modelling).", "B1"],
            ["high_risk_merchant", "merchant_risk >= 0.65",
             "Merchant group / category risk is an aggregation dimension.", "B1, B3"],
            ["card_not_present", "card_present is false",
             "Transaction channel / type (CNP) is a standard fraud attribute.", "B1"],
            ["failed_pin", "failed PIN attempts >= 2",
             "Authentication-failure signal (operational/industry practice; weaker in academic literature).", "-"],
        ],
        (28, 40, 96, 16),
    )
    heading(pdf, "4. Honest limitations (please read)")
    bullets(
        pdf,
        [
            "The banking numeric constants (900, 3.5, 80 km, 60 km, 4, 0.65, 2 PIN attempts) are NOT quoted from any paper. "
            "They are decision boundaries chosen to match the causal fraud generator and to give a sensible score scale. The "
            "literature justifies WHICH features and WHICH direction, not the exact number.",
            "The healthcare cut-points are adult, general-population standards. They are not adjusted for sex, pregnancy, "
            "ethnicity, or measurement context (e.g., home vs office BP), which real clinical use would consider.",
            "The oxygen rule uses < 94 % for a soft risk signal and < 92 % for the hard override; 94-98 % is the normal "
            "target range and < 92 % is a common practical hypoxaemia threshold, so both are defensible but context matters.",
            "Guidelines are revised over time. For example, the 2017 ACC/AHA guideline lowers the US hypertension label to "
            "130/80; this framework follows the WHO / ISH 140/90 convention and cites it explicitly.",
        ],
    )

    # ---- 5. References
    heading(pdf, "5. References (attached in mail/sources unless marked link-only)")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 5.3, "Healthcare", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.5)
    refs_hc = [
        "H1  WHO. Guideline for the pharmacological treatment of hypertension in adults. Geneva: WHO; 2021. "
        "BP threshold >= 140/90 mmHg. [link only] https://iris.who.int/ (CC BY-NC-SA 3.0 IGO).",
        "H2  Unger T, et al. 2020 International Society of Hypertension Global Hypertension Practice Guidelines. "
        "Hypertension/J Hypertens. Grade 1 HTN 140-159/90-99 mmHg. [PDF attached]",
        "H3  American Diabetes Association. 2. Diagnosis and Classification of Diabetes: Standards of Care in Diabetes-2026. "
        "Diabetes Care 2026;49(Suppl 1):S27. FPG >= 126 mg/dL. [PDF attached]",
        "H4  WHO. Obesity and overweight (fact sheet). Adults: overweight BMI >= 25, obesity BMI >= 30. "
        "[link only] https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight",
        "H5  NCEP. Third Report of the Expert Panel (ATP III). NIH/NHLBI. Total cholesterol <200/200-239/>=240; "
        "risk factors include BP >= 140/90, smoking, family history, age. [PDF attached: At-A-Glance + Executive Summary]",
        "H6  Thoracic Society of Australia & NZ, Acute Oxygen Guidelines 2016 (and BTS 2017 / NICE CKS). Target SpO2 "
        "94-98 %; < 92 % practical hypoxaemia threshold. [PDF attached]",
        "H7  Bull FC, et al. WHO 2020 guidelines on physical activity and sedentary behaviour. Br J Sports Med "
        "2020;54:1451. 150-300 min/week moderate activity. [link only] https://bjsm.bmj.com/content/54/24/1451",
        "H8  Hirshkowitz M, et al. National Sleep Foundation's updated sleep duration recommendations. Sleep Health "
        "2015;1(1):40-43. Adults 7-9 h. [link only] https://doi.org/10.1016/j.sleh.2015.10.004",
        "H9  WHO definition of older persons (60+ years), used together with the ATP III age risk factor (H5).",
    ]
    pdf.set_font("Helvetica", "", 9.2)
    for r in refs_hc:
        ensure_room(pdf, 12)
        pdf.multi_cell(0, 4.6, r, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(0.8)
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(0, 5.3, "Banking", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.5)
    refs_bk = [
        "B1  Bahnsen AC, Aouada D, Stojanovic A, Ottersten B. Feature engineering strategies for credit card fraud "
        "detection. Expert Systems with Applications 2016;51:134-142. [PDF attached]",
        "B2  Dal Pozzolo A, Caelen O, Le Borgne Y-A, Waterschoot S, Bontempi G. Learned lessons in credit card fraud "
        "detection from a practitioner perspective. Expert Systems with Applications 2014. [PDF attached]",
        "B3  Whitrow C, Hand DJ, Juszczak P, Weston D, Adams NM. Transaction aggregation as a strategy for credit card "
        "fraud detection. Data Mining and Knowledge Discovery 2009;18:30-55. [PDF attached]",
    ]
    pdf.set_font("Helvetica", "", 9.2)
    for r in refs_bk:
        ensure_room(pdf, 12)
        pdf.multi_cell(0, 4.6, r, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(0.8)

    pdf.output(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
