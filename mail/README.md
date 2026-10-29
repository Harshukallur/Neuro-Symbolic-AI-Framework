# NSF threshold evidence pack

This folder contains the proof for how the numeric thresholds in the symbolic
rule base (`nsf/symbolic/rules.py`) were chosen, ready to attach to an email.

## What to send

- `NSF_Threshold_Evidence.pdf` - the main document. Maps every threshold in both
  domains to its published source, with an explicit, honest limitations section.
- `draft_email.md` - a ready-to-edit email body for your professor.
- `sources/` - the original guideline and research PDFs that could be downloaded.

## Threshold -> source map

### Healthcare (standard clinical cut-points)

| Rule | Threshold in code | Source | File |
|------|-------------------|--------|------|
| hypertension | systolic >= 140 or diastolic >= 90 mmHg | WHO 2021; ISH 2020 | `sources/ISH_2020_Global_Hypertension_Practice_Guidelines.pdf` (WHO 2021 link-only) |
| hyperglycemia | glucose >= 126 mg/dL | ADA Standards of Care 2026, Table 2.1 | `sources/ADA_2026_Diagnosis_Classification_Diabetes.pdf` |
| obesity | BMI >= 30 | WHO obesity definition | link-only (WHO fact sheet) |
| hypoxia | oxygen_sat < 94% | TSANZ 2016 / BTS 2017 / NICE | `sources/TSANZ_2016_Acute_Oxygen_Guidelines.pdf` |
| advanced_age | age >= 60 | WHO older-adult; ATP III age risk factor | `sources/NCEP_ATP_III_*` |
| sedentary / protective_exercise | < 1.5 h and >= 5 h per week | WHO 2020 PA (150-300 min/wk) | link-only (BJSM) |
| protective_sleep | 7-9 h | National Sleep Foundation 2015 | link-only (Sleep Health) |
| smoker / family_history | boolean | ATP III CVD risk factors | `sources/NCEP_ATP_III_*` |
| (cholesterol, used by neural model) | 200 / 240 mg/dL bands | NCEP ATP III | `sources/NCEP_ATP_III_*` |

### Banking (literature-informed, calibrated to the data generator)

The research papers justify the **features and direction**, not the exact numbers.
The numeric constants are engineering decision boundaries matched to the causal
generator in `nsf/data/generators.py`.

| Rule | Threshold in code | Supporting literature | File |
|------|-------------------|-----------------------|------|
| high_amount | amount >= 900 | Bahnsen 2016; Whitrow 2009 | `sources/Bahnsen_2016_*`, `sources/Whitrow_2009_*` |
| amount_spike | ratio >= 3.5 | Bahnsen 2016; Dal Pozzolo | `sources/Bahnsen_2016_*`, `sources/DalPozzolo_*` |
| far_from_home / geo_jump | >= 80 km / >= 60 km | Dal Pozzolo (geolocation / travel speed) | `sources/DalPozzolo_*` |
| txn_burst | >= 4 txn / 1 h | Whitrow 2009; Bahnsen 2016 (velocity) | `sources/Whitrow_2009_*`, `sources/Bahnsen_2016_*` |
| international / night_txn / high_risk_merchant / card_not_present | see code | Bahnsen 2016 (country, periodic time, merchant group, channel) | `sources/Bahnsen_2016_*` |
| failed_pin | >= 2 | operational/industry practice (weaker in academia) | - |

## Not auto-downloaded (publisher bot protection) - cited by link in the PDF

- WHO 2021 hypertension guideline - https://iris.who.int/
- WHO 2020 physical activity guideline (Bull et al., BJSM) - https://bjsm.bmj.com/content/54/24/1451
- WHO obesity & overweight fact sheet - https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight
- National Sleep Foundation 2015 (Hirshkowitz et al.) - https://doi.org/10.1016/j.sleh.2015.10.004

## Regenerating the evidence PDF

```powershell
.\.venv\Scripts\python.exe scripts\export_threshold_evidence_pdf.py
```
