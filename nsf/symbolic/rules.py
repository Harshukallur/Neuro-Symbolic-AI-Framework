"""Domain symbolic rules and logical constraints.

Each rule returns (fired: bool, evidence: str, risk_points: float).
Rules are deterministic and inspectable — no neural generation involved.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

RuleFn = Callable[[Dict[str, Any]], Tuple[bool, str, float]]


def _r(name: str, fn: RuleFn) -> Dict[str, Any]:
    return {"name": name, "fn": fn}


# ---------------------------------------------------------------------------
# Healthcare disease-risk rules (align with dataset causal generators)
# ---------------------------------------------------------------------------

def _hc_hypertension(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    s, d = float(f["systolic_bp"]), float(f["diastolic_bp"])
    if s >= 140 or d >= 90:
        return True, f"Hypertension indicators: BP {s}/{d} mmHg", 1.5
    return False, "", 0.0


def _hc_hyperglycemia(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    g = float(f["glucose_mg_dl"])
    if g >= 126:
        return True, f"Elevated fasting-range glucose: {g} mg/dL", 1.4
    return False, "", 0.0


def _hc_obesity(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    bmi = float(f["bmi"])
    if bmi >= 30:
        return True, f"Obesity range BMI: {bmi}", 1.0
    return False, "", 0.0


def _hc_smoker(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    if bool(f.get("smoker")):
        return True, "Current smoker status is positive", 1.2
    return False, "", 0.0


def _hc_family(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    if bool(f.get("family_history")):
        return True, "Positive family history of cardiovascular/metabolic disease", 1.0
    return False, "", 0.0


def _hc_hypoxia(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    o2 = float(f["oxygen_sat"])
    if o2 < 94:
        return True, f"Low oxygen saturation: {o2}%", 1.6
    return False, "", 0.0


def _hc_sedentary(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    ex = float(f["exercise_hours_week"])
    if ex < 1.5:
        return True, f"Low weekly exercise: {ex} hours", 0.6
    return False, "", 0.0


def _hc_age(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    age = int(f["age"])
    if age >= 60:
        return True, f"Age-related risk factor: age={age}", 0.8
    return False, "", 0.0


def _hc_protective_exercise(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    ex = float(f["exercise_hours_week"])
    if ex >= 5:
        return True, f"Protective exercise level: {ex} hours/week", -0.7
    return False, "", 0.0


def _hc_protective_sleep(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    sleep = float(f["sleep_hours"])
    if 7 <= sleep <= 9:
        return True, f"Healthy sleep duration: {sleep} hours", -0.4
    return False, "", 0.0


HEALTHCARE_RULES: List[Dict[str, Any]] = [
    _r("hypertension", _hc_hypertension),
    _r("hyperglycemia", _hc_hyperglycemia),
    _r("obesity", _hc_obesity),
    _r("smoker", _hc_smoker),
    _r("family_history", _hc_family),
    _r("hypoxia", _hc_hypoxia),
    _r("sedentary", _hc_sedentary),
    _r("advanced_age", _hc_age),
    _r("protective_exercise", _hc_protective_exercise),
    _r("protective_sleep", _hc_protective_sleep),
]


# Thresholds map symbolic risk points -> class labels
HEALTHCARE_THRESHOLDS = {"medium": 2.0, "high": 4.0}


# ---------------------------------------------------------------------------
# Banking fraud rules
# ---------------------------------------------------------------------------

def _bk_high_amount(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    amt = float(f["amount"])
    if amt >= 900:
        return True, f"High transaction amount: ${amt:.2f}", 1.2
    return False, "", 0.0


def _bk_amount_spike(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    ratio = float(f["amount_vs_avg_ratio"])
    if ratio >= 3.5:
        return True, f"Amount spike vs 30d average: ratio={ratio:.2f}", 1.5
    return False, "", 0.0


def _bk_far_from_home(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    d = float(f["distance_from_home_km"])
    if d >= 80:
        return True, f"Far from home location: {d:.1f} km", 1.2
    return False, "", 0.0


def _bk_geo_jump(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    d = float(f["distance_from_last_txn_km"])
    if d >= 60:
        return True, f"Large geo jump from last txn: {d:.1f} km", 1.0
    return False, "", 0.0


def _bk_burst(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    c = int(f["txn_count_1h"])
    if c >= 4:
        return True, f"Transaction burst in 1 hour: count={c}", 1.3
    return False, "", 0.0


def _bk_intl(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    if bool(f.get("is_international")):
        return True, "International transaction flag", 1.0
    return False, "", 0.0


def _bk_night(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    if bool(f.get("is_night")):
        return True, "Night-time transaction (00:00–05:59)", 0.8
    return False, "", 0.0


def _bk_pin_fail(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    n = int(f["failed_pin_attempts"])
    if n >= 2:
        return True, f"Multiple failed PIN attempts: {n}", 1.6
    return False, "", 0.0


def _bk_merchant(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    r = float(f["merchant_risk"])
    if r >= 0.65:
        return True, f"High-risk merchant score: {r:.3f}", 1.1
    return False, "", 0.0


def _bk_card_not_present(f: Dict[str, Any]) -> Tuple[bool, str, float]:
    if not bool(f.get("card_present")):
        return True, "Card-not-present transaction", 0.7
    return False, "", 0.0


BANKING_RULES: List[Dict[str, Any]] = [
    _r("high_amount", _bk_high_amount),
    _r("amount_spike", _bk_amount_spike),
    _r("far_from_home", _bk_far_from_home),
    _r("geo_jump", _bk_geo_jump),
    _r("txn_burst", _bk_burst),
    _r("international", _bk_intl),
    _r("night_txn", _bk_night),
    _r("failed_pin", _bk_pin_fail),
    _r("high_risk_merchant", _bk_merchant),
    _r("card_not_present", _bk_card_not_present),
]

BANKING_FRAUD_THRESHOLD = 3.0
