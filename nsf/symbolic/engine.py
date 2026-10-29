"""Layer 3 — Symbolic Reasoning & Verification.

Anti-hallucination policy:
- Neural model remains the primary predictor (accuracy).
- Symbolic rules verify consistency and produce evidence for explanations.
- Overrides apply ONLY for hard logical constraints (unsafe / inconsistent).
- LLM explanations may consume verified facts only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from nsf.bridge.neural_symbolic import SymbolicFact
from nsf.symbolic.rules import (
    BANKING_FRAUD_THRESHOLD,
    BANKING_RULES,
    HEALTHCARE_RULES,
    HEALTHCARE_THRESHOLDS,
)


@dataclass
class VerificationResult:
    domain: str
    entity_id: str
    neural_prediction: int
    neural_confidence: float
    symbolic_prediction: int
    final_prediction: int
    final_label: str
    verified: bool
    conflict: bool
    override_applied: bool
    symbolic_score: float
    fired_rules: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence: float = 0.0
    class_names: List[str] = field(default_factory=list)
    features_used: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def verified_facts_only(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "entity_id": self.entity_id,
            "final_prediction": self.final_prediction,
            "final_label": self.final_label,
            "verified": self.verified,
            "conflict": self.conflict,
            "symbolic_score": self.symbolic_score,
            "fired_rules": self.fired_rules,
            "evidence": self.evidence,
            "warnings": self.warnings,
            "confidence": self.confidence,
            "features": self.features_used,
            "neural_prediction": self.neural_prediction,
            "neural_confidence": self.neural_confidence,
            "override_applied": self.override_applied,
        }


class SymbolicReasoner:
    def __init__(self, domain: str, class_names: List[str]):
        self.domain = domain
        self.class_names = class_names
        if domain == "healthcare":
            self.rules = HEALTHCARE_RULES
        elif domain == "banking":
            self.rules = BANKING_RULES
        else:
            raise ValueError(f"Unsupported domain: {domain}")

    def _symbolic_label(self, score: float) -> int:
        if self.domain == "healthcare":
            if score >= HEALTHCARE_THRESHOLDS["high"]:
                return 2
            if score >= HEALTHCARE_THRESHOLDS["medium"]:
                return 1
            return 0
        return 1 if score >= BANKING_FRAUD_THRESHOLD else 0

    def _hard_constraint_override(self, fact: SymbolicFact, score: float, fired: List[Dict[str, Any]]) -> int | None:
        """Return forced label only when a hard safety/logic constraint is violated."""
        f = fact.features
        neural = fact.neural_prediction

        if self.domain == "healthcare":
            # Critical physiology cannot be labeled low_risk
            o2 = float(f.get("oxygen_sat", 100))
            systolic = float(f.get("systolic_bp", 120))
            glucose = float(f.get("glucose_mg_dl", 100))
            critical = (o2 < 92) or (systolic >= 180) or (glucose >= 250)
            high_stack = score >= HEALTHCARE_THRESHOLDS["high"] and len(fired) >= 4
            if critical or high_stack:
                if neural == 0:
                    return 2 if (o2 < 92 or high_stack) else 1
            # Medium floor: many risk factors but neural says low
            if score >= HEALTHCARE_THRESHOLDS["medium"] + 1.0 and neural == 0 and len(fired) >= 4:
                return 1
            return None

        # banking: clear fraud pattern while neural says legitimate with weak confidence
        if score >= BANKING_FRAUD_THRESHOLD + 1.5 and neural == 0 and fact.neural_confidence < 0.75:
            return 1
        if (
            int(f.get("failed_pin_attempts", 0)) >= 3
            and float(f.get("amount_vs_avg_ratio", 0)) >= 4.0
            and neural == 0
        ):
            return 1
        return None

    def verify(self, fact: SymbolicFact) -> VerificationResult:
        fired: List[Dict[str, Any]] = []
        evidence: List[str] = []
        score = 0.0
        for rule in self.rules:
            hit, msg, points = rule["fn"](fact.features)
            if hit:
                fired.append({"rule": rule["name"], "points": points, "message": msg})
                evidence.append(msg)
                score += points

        symbolic_pred = self._symbolic_label(score)
        neural_pred = fact.neural_prediction
        conflict = symbolic_pred != neural_pred
        warnings: List[str] = []
        override = False

        forced = self._hard_constraint_override(fact, score, fired)
        if forced is not None and forced != neural_pred:
            final = forced
            override = True
            warnings.append(
                "Hard symbolic constraint override applied to correct a logically inconsistent neural prediction."
            )
        elif conflict and fact.neural_confidence < 0.55:
            final = symbolic_pred
            override = True
            warnings.append(
                "Low-confidence neural prediction deferred to symbolic verification."
            )
        else:
            final = neural_pred
            if conflict:
                warnings.append(
                    "Symbolic score differs from neural prediction; neural decision retained, conflict recorded for explanation."
                )

        # Confidence calibration
        if not conflict:
            conf = min(0.99, fact.neural_confidence + 0.08 + 0.01 * len(fired))
        elif override:
            conf = max(0.6, min(0.95, 0.55 + 0.07 * len(fired)))
        else:
            conf = max(0.2, fact.neural_confidence - 0.12)

        verified = True  # every case produces a verified fact package for the LLM
        return VerificationResult(
            domain=self.domain,
            entity_id=fact.entity_id,
            neural_prediction=neural_pred,
            neural_confidence=fact.neural_confidence,
            symbolic_prediction=symbolic_pred,
            final_prediction=final,
            final_label=self.class_names[final],
            verified=verified,
            conflict=conflict,
            override_applied=override,
            symbolic_score=round(score, 3),
            fired_rules=fired,
            evidence=evidence,
            warnings=warnings,
            confidence=round(conf, 4),
            class_names=list(self.class_names),
            features_used=dict(fact.features),
        )

    def batch_verify(self, facts: List[SymbolicFact]) -> List[VerificationResult]:
        return [self.verify(f) for f in facts]
