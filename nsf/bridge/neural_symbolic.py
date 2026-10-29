"""Layer 2 — Neural → Symbolic Bridge.

Converts neural predictions + raw features into structured symbolic facts
that the reasoning engine can verify. No free-form text enters symbolic logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SymbolicFact:
    domain: str
    entity_id: str
    features: Dict[str, Any]
    neural_prediction: int
    neural_confidence: float
    class_names: List[str]
    neural_probs: List[float] = field(default_factory=list)

    def predicted_label_name(self) -> str:
        return self.class_names[self.neural_prediction]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NeuralSymbolicBridge:
    """Maps neural outputs to typed symbolic objects."""

    def __init__(self, domain: str, class_names: List[str], id_key: str):
        self.domain = domain
        self.class_names = class_names
        self.id_key = id_key

    def convert(
        self,
        raw_features: Dict[str, Any],
        neural_pred: int,
        neural_probs: List[float],
        entity_id: Optional[str] = None,
    ) -> SymbolicFact:
        eid = entity_id or str(raw_features.get(self.id_key, "unknown"))
        conf = float(max(neural_probs)) if neural_probs else 0.0
        # Keep only numeric/bool feature facts (no narrative strings)
        clean = {
            k: (bool(v) if isinstance(v, (bool,)) else v)
            for k, v in raw_features.items()
            if k not in ("true_label",) and not isinstance(v, (list, dict))
        }
        return SymbolicFact(
            domain=self.domain,
            entity_id=eid,
            features=clean,
            neural_prediction=int(neural_pred),
            neural_confidence=conf,
            class_names=list(self.class_names),
            neural_probs=[float(p) for p in neural_probs],
        )
