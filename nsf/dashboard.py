"""Local dashboard data: saved federated models, symbolic checks, explanations."""

from __future__ import annotations

import hmac
import json
import math
import re
import secrets
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from nsf.bridge.neural_symbolic import NeuralSymbolicBridge
from nsf.data.generators import BANKING_FEATURES, HEALTHCARE_FEATURES
from nsf.llm.explainer import VerifiedLLMExplainer
from nsf.neural.models import PerceptionMLP
from nsf.symbolic.engine import SymbolicReasoner
from nsf.utils import load_config

FIELD_SPECS: Dict[str, List[Dict[str, str]]] = {
    "healthcare": [
        {"name": "age", "label": "Age", "unit": "years", "kind": "int"},
        {"name": "bmi", "label": "BMI", "unit": "", "kind": "float"},
        {"name": "systolic_bp", "label": "Systolic blood pressure", "unit": "mmHg", "kind": "float"},
        {"name": "diastolic_bp", "label": "Diastolic blood pressure", "unit": "mmHg", "kind": "float"},
        {"name": "glucose_mg_dl", "label": "Glucose", "unit": "mg/dL", "kind": "float"},
        {"name": "cholesterol_mg_dl", "label": "Cholesterol", "unit": "mg/dL", "kind": "float"},
        {"name": "heart_rate", "label": "Heart rate", "unit": "bpm", "kind": "float"},
        {"name": "oxygen_sat", "label": "Oxygen saturation", "unit": "%", "kind": "float"},
        {"name": "smoker", "label": "Current smoker", "unit": "", "kind": "bool"},
        {"name": "family_history", "label": "Family history", "unit": "", "kind": "bool"},
        {"name": "exercise_hours_week", "label": "Exercise", "unit": "hours/week", "kind": "float"},
        {"name": "sleep_hours", "label": "Sleep", "unit": "hours", "kind": "float"},
    ],
    "banking": [
        {"name": "amount", "label": "Amount", "unit": "currency", "kind": "float"},
        {"name": "hour_of_day", "label": "Hour of day", "unit": "0–23", "kind": "int"},
        {"name": "day_of_week", "label": "Day of week", "unit": "0–6", "kind": "int"},
        {"name": "merchant_risk", "label": "Merchant risk", "unit": "0–1", "kind": "float"},
        {"name": "distance_from_home_km", "label": "Distance from home", "unit": "km", "kind": "float"},
        {"name": "distance_from_last_txn_km", "label": "Distance from last transaction", "unit": "km", "kind": "float"},
        {"name": "txn_count_1h", "label": "Transactions in 1 hour", "unit": "", "kind": "int"},
        {"name": "txn_count_24h", "label": "Transactions in 24 hours", "unit": "", "kind": "int"},
        {"name": "avg_amount_30d", "label": "Average amount, 30 days", "unit": "currency", "kind": "float"},
        {"name": "amount_vs_avg_ratio", "label": "Amount vs 30-day average", "unit": "ratio", "kind": "float"},
        {"name": "is_international", "label": "International", "unit": "", "kind": "bool"},
        {"name": "is_night", "label": "Night transaction", "unit": "", "kind": "bool"},
        {"name": "card_present", "label": "Card present", "unit": "", "kind": "bool"},
        {"name": "failed_pin_attempts", "label": "Failed PIN attempts", "unit": "", "kind": "int"},
    ],
}

BOOL_FEATURES = {
    "healthcare": {"smoker", "family_history"},
    "banking": {"is_international", "is_night", "card_present"},
}
INT_FEATURES = {
    "healthcare": {"age"},
    "banking": {"hour_of_day", "day_of_week", "txn_count_1h", "txn_count_24h", "failed_pin_attempts"},
}
FEATURE_ORDER = {"healthcare": HEALTHCARE_FEATURES, "banking": BANKING_FEATURES}
ID_KEY = {"healthcare": "patient_id", "banking": "txn_id"}
SKIP_FEATURES = {"patient_id", "txn_id", "true_label"}
ENTITY_RE = re.compile(r"[A-Za-z0-9_-]{1,40}")


class DashboardError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def _plain(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    raise ValueError


def _as_float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str) and value.strip():
        number = float(value.strip())
    else:
        raise ValueError
    if math.isnan(number) or math.isinf(number):
        raise ValueError
    return number


def _as_int(value: Any) -> int:
    return int(round(_as_float(value)))


class DashboardStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.cfg = load_config(self.root / "configs" / "default.yaml")
        self.data_dir = self.root / self.cfg["paths"]["data_dir"]
        self.results_dir = self.root / self.cfg["paths"]["results_dir"]
        self.models_dir = self.root / self.cfg["paths"]["models_dir"]
        self.plots_dir = self.results_dir / "plots"
        self._lock = threading.Lock()
        self._cases: Dict[tuple, List[dict]] = {}
        self._client_data: Dict[tuple, Any] = {}
        self._client_cases: Dict[tuple, List[dict]] = {}
        # Demo-grade client authentication: per-client password + in-memory
        # session tokens. A client endpoint derives the client id from the
        # token, so the browser can never request another client's data.
        self.num_clients = int(self.cfg.get("num_clients", 5))
        self.logins = self._load_or_create_logins()
        self.sessions: Dict[str, int] = {}
        self.data: Dict[str, Dict[str, Any]] = {}
        self.models: Dict[tuple, torch.nn.Module] = {}
        self.bridges: Dict[str, NeuralSymbolicBridge] = {}
        self.reasoners: Dict[str, SymbolicReasoner] = {}
        self.explainer = VerifiedLLMExplainer()
        self.summary: Dict[str, Any] = {}
        self.details: Dict[tuple, Dict[str, Any]] = {}

        for domain, specs in FIELD_SPECS.items():
            names = [field["name"] for field in specs]
            if names != list(FEATURE_ORDER[domain]):
                raise RuntimeError(f"Field order does not match {domain} features")

    def load(self) -> None:
        summary_path = self.results_dir / "summary.json"
        if summary_path.exists():
            self.summary = json.loads(summary_path.read_text(encoding="utf-8"))

        missing_models = []
        for domain in ("healthcare", "banking"):
            self._load_domain(domain)
            class_names = self.cfg["domains"][domain]["class_names"]
            self.bridges[domain] = NeuralSymbolicBridge(domain, class_names, ID_KEY[domain])
            self.reasoners[domain] = SymbolicReasoner(domain, class_names)
            for strategy in self.cfg["federated_strategies"]:
                model_path = self.models_dir / f"{domain}_{strategy}.pt"
                result_path = self.results_dir / f"{domain}_{strategy}_results.json"
                if not model_path.exists() or not result_path.exists():
                    missing_models.append(model_path.name)
                    continue
                self.models[(domain, strategy)] = self._load_model(domain, model_path)
                self.details[(domain, strategy)] = json.loads(result_path.read_text(encoding="utf-8"))

        if missing_models:
            names = ", ".join(missing_models)
            raise RuntimeError(
                f"Missing trained results ({names}). Run: python scripts/train_and_evaluate.py"
            )

    def _load_domain(self, domain: str) -> None:
        domain_dir = self.data_dir / domain
        test_csv = domain_dir / "test.csv"
        facts_path = domain_dir / "test_facts.json"
        meta_path = domain_dir / "metadata.json"
        train_csv = domain_dir / "train.csv"
        if not test_csv.exists() or not facts_path.exists() or not train_csv.exists():
            raise RuntimeError(
                f"Missing {domain} dataset. Run: python scripts/generate_datasets.py"
            )
        features = FEATURE_ORDER[domain]
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        train_x = torch.tensor(train_df[features].values, dtype=torch.float32)
        self.data[domain] = {
            "test_df": test_df,
            "test_facts": json.loads(facts_path.read_text(encoding="utf-8")),
            "feat_mean": train_x.mean(dim=0),
            "feat_std": train_x.std(dim=0).clamp_min(1e-6),
            "meta": json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {},
        }
        if len(test_df) != len(self.data[domain]["test_facts"]):
            raise RuntimeError(f"{domain} test rows and facts differ in length")

    def _load_model(self, domain: str, path: Path) -> torch.nn.Module:
        dcfg = self.cfg["domains"][domain]
        model = PerceptionMLP(dcfg["feature_dim"], dcfg["num_classes"])
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    def overview(self) -> Dict[str, Any]:
        domains = {}
        for domain in ("healthcare", "banking"):
            domains[domain] = self._domain_overview(domain)
        return {"domains": domains}

    def _domain_overview(self, domain: str) -> Dict[str, Any]:
        dcfg = self.cfg["domains"][domain]
        class_names = list(dcfg["class_names"])
        meta = self.data[domain]["meta"]
        strategies = []
        best_acc = -1.0
        best = self.cfg["federated_strategies"][0]
        for strategy in self.cfg["federated_strategies"]:
            packed = self._strategy_public(domain, strategy, class_names)
            strategies.append(packed)
            if packed["neurosymbolic_accuracy"] > best_acc:
                best_acc = packed["neurosymbolic_accuracy"]
                best = strategy
        summary_best = (
            self.summary.get("domains", {}).get(domain, {}).get("best_strategy")
        )
        if summary_best in self.cfg["federated_strategies"]:
            best = summary_best

        test_counts = [
            int(meta.get("label_distribution_test", {}).get(str(i), 0))
            for i in range(len(class_names))
        ]
        clients = []
        for client in meta.get("clients", []):
            counts = client.get("label_counts", {})
            clients.append(
                {
                    "client_id": int(client["client_id"]),
                    "n_samples": int(client["n_samples"]),
                    "counts": [int(counts.get(str(i), 0)) for i in range(len(class_names))],
                }
            )
        facts0 = self.data[domain]["test_facts"][0]
        row0 = self.data[domain]["test_df"].iloc[0]
        compare = "neural_vs_neurosymbolic.png"
        accuracy_plot = f"{domain}_accuracy_curves.png"
        return {
            "id": domain,
            "title": "Healthcare" if domain == "healthcare" else "Banking",
            "task": dcfg["task"],
            "class_names": class_names,
            "best_strategy": best,
            "test_size": int(meta.get("test_size", len(self.data[domain]["test_df"]))),
            "train_size": int(meta.get("train_size", 0)),
            "num_clients": int(meta.get("num_clients", 0)),
            "non_iid_alpha": meta.get("non_iid_alpha"),
            "test_counts": test_counts,
            "clients": clients,
            "fields": FIELD_SPECS[domain],
            "example": {
                "entity_id": str(facts0.get(ID_KEY[domain], "EXAMPLE")),
                "inputs": self._inputs_from_row(domain, row0),
            },
            "strategies": strategies,
            "plots": {
                "accuracy": accuracy_plot if (self.plots_dir / accuracy_plot).is_file() else None,
                "compare": compare if (self.plots_dir / compare).is_file() else None,
            },
        }

    def _strategy_public(self, domain: str, strategy: str, class_names: List[str]) -> Dict[str, Any]:
        bundle = self.details[(domain, strategy)]
        federated = bundle["federated"]
        evaluation = bundle["evaluation"]
        neural = evaluation["neural_metrics"]
        final = evaluation["neurosymbolic_metrics"]
        history = [
            {
                "round": int(item["round"]),
                "accuracy": float(item["accuracy"]),
                "loss": float(item["loss"]),
            }
            for item in federated.get("history", [])
        ]
        return {
            "id": strategy,
            "neural_accuracy": float(neural["accuracy"]),
            "neurosymbolic_accuracy": float(final["accuracy"]),
            "neural_f1": float(neural["f1_macro"]),
            "neurosymbolic_f1": float(final["f1_macro"]),
            "conflicts": int(evaluation["conflicts"]),
            "overrides": int(evaluation["overrides"]),
            "hallucination_free_ratio": float(evaluation["explanation_hallucination_free_ratio"]),
            "communication_mb": float(federated["communication_mb"]),
            "wall_time_sec": float(federated["wall_time_sec"]),
            "history": history,
            "confusion_neural": neural["confusion_matrix"],
            "confusion_neurosymbolic": final["confusion_matrix"],
            "class_names": class_names,
        }

    def cases(self, domain: str, strategy: str) -> List[dict]:
        self._require(domain, strategy)
        key = (domain, strategy)
        with self._lock:
            cached = self._cases.get(key)
            if cached is None:
                cached = self._score(domain, strategy)
                self._cases[key] = cached
            return cached

    def case(self, domain: str, strategy: str, index: int) -> dict:
        records = self.cases(domain, strategy)
        if index < 0 or index >= len(records):
            raise DashboardError("Case index is out of range", 404)
        return records[index]

    # ------------------------------------------------------------------
    # Client authentication (demo-grade: local password file + in-memory
    # tokens, no TLS). Enough to enforce that a logged-in client can only
    # pull its own partition.
    # ------------------------------------------------------------------
    def _load_or_create_logins(self) -> Dict[str, str]:
        path = self.root / "client_logins.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data:
                    return {str(k): str(v) for k, v in data.items()}
            except (json.JSONDecodeError, OSError):
                pass
        data = {str(i): secrets.token_hex(3) for i in range(self.num_clients)}
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass
        return data

    def login_hint(self) -> Dict[str, str]:
        """Return the demo credentials so the operator can share them."""
        return dict(self.logins)

    def login(self, client_id: Any, password: Any) -> Dict[str, Any]:
        try:
            cid = int(client_id)
        except (TypeError, ValueError):
            raise DashboardError("Invalid client id or password", 401) from None
        expected = self.logins.get(str(cid))
        if expected is None or not hmac.compare_digest(str(password or ""), expected):
            raise DashboardError("Invalid client id or password", 401)
        token = secrets.token_urlsafe(24)
        with self._lock:
            self.sessions[token] = cid
        return {"token": token, "client_id": cid, "num_clients": self.num_clients}

    def logout(self, token: Any) -> Dict[str, Any]:
        if isinstance(token, str):
            with self._lock:
                self.sessions.pop(token, None)
        return {"ok": True}

    def client_from_token(self, token: Any) -> int:
        if not isinstance(token, str) or not token:
            raise DashboardError("Login required", 401)
        with self._lock:
            cid = self.sessions.get(token)
        if cid is None:
            raise DashboardError("Session expired. Please log in again.", 401)
        return int(cid)

    # ------------------------------------------------------------------
    # Client (single-participant) views. A client only ever sees its own
    # partition - never another client's data or aggregate counts.
    # ------------------------------------------------------------------
    def _client_frame(self, domain: str, client_id: int):
        if domain not in self.cfg["domains"]:
            raise DashboardError(f"Unknown domain: {domain}", 404)
        num_clients = int(self.data[domain]["meta"].get("num_clients", 0))
        if client_id < 0 or client_id >= num_clients:
            raise DashboardError("Client id is out of range", 404)
        key = (domain, client_id)
        cached = self._client_data.get(key)
        if cached is None:
            domain_dir = self.data_dir / domain
            csv = domain_dir / f"client_{client_id}.csv"
            facts_path = domain_dir / f"client_{client_id}_facts.json"
            if not csv.exists() or not facts_path.exists():
                raise DashboardError(f"Missing data for client {client_id}", 404)
            df = pd.read_csv(csv)
            facts = json.loads(facts_path.read_text(encoding="utf-8"))
            cached = (df, facts)
            self._client_data[key] = cached
        return cached

    def client_overview(self, domain: str, token: Any) -> Dict[str, Any]:
        client_id = self.client_from_token(token)
        df, facts = self._client_frame(domain, client_id)
        class_names = list(self.cfg["domains"][domain]["class_names"])
        counts = [int((df["label"] == i).sum()) for i in range(len(class_names))]
        row0 = df.iloc[0]
        facts0 = facts[0]
        return {
            "domain": domain,
            "client_id": int(client_id),
            "num_clients": int(self.data[domain]["meta"].get("num_clients", 0)),
            "n_samples": int(len(df)),
            "class_names": class_names,
            "counts": counts,
            "fields": FIELD_SPECS[domain],
            "example": {
                "entity_id": str(facts0.get(ID_KEY[domain], "EXAMPLE")),
                "inputs": self._inputs_from_row(domain, row0),
            },
        }

    def client_cases(self, domain: str, strategy: str, token: Any) -> List[dict]:
        client_id = self.client_from_token(token)
        self._require(domain, strategy)
        key = (domain, strategy, client_id)
        with self._lock:
            cached = self._client_cases.get(key)
            if cached is None:
                cached = self._score_client(domain, strategy, client_id)
                self._client_cases[key] = cached
            return cached

    def _score_client(self, domain: str, strategy: str, client_id: int) -> List[dict]:
        df, facts = self._client_frame(domain, client_id)
        features = FEATURE_ORDER[domain]
        data = self.data[domain]
        x = torch.tensor(df[features].values, dtype=torch.float32)
        x = (x - data["feat_mean"]) / data["feat_std"]
        model = self.models[(domain, strategy)]
        with torch.no_grad():
            probs_all = model.predict_proba(x).numpy()
        records = []
        class_names = self.cfg["domains"][domain]["class_names"]
        for index, probs in enumerate(probs_all):
            raw = dict(facts[index])
            true_label = class_names[int(raw["true_label"])]
            inputs = self._inputs_from_row(domain, df.iloc[index])
            records.append(
                self._run(domain, strategy, index, raw, probs.tolist(), inputs, true_label)
            )
        return records

    def predict(
        self,
        domain: Any,
        strategy: Any,
        inputs: Any,
        entity_id: Any = None,
    ) -> dict:
        if not isinstance(domain, str) or not isinstance(strategy, str):
            raise DashboardError("domain and strategy are required")
        self._require(domain, strategy)
        if not isinstance(inputs, dict):
            raise DashboardError("inputs must be an object")
        cleaned, raw = self._parse_inputs(domain, inputs, entity_id)
        values = []
        for name in FEATURE_ORDER[domain]:
            if name in BOOL_FEATURES[domain]:
                values.append(1.0 if cleaned[name] else 0.0)
            else:
                values.append(float(cleaned[name]))
        with self._lock:
            probs = self._forward(domain, strategy, values)
            return self._run(domain, strategy, None, raw, probs, cleaned, None)

    def plot_path(self, name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+\.png", name or ""):
            raise DashboardError("Not found", 404)
        path = (self.plots_dir / name).resolve()
        root = self.plots_dir.resolve()
        if root not in path.parents or not path.is_file():
            raise DashboardError("Not found", 404)
        return path

    def _require(self, domain: str, strategy: str) -> None:
        if domain not in self.cfg["domains"]:
            raise DashboardError(f"Unknown domain: {domain}", 404)
        if strategy not in self.cfg["federated_strategies"]:
            raise DashboardError(f"Unknown strategy: {strategy}", 404)
        if (domain, strategy) not in self.models:
            raise DashboardError(f"No model for {domain}/{strategy}", 404)

    def _score(self, domain: str, strategy: str) -> List[dict]:
        data = self.data[domain]
        features = FEATURE_ORDER[domain]
        frame = data["test_df"]
        x = torch.tensor(frame[features].values, dtype=torch.float32)
        x = (x - data["feat_mean"]) / data["feat_std"]
        model = self.models[(domain, strategy)]
        with torch.no_grad():
            probs_all = model.predict_proba(x).numpy()
        records = []
        for index, probs in enumerate(probs_all):
            raw = dict(data["test_facts"][index])
            true_label = self.cfg["domains"][domain]["class_names"][int(raw["true_label"])]
            inputs = self._inputs_from_row(domain, frame.iloc[index])
            records.append(
                self._run(
                    domain,
                    strategy,
                    index,
                    raw,
                    probs.tolist(),
                    inputs,
                    true_label,
                )
            )
        return records

    def _forward(self, domain: str, strategy: str, values: List[float]) -> List[float]:
        mean = self.data[domain]["feat_mean"]
        std = self.data[domain]["feat_std"]
        x = torch.tensor([values], dtype=torch.float32)
        x = (x - mean) / std
        model = self.models[(domain, strategy)]
        with torch.no_grad():
            return model.predict_proba(x).numpy()[0].tolist()

    def _parse_inputs(self, domain: str, inputs: Dict[str, Any], entity_id: Any):
        missing = [name for name in FEATURE_ORDER[domain] if name not in inputs]
        if missing:
            raise DashboardError("Missing fields: " + ", ".join(missing))
        cleaned: Dict[str, Any] = {}
        for spec in FIELD_SPECS[domain]:
            name = spec["name"]
            try:
                if spec["kind"] == "bool":
                    cleaned[name] = _as_bool(inputs[name])
                elif spec["kind"] == "int":
                    cleaned[name] = _as_int(inputs[name])
                else:
                    cleaned[name] = _as_float(inputs[name])
            except (TypeError, ValueError):
                raise DashboardError(f"Invalid value for {spec['label']}") from None
        eid = "CUSTOM" if entity_id is None else str(entity_id).strip() or "CUSTOM"
        if not ENTITY_RE.fullmatch(eid):
            raise DashboardError("Entity id must be 1–40 letters, numbers, underscores, or hyphens")
        raw = dict(cleaned)
        raw[ID_KEY[domain]] = eid
        return cleaned, raw

    def _inputs_from_row(self, domain: str, row: Any) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}
        for name in FEATURE_ORDER[domain]:
            value = row[name]
            if name in BOOL_FEATURES[domain]:
                inputs[name] = bool(int(value))
            elif name in INT_FEATURES[domain]:
                inputs[name] = int(value)
            else:
                inputs[name] = float(value)
        return inputs

    def _run(
        self,
        domain: str,
        strategy: str,
        index: Optional[int],
        raw: Dict[str, Any],
        probs: List[float],
        inputs: Dict[str, Any],
        true_label: Optional[str],
    ) -> dict:
        class_names = self.cfg["domains"][domain]["class_names"]
        entity_id = str(raw.get(ID_KEY[domain], "unknown"))
        npred = int(max(range(len(probs)), key=lambda j: probs[j]))
        fact = self.bridges[domain].convert(raw, npred, [float(p) for p in probs], entity_id=entity_id)
        verified = self.reasoners[domain].verify(fact)
        trace = self.explainer.explain(verified)
        guard = trace.hallucination_guard
        features = {
            key: _plain(value)
            for key, value in verified.features_used.items()
            if key not in SKIP_FEATURES
        }
        final_label = verified.final_label
        return {
            "index": index,
            "entity_id": entity_id,
            "domain": domain,
            "strategy": strategy,
            "model_file": f"{domain}_{strategy}.pt",
            "true_label": true_label,
            "neural_label": class_names[verified.neural_prediction],
            "neural_confidence": float(verified.neural_confidence),
            "probabilities": [
                {"label": class_names[i], "value": float(probs[i])} for i in range(len(class_names))
            ],
            "symbolic_label": class_names[verified.symbolic_prediction],
            "symbolic_score": float(verified.symbolic_score),
            "final_label": final_label,
            "confidence": float(verified.confidence),
            "conflict": bool(verified.conflict),
            "override_applied": bool(verified.override_applied),
            "verified": bool(verified.verified),
            "correct": None if true_label is None else final_label == true_label,
            "fired_rules": verified.fired_rules,
            "warnings": list(verified.warnings),
            "explanation": trace.explanation,
            "hallucination_guard": {
                "passed": bool(guard.get("passed")),
                "illegal_rules": list(guard.get("illegal_rules") or []),
                "entity_present": bool(guard.get("entity_present")),
            },
            "features": features,
            "inputs": inputs,
        }
