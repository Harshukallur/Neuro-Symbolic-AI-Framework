"""Interactive-style demo: run verified inference on sample cases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsf.bridge.neural_symbolic import NeuralSymbolicBridge
from nsf.data.generators import load_domain_data
from nsf.llm.explainer import VerifiedLLMExplainer
from nsf.neural.models import PerceptionMLP
from nsf.symbolic.engine import SymbolicReasoner
from nsf.utils import load_config


def run_domain(domain: str, cfg: dict, n: int = 3) -> None:
    data_dir = ROOT / cfg["paths"]["data_dir"]
    models_dir = ROOT / cfg["paths"]["models_dir"]
    dcfg = cfg["domains"][domain]
    data = load_domain_data(domain, data_dir, cfg["batch_size"])

    # Prefer best strategy from summary if available
    summary_path = ROOT / cfg["paths"]["results_dir"] / "summary.json"
    strategy = "fedavg"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        strategy = summary["domains"][domain].get("best_strategy") or "fedavg"

    model_path = models_dir / f"{domain}_{strategy}.pt"
    if not model_path.exists():
        # fallback any available
        candidates = list(models_dir.glob(f"{domain}_*.pt"))
        if not candidates:
            print(f"No trained model for {domain}. Run scripts/train_and_evaluate.py first.")
            return
        model_path = candidates[0]
        strategy = model_path.stem.split("_", 1)[1]

    model = PerceptionMLP(dcfg["feature_dim"], dcfg["num_classes"])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    id_key = "patient_id" if domain == "healthcare" else "txn_id"
    bridge = NeuralSymbolicBridge(domain, dcfg["class_names"], id_key)
    reasoner = SymbolicReasoner(domain, dcfg["class_names"])
    explainer = VerifiedLLMExplainer()

    features = data["features"]
    test_df = data["test_df"]
    x = torch.tensor(test_df[features].values, dtype=torch.float32)
    x = (x - data["feat_mean"]) / data["feat_std"]

    print("=" * 72)
    print(f"DOMAIN: {domain.upper()} | model={model_path.name}")
    print("=" * 72)

    with torch.no_grad():
        probs_all = model.predict_proba(x).numpy()

    for i in range(min(n, len(test_df))):
        probs = probs_all[i].tolist()
        npred = int(max(range(len(probs)), key=lambda j: probs[j]))
        raw = dict(data["test_facts"][i])
        fact = bridge.convert(raw, npred, probs, entity_id=raw.get(id_key))
        verified = reasoner.verify(fact)
        trace = explainer.explain(verified)
        print(f"\n--- Case {i+1}: {raw.get(id_key)} ---")
        print(f"True label     : {dcfg['class_names'][raw['true_label']]}")
        print(f"Neural pred    : {dcfg['class_names'][npred]} ({max(probs):.3f})")
        print(f"Symbolic pred  : {dcfg['class_names'][verified.symbolic_prediction]} (score={verified.symbolic_score})")
        print(f"Final decision : {verified.final_label} (conf={verified.confidence:.3f})")
        print(f"Conflict/Override: {verified.conflict}/{verified.override_applied}")
        print(f"Fired rules    : {[r['rule'] for r in verified.fired_rules]}")
        print(f"Explanation    : {trace.explanation}")
        print(f"Hallucination guard passed: {trace.hallucination_guard['passed']}")


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    for domain in ("healthcare", "banking"):
        run_domain(domain, cfg, n=3)


if __name__ == "__main__":
    main()
