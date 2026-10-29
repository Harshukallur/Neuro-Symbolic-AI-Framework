"""Re-evaluate saved models with the current symbolic engine."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsf.data.generators import load_domain_data
from nsf.neural.models import PerceptionMLP
from nsf.pipeline import NeuroSymbolicFramework
from nsf.utils import load_config, save_json


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    for key in ("data_dir", "results_dir", "models_dir"):
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            cfg["paths"][key] = str(ROOT / p)

    fw = NeuroSymbolicFramework(cfg)
    report = {}
    for domain in ("healthcare", "banking"):
        data = load_domain_data(domain, cfg["paths"]["data_dir"], cfg["batch_size"])
        dcfg = cfg["domains"][domain]
        report[domain] = {}
        for strategy in cfg["federated_strategies"]:
            mp = Path(cfg["paths"]["models_dir"]) / f"{domain}_{strategy}.pt"
            model = PerceptionMLP(dcfg["feature_dim"], dcfg["num_classes"])
            model.load_state_dict(torch.load(mp, map_location="cpu"))
            ev = fw.infer_and_verify(domain, model, data, max_examples=30)
            row = {
                "neural_accuracy": ev["neural_metrics"]["accuracy"],
                "neurosymbolic_accuracy": ev["neurosymbolic_metrics"]["accuracy"],
                "hallucination_free_ratio": ev["explanation_hallucination_free_ratio"],
                "overrides": ev["overrides"],
                "conflicts": ev["conflicts"],
            }
            report[domain][strategy] = row
            print(
                f"{domain:12s} {strategy:12s} "
                f"neural={row['neural_accuracy']:.4f} "
                f"neurosym={row['neurosymbolic_accuracy']:.4f} "
                f"hallu_free={row['hallucination_free_ratio']:.2%} "
                f"overrides={row['overrides']} conflicts={row['conflicts']}"
            )
            save_json(
                ev["sample_traces"],
                Path(cfg["paths"]["results_dir"]) / f"{domain}_{strategy}_traces.json",
            )
    save_json(report, Path(cfg["paths"]["results_dir"]) / "reeval_summary.json")


if __name__ == "__main__":
    main()
