"""Train federated neuro-symbolic models for both domains and evaluate."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table

from nsf.pipeline import NeuroSymbolicFramework
from nsf.utils import load_config

console = Console()


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    # Resolve relative paths against project root
    for key in ("data_dir", "results_dir", "models_dir"):
        p = Path(cfg["paths"][key])
        if not p.is_absolute():
            cfg["paths"][key] = str(ROOT / p)

    framework = NeuroSymbolicFramework(cfg)
    summary = framework.run_all()

    table = Table(title="Federated Neuro-Symbolic Results")
    table.add_column("Domain")
    table.add_column("Strategy")
    table.add_column("Neural Acc")
    table.add_column("NeuroSym Acc")
    table.add_column("Hallucination-Free")
    table.add_column("Comm (MB)")

    for domain, dsum in summary["domains"].items():
        for strategy, s in dsum["strategies"].items():
            table.add_row(
                domain,
                strategy,
                f"{s['neural_only_accuracy']:.4f}",
                f"{s['neurosymbolic_accuracy']:.4f}",
                f"{s['hallucination_free_ratio']:.2%}",
                f"{s['communication_mb']:.3f}",
            )
    console.print(table)
    console.print(f"\nSummary written to {cfg['paths']['results_dir']}/summary.json")


if __name__ == "__main__":
    main()
