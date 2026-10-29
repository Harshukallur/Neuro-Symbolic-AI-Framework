"""Plot federated accuracy curves and domain comparison charts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsf.utils import load_config


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    results_dir = ROOT / cfg["paths"]["results_dir"]
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for domain in ("healthcare", "banking"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for strategy in cfg["federated_strategies"]:
            path = results_dir / f"{domain}_{strategy}_results.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            hist = data["federated"]["history"]
            rounds = [h["round"] for h in hist]
            accs = [h["accuracy"] for h in hist]
            ax.plot(rounds, accs, marker="o", label=strategy)
        ax.set_title(f"{domain.title()} — Federated Test Accuracy")
        ax.set_xlabel("Global Round")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{domain}_accuracy_curves.png", dpi=140)
        plt.close(fig)

    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = []
        neural = []
        neuro = []
        for domain, dsum in summary["domains"].items():
            best = dsum["best_strategy"]
            s = dsum["strategies"][best]
            labels.append(f"{domain}\n({best})")
            neural.append(s["neural_only_accuracy"])
            neuro.append(s["neurosymbolic_accuracy"])
        x = range(len(labels))
        ax.bar([i - 0.18 for i in x], neural, width=0.36, label="Neural only")
        ax.bar([i + 0.18 for i in x], neuro, width=0.36, label="Neuro-Symbolic")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title("Best Strategy: Neural vs Neuro-Symbolic Accuracy")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "neural_vs_neurosymbolic.png", dpi=140)
        plt.close(fig)

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
