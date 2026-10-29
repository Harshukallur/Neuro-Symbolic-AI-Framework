"""End-to-end Domain-Agnostic Federated Neuro-Symbolic pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from nsf.bridge.neural_symbolic import NeuralSymbolicBridge
from nsf.data.generators import generate_all_datasets, load_domain_data
from nsf.federated.trainer import FederatedTrainer
from nsf.llm.explainer import VerifiedLLMExplainer
from nsf.symbolic.engine import SymbolicReasoner
from nsf.utils import classification_report_dict, ensure_dirs, save_json, set_seed


class NeuroSymbolicFramework:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        set_seed(cfg["seed"])
        ensure_dirs(cfg["paths"]["data_dir"], cfg["paths"]["results_dir"], cfg["paths"]["models_dir"])
        self.explainer = VerifiedLLMExplainer()

    def prepare_data(self) -> Dict[str, Any]:
        return generate_all_datasets(self.cfg, self.cfg["paths"]["data_dir"])

    def train_domain(self, domain: str, strategy: str) -> Dict[str, Any]:
        data = load_domain_data(domain, self.cfg["paths"]["data_dir"], self.cfg["batch_size"])
        dcfg = self.cfg["domains"][domain]
        trainer = FederatedTrainer(
            domain=domain,
            input_dim=dcfg["feature_dim"],
            num_classes=dcfg["num_classes"],
            client_loaders=[c["loader"] for c in data["clients"]],
            test_loader=data["test_loader"],
            strategy=strategy,
            device=self.cfg.get("device", "cpu"),
            lr=self.cfg["learning_rate"],
            mu=self.cfg["fedprox_mu"],
            local_epochs=self.cfg["local_epochs"],
            global_rounds=self.cfg["global_rounds"],
            dp=self.cfg.get("differential_privacy"),
            seed=self.cfg["seed"],
        )
        result = trainer.train()
        model_path = Path(self.cfg["paths"]["models_dir"]) / f"{domain}_{strategy}.pt"
        torch.save(result["global_model"].state_dict(), model_path)
        result["model_path"] = str(model_path)
        # detach model object for JSON
        model = result.pop("global_model")
        result["model"] = model
        return result, data

    def infer_and_verify(
        self,
        domain: str,
        model: torch.nn.Module,
        data: Dict[str, Any],
        max_examples: int = 50,
    ) -> Dict[str, Any]:
        dcfg = self.cfg["domains"][domain]
        id_key = "patient_id" if domain == "healthcare" else "txn_id"
        bridge = NeuralSymbolicBridge(domain, dcfg["class_names"], id_key)
        reasoner = SymbolicReasoner(domain, dcfg["class_names"])

        model.eval()
        device = next(model.parameters()).device

        # Build standardized feature tensor matching training stats
        features = data["features"]
        test_df = data["test_df"]
        x = torch.tensor(test_df[features].values, dtype=torch.float32)
        x = (x - data["feat_mean"]) / data["feat_std"]

        y_true = test_df["label"].tolist()
        neural_preds: List[int] = []
        final_preds: List[int] = []
        traces = []
        overrides = 0
        conflicts = 0
        verified_count = 0
        guard_pass = 0

        with torch.no_grad():
            probs_all = model.predict_proba(x.to(device)).cpu().numpy()

        for i in range(len(test_df)):
            probs = probs_all[i].tolist()
            npred = int(np.argmax(probs))
            neural_preds.append(npred)
            raw = dict(data["test_facts"][i])
            fact = bridge.convert(raw, npred, probs, entity_id=raw.get(id_key))
            v = reasoner.verify(fact)
            final_preds.append(v.final_prediction)
            if v.override_applied:
                overrides += 1
            if v.conflict:
                conflicts += 1
            if v.verified:
                verified_count += 1
            if i < max_examples:
                trace = self.explainer.explain(v)
                if trace.hallucination_guard.get("passed"):
                    guard_pass += 1
                traces.append(trace.to_dict())

        neural_metrics = classification_report_dict(y_true, neural_preds, dcfg["class_names"])
        final_metrics = classification_report_dict(y_true, final_preds, dcfg["class_names"])

        # Hallucination rate among exported traces
        n_traces = max(len(traces), 1)
        return {
            "domain": domain,
            "n_test": len(test_df),
            "neural_metrics": neural_metrics,
            "neurosymbolic_metrics": final_metrics,
            "conflicts": conflicts,
            "overrides": overrides,
            "verified_ratio": verified_count / len(test_df),
            "explanation_hallucination_free_ratio": guard_pass / n_traces,
            "sample_traces": traces,
        }

    def run_all(self, regenerate_data: bool = True) -> Dict[str, Any]:
        data_dir = Path(self.cfg["paths"]["data_dir"])
        if regenerate_data or not (data_dir / "generation_report.json").exists():
            print("=== Generating custom datasets ===")
            data_report = self.prepare_data()
        else:
            print("=== Using existing datasets ===")
            data_report = json.loads((data_dir / "generation_report.json").read_text(encoding="utf-8"))
        summary: Dict[str, Any] = {"datasets": data_report, "domains": {}}

        for domain in ("healthcare", "banking"):
            domain_summary: Dict[str, Any] = {"strategies": {}}
            best_acc = -1.0
            best_bundle = None
            data = None

            for strategy in self.cfg["federated_strategies"]:
                print(f"\n=== Training {domain} with {strategy} ===")
                train_result, data = self.train_domain(domain, strategy)
                model = train_result.pop("model")
                serializable = {k: v for k, v in train_result.items() if k != "history"}
                serializable["history"] = train_result["history"]
                eval_result = self.infer_and_verify(domain, model, data)
                bundle = {
                    "federated": serializable,
                    "evaluation": {
                        k: v for k, v in eval_result.items() if k != "sample_traces"
                    },
                }
                # Save traces separately (can be large)
                save_json(
                    eval_result["sample_traces"],
                    Path(self.cfg["paths"]["results_dir"]) / f"{domain}_{strategy}_traces.json",
                )
                save_json(
                    bundle,
                    Path(self.cfg["paths"]["results_dir"]) / f"{domain}_{strategy}_results.json",
                )
                domain_summary["strategies"][strategy] = {
                    "final_accuracy_federated": train_result["final_accuracy"],
                    "neurosymbolic_accuracy": eval_result["neurosymbolic_metrics"]["accuracy"],
                    "neural_only_accuracy": eval_result["neural_metrics"]["accuracy"],
                    "communication_mb": train_result["communication_mb"],
                    "wall_time_sec": train_result["wall_time_sec"],
                    "conflicts": eval_result["conflicts"],
                    "overrides": eval_result["overrides"],
                    "hallucination_free_ratio": eval_result["explanation_hallucination_free_ratio"],
                }
                ns_acc = eval_result["neurosymbolic_metrics"]["accuracy"]
                if ns_acc > best_acc:
                    best_acc = ns_acc
                    best_bundle = {
                        "strategy": strategy,
                        "accuracy": ns_acc,
                        "model": model,
                        "eval": eval_result,
                    }

            domain_summary["best_strategy"] = best_bundle["strategy"] if best_bundle else None
            domain_summary["best_neurosymbolic_accuracy"] = best_acc
            summary["domains"][domain] = domain_summary

        save_json(summary, Path(self.cfg["paths"]["results_dir"]) / "summary.json")
        return summary
