"""Federated training orchestration with communication/compute metrics."""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nsf.federated.aggregation import get_strategy
from nsf.federated.client import FederatedClient
from nsf.neural.models import PerceptionMLP, get_model_params, set_model_params
from nsf.utils import accuracy_from_logits


class FederatedTrainer:
    def __init__(
        self,
        domain: str,
        input_dim: int,
        num_classes: int,
        client_loaders: List[DataLoader],
        test_loader: DataLoader,
        strategy: str = "fedavg",
        device: str = "cpu",
        lr: float = 0.01,
        mu: float = 0.01,
        local_epochs: int = 3,
        global_rounds: int = 8,
        dp: Dict[str, Any] | None = None,
        seed: int = 42,
    ):
        self.domain = domain
        self.device = torch.device(device)
        self.strategy_name = strategy
        self.server = get_strategy(strategy)
        self.local_epochs = local_epochs
        self.global_rounds = global_rounds
        self.global_model = PerceptionMLP(input_dim, num_classes).to(self.device)
        self.test_loader = test_loader

        dp = dp or {}
        self.dp_enabled = bool(dp.get("enabled", False))
        self.dp_clip_norm = float(dp.get("clip_norm", 1.0))
        self.dp_noise_multiplier = float(dp.get("noise_multiplier", 0.0))
        # Deterministic DP noise so runs stay reproducible
        dp_generator = torch.Generator().manual_seed(seed) if self.dp_enabled else None

        clients = []
        for i, loader in enumerate(client_loaders):
            local_model = PerceptionMLP(input_dim, num_classes)
            mu_val = mu if strategy == "fedprox" else 0.0
            clients.append(
                FederatedClient(
                    client_id=i,
                    model=local_model,
                    loader=loader,
                    device=self.device,
                    lr=lr,
                    mu=mu_val,
                    dp_enabled=self.dp_enabled,
                    dp_clip_norm=self.dp_clip_norm,
                    dp_noise_multiplier=self.dp_noise_multiplier,
                    dp_generator=dp_generator,
                )
            )
        self.clients = clients
        self.history: List[Dict[str, Any]] = []

    def evaluate(self) -> Dict[str, float]:
        self.global_model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss(reduction="sum")
        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.global_model(xb)
                loss_sum += float(criterion(logits, yb).item())
                correct += int((logits.argmax(1) == yb).sum().item())
                total += len(yb)
        return {
            "accuracy": correct / max(total, 1),
            "loss": loss_sum / max(total, 1),
            "n": total,
        }

    def _param_bytes(self, params: Dict[str, torch.Tensor]) -> int:
        return sum(v.numel() * v.element_size() for v in params.values())

    def train(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        bytes_up = 0
        bytes_down = 0
        local_train_s = 0.0

        for rnd in range(1, self.global_rounds + 1):
            global_params = get_model_params(self.global_model)
            down = self._param_bytes(global_params) * len(self.clients)
            bytes_down += down

            client_params = []
            client_sizes = []
            t_local = time.perf_counter()
            for client in self.clients:
                updated = client.local_train(global_params, self.local_epochs)
                client_params.append(updated)
                client_sizes.append(client.n_samples)
                bytes_up += self._param_bytes(updated)
            local_train_s += time.perf_counter() - t_local

            new_params = self.server.aggregate(client_params, client_sizes, global_params)
            set_model_params(self.global_model, new_params)
            metrics = self.evaluate()
            row = {
                "round": rnd,
                "strategy": self.strategy_name,
                "domain": self.domain,
                **metrics,
            }
            self.history.append(row)
            print(
                f"[{self.domain}|{self.strategy_name}] round {rnd}/{self.global_rounds} "
                f"acc={metrics['accuracy']:.4f} loss={metrics['loss']:.4f}"
            )

        elapsed = time.perf_counter() - t0
        final = self.evaluate()
        return {
            "domain": self.domain,
            "strategy": self.strategy_name,
            "history": self.history,
            "final_accuracy": final["accuracy"],
            "final_loss": final["loss"],
            "wall_time_sec": elapsed,
            "local_train_time_sec": local_train_s,
            "bytes_uploaded": bytes_up,
            "bytes_downloaded": bytes_down,
            "communication_mb": (bytes_up + bytes_down) / (1024 * 1024),
            "differential_privacy": {
                "enabled": self.dp_enabled,
                "clip_norm": self.dp_clip_norm,
                "noise_multiplier": self.dp_noise_multiplier,
            },
            "global_model": copy.deepcopy(self.global_model),
        }
