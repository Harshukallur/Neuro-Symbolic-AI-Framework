"""Layer 4 — Federated aggregation strategies: FedAvg, FedProx, FedAdagrad."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def weighted_average(
    param_list: List[Dict[str, torch.Tensor]], weights: List[float]
) -> Dict[str, torch.Tensor]:
    total = sum(weights)
    weights = [w / total for w in weights]
    keys = param_list[0].keys()
    avg = {}
    for k in keys:
        avg[k] = sum(w * p[k] for w, p in zip(weights, param_list))
    return avg


class FedAvgServer:
    name = "fedavg"

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
        global_params: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        return weighted_average(client_params, [float(s) for s in client_sizes])


class FedProxServer:
    """Server-side aggregation identical to FedAvg; proximal term is client-side."""

    name = "fedprox"

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
        global_params: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        return weighted_average(client_params, [float(s) for s in client_sizes])


class FedAdagradServer:
    """Adaptive federated optimization (FedAdagrad-style server update)."""

    name = "fedadagrad"

    def __init__(self, eta: float = 0.01, tau: float = 1e-2):
        self.eta = eta
        self.tau = tau
        self.v: Dict[str, torch.Tensor] | None = None

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
        global_params: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if global_params is None:
            raise ValueError("FedAdagrad requires global_params")
        avg_client = weighted_average(client_params, [float(s) for s in client_sizes])
        # Pseudo-gradient = global - avg_client (clients moved away from global)
        delta = {k: global_params[k] - avg_client[k] for k in global_params}
        if self.v is None:
            self.v = {k: torch.zeros_like(t) for k, t in delta.items()}
        updated = {}
        for k in global_params:
            self.v[k] = self.v[k] + delta[k] ** 2
            updated[k] = global_params[k] - self.eta * delta[k] / (torch.sqrt(self.v[k]) + self.tau)
        return updated


def get_strategy(name: str):
    name = name.lower()
    if name == "fedavg":
        return FedAvgServer()
    if name == "fedprox":
        return FedProxServer()
    if name == "fedadagrad":
        return FedAdagradServer()
    raise ValueError(f"Unknown strategy: {name}")
