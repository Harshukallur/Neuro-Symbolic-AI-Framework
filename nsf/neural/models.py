"""Layer 1 — Neural Perception models."""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn


class PerceptionMLP(nn.Module):
    """Compact MLP for tabular domain perception / classification."""

    def __init__(self, input_dim: int, num_classes: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


def get_model_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_model_params(model: nn.Module, params: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(params)


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)
