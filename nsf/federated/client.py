"""Federated client local training. Raw data never leaves the client.

Optional differential-privacy guardrail: before returning its update, a client
can clip the update to a fixed L2 norm and add Gaussian noise. This bounds the
influence of any single client / record on the shared model and is the standard
defence against gradient-inversion and membership-inference attacks on the
transmitted parameters.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nsf.neural.models import get_model_params, set_model_params


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        lr: float = 0.01,
        mu: float = 0.0,
        dp_enabled: bool = False,
        dp_clip_norm: float = 1.0,
        dp_noise_multiplier: float = 0.0,
        dp_generator: Optional[torch.Generator] = None,
    ):
        self.client_id = client_id
        self.model = model
        self.loader = loader
        self.device = device
        self.lr = lr
        self.mu = mu  # FedProx proximal coefficient
        self.n_samples = sum(len(b[0]) for b in loader)
        # Differential privacy settings (applied to the client update)
        self.dp_enabled = dp_enabled
        self.dp_clip_norm = float(dp_clip_norm)
        self.dp_noise_multiplier = float(dp_noise_multiplier)
        self.dp_generator = dp_generator

    def local_train(self, global_params: Dict[str, torch.Tensor], epochs: int) -> Dict[str, torch.Tensor]:
        set_model_params(self.model, global_params)
        self.model.to(self.device)
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        global_tensors = {k: v.to(self.device) for k, v in global_params.items()}

        for _ in range(epochs):
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                if self.mu > 0:
                    prox = 0.0
                    for name, param in self.model.named_parameters():
                        prox = prox + torch.sum((param - global_tensors[name]) ** 2)
                    loss = loss + (self.mu / 2.0) * prox
                loss.backward()
                opt.step()

        updated = get_model_params(self.model)
        if self.dp_enabled:
            updated = self._privatize_update(global_params, updated)
        return updated

    def _privatize_update(
        self,
        global_params: Dict[str, torch.Tensor],
        updated: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Clip the client update to dp_clip_norm (L2) and add Gaussian noise.

        The update is delta = updated - global. We scale the whole delta so its
        global L2 norm is at most dp_clip_norm, then add N(0, (noise_mult*clip)^2)
        noise to each floating-point parameter. Non-float buffers pass through.
        """
        float_keys = [k for k, v in updated.items() if torch.is_floating_point(v)]
        delta = {k: (updated[k] - global_params[k]).float() for k in float_keys}

        total_norm = math.sqrt(sum(float(d.pow(2).sum()) for d in delta.values()))
        clip = self.dp_clip_norm
        scale = min(1.0, clip / (total_norm + 1e-12))

        std = self.dp_noise_multiplier * clip
        private: Dict[str, torch.Tensor] = {}
        for k, v in updated.items():
            if k not in float_keys:
                private[k] = v  # integer/boolean buffers unchanged
                continue
            d = delta[k] * scale
            if std > 0:
                if self.dp_generator is not None:
                    noise = torch.normal(
                        mean=0.0, std=std, size=d.shape, generator=self.dp_generator
                    )
                else:
                    noise = torch.normal(mean=0.0, std=std, size=d.shape)
                d = d + noise
            private[k] = global_params[k] + d
        return private
