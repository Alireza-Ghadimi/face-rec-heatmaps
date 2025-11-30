"""Simple MLP autoencoder mapping posed landmarks to canonical landmarks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LandmarkAutoencoder(nn.Module):
    def __init__(self, landmark_dim: int = 128, pose_dim: int = 3, hidden: int | None = None, noise_std: float = 0.01):
        super().__init__()
        in_dim = landmark_dim + pose_dim
        hidden_dim = hidden or in_dim  # no bottleneck by default
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, landmark_dim)
        self._init_identity_like(self.encoder, noise_std)
        self._init_identity_like(self.decoder, noise_std)

    def _init_identity_like(self, layer: nn.Linear, noise_std: float) -> None:
        with torch.no_grad():
            weight = layer.weight
            weight.zero_()
            rows, cols = weight.shape
            diag = min(rows, cols)
            for i in range(diag):
                weight[i, i] = 1.0
            weight.add_(noise_std * torch.randn_like(weight))
            if layer.bias is not None:
                layer.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.encoder(x))
        out = self.decoder(z)
        return out
