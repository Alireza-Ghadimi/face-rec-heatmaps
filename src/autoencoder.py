"""Simple MLP autoencoder mapping posed landmarks to canonical landmarks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LandmarkAutoencoder(nn.Module):
    def __init__(self, landmark_dim: int = 128, pose_dim: int = 3, hidden: int = 256):
        super().__init__()
        in_dim = landmark_dim + pose_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, landmark_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out
