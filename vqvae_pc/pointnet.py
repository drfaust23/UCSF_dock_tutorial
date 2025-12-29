"""PointNet-style encoder for 3D point clouds."""

import torch
from torch import nn


class PointNetEncoder(nn.Module):
    """PointNet-style encoder that pools point features into a latent vector."""

    def __init__(self, latent_dim: int):
        """Build MLP layers that lift (x,y,z) points into latent features."""
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, latent_dim, 1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Encode points of shape (B, N, 3) into latent (B, D)."""
        if points.dim() != 3 or points.size(-1) != 3:
            raise ValueError("Expected points with shape (B, N, 3).")
        x = points.transpose(1, 2)
        x = self.mlp(x)
        x = torch.max(x, dim=2).values
        return x


class PointNetDecoder(nn.Module):
    """MLP decoder that expands a latent vector back to point coordinates."""

    def __init__(self, latent_dim: int, num_points: int):
        """Initialize decoder MLP and target point count."""
        super().__init__()
        self.num_points = num_points
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_points * 3),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent (B, D) into reconstructed points (B, N, 3)."""
        x = self.mlp(latents)
        x = x.view(latents.size(0), self.num_points, 3)
        return x
