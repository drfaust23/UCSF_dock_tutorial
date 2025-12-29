"""VQ-VAE model for point cloud patch reconstruction and classification."""

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from .pointnet import PointNetDecoder, PointNetEncoder
from .quantizer import VectorQuantizer


@dataclass
class ModelOutput:
    """Structured outputs produced by the VQ-VAE model."""

    reconstructed: torch.Tensor  # reconstructed points
    commitment_loss: torch.Tensor  # VQ loss term
    classification_logits: torch.Tensor  # class logits for labels
    codes: torch.Tensor  # discrete code indices
    perplexity: torch.Tensor  # code usage perplexity


class VQVAEPointCloud(nn.Module):
    """End-to-end VQ-VAE for point cloud reconstruction and classification."""
    def __init__(
        self,
        num_points: int,
        latent_dim: int,
        num_codes: int,
        num_classes: int,
        commitment_cost: float = 0.25,
    ):
        """Construct encoder, quantizer, decoder, and classifier head."""
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim)
        self.quantizer = VectorQuantizer(num_codes, latent_dim, commitment_cost)
        self.decoder = PointNetDecoder(latent_dim, num_points)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, points: torch.Tensor) -> ModelOutput:
        """Forward pass returning reconstructed points and auxiliary outputs."""
        latents = self.encoder(points)
        quantized_out = self.quantizer(latents)
        reconstructed = self.decoder(quantized_out.quantized)
        logits = self.classifier(quantized_out.quantized)
        return ModelOutput(
            reconstructed=reconstructed,
            commitment_loss=quantized_out.loss,
            classification_logits=logits,
            codes=quantized_out.codes,
            perplexity=quantized_out.perplexity,
        )

    def reconstruction_loss(self, points: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Compute MSE reconstruction loss for point coordinates."""
        return F.mse_loss(reconstructed, points)

    def classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy classification loss."""
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def encode_codes(self, points: torch.Tensor) -> torch.Tensor:
        """Return code indices for retrieval or indexing."""
        latents = self.encoder(points)
        return self.quantizer(latents).codes

    @torch.no_grad()
    def generate_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Generate point clouds from discrete codes."""
        if codes.dim() != 1:
            raise ValueError("Codes should have shape (B,) for generation.")
        quantized = self.quantizer.codebook(codes)
        return self.decoder(quantized)
