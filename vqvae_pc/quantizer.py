"""Vector quantization module for VQ-VAE."""

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class QuantizerOutput:
    """Container for vector-quantization outputs."""

    quantized: torch.Tensor  # straight-through quantized latents
    loss: torch.Tensor  # combined codebook + commitment loss
    codes: torch.Tensor  # codebook indices
    perplexity: torch.Tensor  # code usage perplexity


class VectorQuantizer(nn.Module):
    """VQ layer that maps continuous latents to nearest codebook entry."""

    def __init__(self, num_codes: int, code_dim: int, commitment_cost: float = 0.25):
        """Initialize the codebook and commitment weight."""
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, inputs: torch.Tensor) -> QuantizerOutput:
        """Quantize inputs of shape (B, D) and compute VQ losses."""
        if inputs.dim() != 2:
            raise ValueError("VectorQuantizer expects (B, D) inputs.")

        flat_inputs = inputs
        distances = (
            flat_inputs.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1)
        )

        codes = torch.argmin(distances, dim=1)
        quantized = self.codebook(codes)

        codebook_loss = F.mse_loss(quantized, flat_inputs.detach())
        commitment_loss = F.mse_loss(quantized.detach(), flat_inputs)
        loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized = flat_inputs + (quantized - flat_inputs).detach()
        avg_probs = torch.mean(F.one_hot(codes, self.num_codes).float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return QuantizerOutput(
            quantized=quantized,
            loss=loss,
            codes=codes,
            perplexity=perplexity,
        )
