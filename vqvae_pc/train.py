"""Training script for VQ-VAE point cloud model."""

import argparse
from dataclasses import asdict
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import load_dataset
from .models import VQVAEPointCloud


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train VQ-VAE for point cloud patches.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to .npz dataset file.")
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--num-points", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-codes", type=int, default=512)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_epoch(
    model: VQVAEPointCloud,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch and return aggregated metrics."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_cls = 0.0
    total_commit = 0.0
    total_perplexity = 0.0

    for points, labels in tqdm(loader, desc="Training", leave=False):
        points = points.to(device)
        labels = labels.to(device)

        output = model(points)
        recon_loss = model.reconstruction_loss(points, output.reconstructed)
        cls_loss = model.classification_loss(output.classification_logits, labels)
        loss = recon_loss + cls_loss + output.commitment_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_cls += cls_loss.item()
        total_commit += output.commitment_loss.item()
        total_perplexity += output.perplexity.item()

    num_batches = len(loader)
    return {
        "loss": total_loss / num_batches,
        "recon": total_recon / num_batches,
        "cls": total_cls / num_batches,
        "commit": total_commit / num_batches,
        "perplexity": total_perplexity / num_batches,
    }


def main() -> None:
    """Entry point for CLI training."""
    args = parse_args()
    device = torch.device(args.device)

    dataset = load_dataset(args.dataset, args.num_samples, args.num_points, args.num_classes)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = VQVAEPointCloud(
        num_points=args.num_points,
        latent_dim=args.latent_dim,
        num_codes=args.num_codes,
        num_classes=args.num_classes,
        commitment_cost=args.commitment_cost,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Training configuration:")
    for key, value in asdict(args).items() if hasattr(args, "__dataclass_fields__") else vars(args).items():
        print(f"  {key}: {value}")

    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(model, loader, optimizer, device)
        print(
            f"Epoch {epoch:03d} | loss={metrics['loss']:.4f} "
            f"recon={metrics['recon']:.4f} cls={metrics['cls']:.4f} "
            f"commit={metrics['commit']:.4f} perp={metrics['perplexity']:.2f}"
        )

    torch.save(model.state_dict(), "vqvae_pointcloud.pt")
    print("Saved model to vqvae_pointcloud.pt")


if __name__ == "__main__":
    main()
