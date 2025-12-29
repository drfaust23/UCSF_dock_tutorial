"""Dataset utilities for point cloud patches."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    """Loads point cloud patches from an .npz file.

    Expected keys:
        points: shape (N, P, 3)
        labels: shape (N,)
    """

    def __init__(self, npz_path: Path):
        """Load points/labels arrays from disk."""
        data = np.load(npz_path)
        self.points = data["points"].astype(np.float32)
        self.labels = data["labels"].astype(np.int64)

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.points)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single (points, label) sample."""
        points = torch.from_numpy(self.points[idx])
        labels = torch.tensor(self.labels[idx])
        return points, labels


class RandomPointCloudDataset(Dataset):
    """Synthetic dataset for quick sanity checks."""

    def __init__(self, num_samples: int, num_points: int, num_classes: int, seed: int = 0):
        """Generate random points/labels with a fixed RNG seed."""
        rng = np.random.default_rng(seed)
        self.points = rng.normal(size=(num_samples, num_points, 3)).astype(np.float32)
        self.labels = rng.integers(0, num_classes, size=(num_samples,), dtype=np.int64)

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.points)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single (points, label) sample."""
        return torch.from_numpy(self.points[idx]), torch.tensor(self.labels[idx])


def load_dataset(
    dataset_path: Optional[str],
    num_samples: int,
    num_points: int,
    num_classes: int,
) -> Dataset:
    """Load real .npz data or fallback to a synthetic dataset."""
    if dataset_path:
        return PointCloudDataset(Path(dataset_path))
    return RandomPointCloudDataset(num_samples, num_points, num_classes)
