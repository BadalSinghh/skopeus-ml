"""PyTorch Dataset and DataLoader implementations for paired satellite denoising."""

import csv
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SatelliteDenoisingDataset(Dataset):
    """Memory-efficient PyTorch Dataset for paired satellite audio segments."""

    def __init__(
        self,
        manifest_csv: Path,
        split: Optional[str] = None,
        return_metadata: bool = False,
    ):
        """Initialize Dataset.

        Args:
            manifest_csv: Path to dataset_manifest.csv.
            split: Optional filter ('train', 'validation', 'test').
            return_metadata: If True, __getitem__ returns (noisy, clean, meta).
        """
        self.manifest_csv = Path(manifest_csv)
        self.split = split
        self.return_metadata = return_metadata

        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {self.manifest_csv}")

        with open(self.manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)

        if split is not None:
            self.samples = [r for r in records if r["split"] == split]
        else:
            self.samples = records

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        file_path = rec["file_path"]

        # Load on-demand from disk to keep RAM usage minimal
        with np.load(file_path) as data:
            clean = data["clean"].astype(np.float32)
            noisy = data["noisy"].astype(np.float32)

        # Convert to torch tensor with channel dimension: [1, L]
        clean_tensor = torch.from_numpy(clean).unsqueeze(0)
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0)

        if self.return_metadata:
            meta = {
                "sample_id": rec["sample_id"],
                "split": rec["split"],
                "satellite": rec["satellite"],
                "noise_type": rec["noise_type"],
                "target_snr_db": float(rec.get("target_snr_db", 0.0)),
                "sqi_snr_db": float(rec.get("sqi_snr_db", 0.0)),
            }
            return noisy_tensor, clean_tensor, meta

        return noisy_tensor, clean_tensor


def get_dataloaders(
    manifest_csv: Path,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders."""
    # Ensure pin_memory is only true when CUDA is available
    if not torch.cuda.is_available():
        pin_memory = False

    train_ds = SatelliteDenoisingDataset(manifest_csv, split="train")
    val_ds = SatelliteDenoisingDataset(manifest_csv, split="validation")
    test_ds = SatelliteDenoisingDataset(manifest_csv, split="test", return_metadata=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if len(train_ds) > batch_size else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
