"""
PyTorch Dataset for Battery Degradation Sequences
===================================================
Creates sliding-window samples:  X = cycles [t-N … t],  y = SOH at cycle t+1.

Windows are constructed *per battery* so sequences never cross battery boundaries.
"""
from __future__ import annotations
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.config import FeatureConfig

FEAT_CFG = FeatureConfig()


class BatteryDataset(Dataset):
    """
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: battery_id, cycle_index, feature_cols…, soh.
    seq_length : int
        Number of past cycles in each input window.
    noise_std : float
        Standard deviation of Gaussian noise added to input features
        during training (data augmentation). Set to 0.0 to disable.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_length: int = 50,
        noise_std: float = 0.0,
    ) -> None:
        self.seq_length = seq_length
        self.noise_std = noise_std
        self.feature_cols = list(FEAT_CFG.feature_cols)
        self.samples: list[tuple[np.ndarray, float]] = []

        # Build windows per battery (never mix batteries in a single window)
        for _bid, grp in df.sort_values("cycle_index").groupby("battery_id"):
            features = grp[self.feature_cols].values.astype(np.float32)
            targets = grp["soh"].values.astype(np.float32)

            for i in range(seq_length, len(grp)):
                x = features[i - seq_length : i]   
                y = targets[i]                       
                self.samples.append((x, y))
                

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        x = torch.from_numpy(x)

        # Gaussian noise augmentation (only effective when noise_std > 0)
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        return x, torch.tensor(y, dtype=torch.float32)
