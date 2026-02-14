"""
Visualization: Degradation Curves
==================================
Plots actual vs. predicted SOH for the test battery, plus
training history if available.

Usage:
    python -m src.models.visualize
"""
from __future__ import annotations
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.utils.config import Paths, TrainConfig
from src.models.architecture import LSTMRegressor
from src.models.dataset import BatteryDataset

PATHS = Paths()
CFG = TrainConfig()


@torch.no_grad()
def collect_predictions(
    model: LSTMRegressor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather all (actual, predicted) pairs from a DataLoader."""
    model.eval()
    actuals, preds = [], []
    for X_batch, y_batch in loader:
        out = model(X_batch.to(device))
        actuals.append(y_batch.numpy())
        preds.append(out.cpu().numpy())
    return np.concatenate(actuals), np.concatenate(preds)


def plot_degradation_curve(save_path: Path | None = None) -> None:
    """Load best checkpoint, run on test battery, plot results."""
    device = torch.device("cpu")
    ckpt = PATHS.checkpoints / "best_model.pth"

    model = LSTMRegressor()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)

    df = pd.read_parquet(PATHS.processed / "battery_features.parquet")
    test_df = df[df["battery_id"].isin(CFG.test_batteries)]
    test_ds = BatteryDataset(test_df, seq_length=CFG.seq_length)
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False)

    actual, predicted = collect_predictions(model, test_loader, device)

    # Plot 
    fig, ax = plt.subplots(figsize=(10, 5))
    cycles = np.arange(CFG.seq_length, CFG.seq_length + len(actual))

    ax.plot(cycles, actual, label="Actual SOH", linewidth=2, color="#2563eb")
    ax.plot(cycles, predicted, label="Predicted SOH", linewidth=2, linestyle="--", color="#dc2626")

    ax.axhline(y=0.7, color="gray", linestyle=":", alpha=0.7, label="EOL Threshold (70%)")

    ax.set_xlabel("Discharge Cycle", fontsize=12)
    ax.set_ylabel("State of Health (SOH)", fontsize=12)
    ax.set_title(f"Battery Degradation — Test Battery ({', '.join(CFG.test_batteries)})", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = save_path or Path("outputs/degradation_curve.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot → {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_degradation_curve()
