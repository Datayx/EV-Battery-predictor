"""
Training Pipeline
=================
End-to-end: load features → build DataLoaders → train LSTM → evaluate on
held-out battery → save best checkpoint.

Usage:
    python -m src.models.train
"""
from __future__ import annotations
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import Paths, TrainConfig
from src.models.architecture import LSTMRegressor
from src.models.dataset import BatteryDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PATHS = Paths()
CFG = TrainConfig()


# Helpers


def set_seed(seed: int) -> None:
    """Ensure reproducible results across runs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



# Data Loading


def _build_loaders(df: pd.DataFrame) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split by battery ID → wrap in DataLoaders."""
    train_df = df[df["battery_id"].isin(CFG.train_batteries)]
    val_df = df[df["battery_id"].isin(CFG.val_batteries)]
    test_df = df[df["battery_id"].isin(CFG.test_batteries)]

    log.info(
        f"Split sizes  →  train: {len(train_df)} cycles | "
        f"val: {len(val_df)} cycles | test: {len(test_df)} cycles"
    )

    train_ds = BatteryDataset(train_df, seq_length=CFG.seq_length, noise_std=0.02)
    val_ds = BatteryDataset(val_df, seq_length=CFG.seq_length)   # no noise for eval
    test_ds = BatteryDataset(test_df, seq_length=CFG.seq_length)  # no noise for eval

    log.info(
        f"Window samples →  train: {len(train_ds)} | "
        f"val: {len(val_ds)} | test: {len(test_ds)}"
    )

    common = dict(num_workers=0, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader


# Training & Evaluation


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one epoch of training; return mean loss."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate model; return (MSE, RMSE, MAE)."""
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch)

        total_mse += criterion(preds, y_batch).item() * X_batch.size(0)
        total_mae += torch.abs(preds - y_batch).sum().item()
        n += X_batch.size(0)

    mse = total_mse / n
    return mse, mse ** 0.5, total_mae / n



# Early Stopping


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int, checkpoint_path: Path) -> None:
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.checkpoint_path)
            log.info(f"  ✓ New best model saved (val_loss={val_loss:.6f})")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True



# Main Training Loop


def train(data_path: Path | None = None) -> None:
    set_seed(CFG.seed)
    device = get_device()
    log.info(f"Device: {device}")

    # Load processed features
    data_path = data_path or PATHS.processed / "battery_features.parquet"
    df = pd.read_parquet(data_path)
    log.info(f"Loaded {len(df)} rows from {data_path}")

    train_loader, val_loader, test_loader = _build_loaders(df)

    # Model, Loss, Optimizer 
    model = LSTMRegressor(
        input_size=CFG.input_size, 
        hidden_1=CFG.hidden_size_1, 
        hidden_2=CFG.hidden_size_2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
    )

    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"Training for up to {CFG.epochs} epochs (patience={CFG.patience})\n")

    ckpt_path = PATHS.checkpoints / "best_model.pth"
    stopper = EarlyStopping(patience=CFG.patience, checkpoint_path=ckpt_path)

    # Training 
    t0 = time.perf_counter()

    for epoch in range(1, CFG.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_mse, val_rmse, val_mae = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_mse)

        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            log.info(
                f"Epoch {epoch:>3}/{CFG.epochs} | "
                f"train_loss: {train_loss:.6f} | "
                f"val_rmse: {val_rmse:.6f} | val_mae: {val_mae:.6f} | "
                f"lr: {lr:.2e}"
            )

        stopper.step(val_mse, model)
        if stopper.should_stop:
            log.info(f"Early stopping at epoch {epoch}")
            break

    elapsed = time.perf_counter() - t0
    log.info(f"\nTraining complete in {elapsed:.1f}s")

    #  Final Evaluation on Test Battery (B0007)
    log.info("\n=== Test Set Evaluation (unseen battery) ===")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_mse, test_rmse, test_mae = evaluate(model, test_loader, criterion, device)
    log.info(f"  MSE:  {test_mse:.6f}")
    log.info(f"  RMSE: {test_rmse:.6f}")
    log.info(f"  MAE:  {test_mae:.6f}")
    log.info(f"\nBest checkpoint → {ckpt_path}")


if __name__ == "__main__":
    train()