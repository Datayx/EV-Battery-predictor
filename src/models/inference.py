"""
Inference Utility
=================
Load a trained checkpoint + fitted scaler → predict SOH from recent cycle history.
Designed to be called by the FastAPI service or used standalone.

Usage:
    from src.models.inference import predict_soh
    soh = predict_soh(recent_cycles_df)
"""



from __future__ import annotations
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.utils.config import Paths, TrainConfig, FeatureConfig
from src.models.architecture import LSTMRegressor

PATHS = Paths()
CFG = TrainConfig()
FEAT_CFG = FeatureConfig()


def load_model(
    checkpoint: Path | None = None,
    device: torch.device | None = None,
) -> tuple[LSTMRegressor, torch.device]:
    """Load trained LSTM from disk."""
    device = device or torch.device("cpu")
    checkpoint = checkpoint or PATHS.checkpoints / "best_model.pth"

    model = LSTMRegressor()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device).eval()
    return model, device


@torch.no_grad()
def predict_soh(
    recent_cycles: pd.DataFrame,
    model: LSTMRegressor | None = None,
    scaler_path: Path | None = None,
    scaler: object | None = None,
) -> float:
    """
    Predict SOH for the *next* cycle given the most recent `seq_length` cycles.

    Parameters
    ----------
    recent_cycles : DataFrame with columns matching FeatureConfig.feature_cols.
                    Must have at least `seq_length` rows.
    model : Pre-loaded model (optional; loads from checkpoint if None).
    scaler_path : Path to the fitted MinMaxScaler.
    scaler : Pre-fitted scaler object (optional; if provided, scaler_path is ignored).

    Returns
    -------
    Predicted SOH as a float in [0, 1].
    """
    feature_cols = list(FEAT_CFG.feature_cols)

    if len(recent_cycles) < CFG.seq_length:
        raise ValueError(
            f"Need at least {CFG.seq_length} cycles, got {len(recent_cycles)}"
        )

    # Load model if not provided
    if model is None:
        model, device = load_model()
    else:
        device = next(model.parameters()).device

    # Scale features using the same scaler from training
    if scaler is None:
        scaler_path = scaler_path or PATHS.processed / "feature_scaler.joblib"
        scaler = joblib.load(scaler_path)

    features = recent_cycles[feature_cols].tail(CFG.seq_length).copy()
    features[feature_cols] = scaler.transform(features[feature_cols])

    # Shape: (1, seq_length, n_features)
    x = torch.tensor(features.values, dtype=torch.float32).unsqueeze(0).to(device)

    prediction = model(x).item()
    return float(np.clip(prediction, 0.0, 1.0))
