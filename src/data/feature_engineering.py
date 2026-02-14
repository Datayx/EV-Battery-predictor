"""
Feature Engineering Pipeline
=============================
Parses NASA PCoE .mat files → extracts cycle-level discharge features →
normalizes → saves processed DataFrames (one per battery + combined).

Usage:
    python -m src.data.feature_engineering
"""
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import joblib

from src.utils.config import Paths, FeatureConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

PATHS = Paths()
FEAT_CFG = FeatureConfig()


# 1. Extract cycle-level features from a single .mat file


def _extract_discharge_features(mat_path: Path) -> pd.DataFrame:
    """
    Parse a NASA .mat file and extract one row per *discharge* cycle with
    aggregated features suitable for time-series modelling.

    Returns a DataFrame with columns:
        battery_id | cycle_index | max_temperature | avg_voltage_load |
        time_to_discharge | discharge_capacity | soh
    """
    battery_id = mat_path.stem  
    mat = loadmat(str(mat_path), simplify_cells=True)
    cycles = mat[battery_id]["cycle"]

    records: list[dict] = []
    discharge_idx = 0

    for cyc in cycles:
        if cyc["type"] != "discharge":
            continue

        data = cyc["data"]

        # --- Guard against empty / corrupt cycles ---
        voltage = np.asarray(data["Voltage_measured"], dtype=np.float64)
        temperature = np.asarray(data["Temperature_measured"], dtype=np.float64)
        time_vec = np.asarray(data["Time"], dtype=np.float64)
        capacity = np.asarray(data["Capacity"], dtype=np.float64)

        if voltage.size == 0 or capacity.size == 0:
            continue

        records.append(
            {
                "battery_id": battery_id,
                "cycle_index": discharge_idx,
                "max_temperature": float(np.nanmax(temperature)),
                "avg_voltage_load": float(np.nanmean(voltage)),
                "time_to_discharge": float(time_vec[-1] - time_vec[0]),
                "discharge_capacity": float(capacity), # final Ah value
            }
        )
        discharge_idx += 1

    df = pd.DataFrame(records)

    # SOH = current_capacity / rated_capacity (clipped to [0, 1])
    df["soh"] = (df["discharge_capacity"] / FEAT_CFG.rated_capacity).clip(0.0, 1.0)

    log.info(f"{battery_id}: extracted {len(df)} discharge cycles")
    return df



# 2. Aggregate all batteries into one clean DataFrame


def build_feature_dataframe(
    battery_ids: list[str] | None = None,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Load and concatenate discharge features for the requested batteries."""
    raw_dir = raw_dir or PATHS.raw_data
    if battery_ids is None:
        battery_ids = ["B0005", "B0006", "B0007", "B0018"]

    dfs = []
    for bid in battery_ids:
        mat_path = raw_dir / f"{bid}.mat"
        if not mat_path.exists():
            log.warning(f"File not found: {mat_path}, skipping.")
            continue
        dfs.append(_extract_discharge_features(mat_path))

    if not dfs:
        raise FileNotFoundError(f"No .mat files found in {raw_dir}")

    return pd.concat(dfs, ignore_index=True)


# 3. Fit scaler on TRAIN batteries only, transform all


def normalize_features(
    df: pd.DataFrame,
    train_batteries: tuple[str, ...],
    scaler_path: Path | None = None,
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Fit a MinMaxScaler on training batteries only (prevents data leakage),
    then transform the entire DataFrame in-place.

    Saves the fitted scaler for inference-time reuse.
    """
    feature_cols = list(FEAT_CFG.feature_cols)
    scaler = MinMaxScaler()

    train_mask = df["battery_id"].isin(train_batteries)
    scaler.fit(df.loc[train_mask, feature_cols])

    df[feature_cols] = scaler.transform(df[feature_cols])

    # Persist scaler
    if scaler_path is None:
        scaler_path = PATHS.processed / "feature_scaler.joblib"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    log.info(f"Scaler saved → {scaler_path}")

    return df, scaler


# 4. Quick sanity-check & summary stats


def print_summary(df: pd.DataFrame) -> None:
    """Log a concise overview of the engineered features."""
    log.info("\n=== Feature Engineering Summary ===")
    for bid, grp in df.groupby("battery_id"):
        log.info(
            f"  {bid}: {len(grp):>4} cycles | "
            f"SOH range [{grp['soh'].min():.3f}, {grp['soh'].max():.3f}]"
        )
    log.info(f"  Total samples: {len(df)}")
    log.info(f"  Features: {list(FEAT_CFG.feature_cols)}")
    log.info(f"  Target: {FEAT_CFG.target_col} (+ derived SOH)\n")



# 5. CLI entry point


def main() -> None:
    from src.utils.config import TrainConfig
    cfg = TrainConfig()

    log.info("Step 1/3 — Extracting cycle-level features from .mat files …")
    df = build_feature_dataframe()

    log.info("Step 2/3 — Normalizing features (fit on train set only) …")
    df, scaler = normalize_features(df, train_batteries=cfg.train_batteries)

    log.info("Step 3/3 — Saving processed data …")
    out_path = PATHS.processed / "battery_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info(f"Saved → {out_path}")

    print_summary(df)


if __name__ == "__main__":
    main()
