# src/utils/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    raw_data: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    checkpoints: Path = Path("checkpoints")

@dataclass(frozen=True)
class FeatureConfig:
    """Cycle-level features extracted from raw discharge data."""
    feature_cols: tuple = (
        "max_temperature",
        "avg_voltage",
        "time_to_discharge",
        "cycle_index",    
    )
    target_col: str = "discharge_capacity"
    rated_capacity: float = 2.0

@dataclass(frozen=True)
class TrainConfig:
    train_batteries: tuple = ("B0005", "B0006", "B0007")
    val_batteries: tuple = ("B0018",)
    test_batteries: tuple = ("B0018",)
    
    seq_length: int = 50
    input_size: int = 4       
    hidden_size_1: int = 64
    hidden_size_2: int = 32
    num_layers: int = 2
    dropout: float = 0.3
    
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    seed: int = 42