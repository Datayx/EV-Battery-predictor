"""
LSTM Regressor for Battery SOH Prediction
==========================================
A 2-layer LSTM that reads a window of past cycle features
and outputs a single scalar: the predicted SOH for the next cycle.
"""
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)


import torch
import torch.nn as nn
import torch
import torch.nn as nn

from src.utils.config import TrainConfig

CFG = TrainConfig()


class LSTMRegressor(nn.Module):
    """
    Architecture
    ------------
    Input  → (batch, seq_len, input_size)
    LSTM-1 → 64 hidden units, dropout 0.3
    LSTM-2 → 64 hidden units, dropout 0.3
    Dense  → 32 units, ReLU
    FC     → 1 (predicted SOH)
    """

    def __init__(
        self,
        input_size: int = CFG.input_size,
        hidden_1: int = CFG.hidden_size_1,
        hidden_2: int = CFG.hidden_size_2,
        dropout: float = CFG.dropout,
    ) -> None:
        super().__init__()

        self.lstm_1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_1,
            batch_first=True,
            dropout=0.0,       # dropout only between stacked layers
        )
        self.dropout_1 = nn.Dropout(dropout)

        self.lstm_2 = nn.LSTM(
            input_size=hidden_1,
            hidden_size=hidden_2,
            batch_first=True,
        )
        self.dropout_2 = nn.Dropout(dropout)

        # Extra dense layer to learn non-linear mapping before output
        self.dense = nn.Linear(hidden_2, 32)
        self.relu = nn.ReLU()
        self.dropout_3 = nn.Dropout(dropout)

        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, input_size)

        Returns
        -------
        Tensor of shape (batch,) — predicted SOH values.
        """
        out, _ = self.lstm_1(x)            
        out = self.dropout_1(out)
        out, _ = self.lstm_2(out)         

       
        out = out[:, -1, :]             
        out = self.dropout_2(out)

        # Dense head for better capacity mapping
        out = self.relu(self.dense(out)) 
        out = self.dropout_3(out)
        out = self.fc(out).squeeze(-1)     
        return out
