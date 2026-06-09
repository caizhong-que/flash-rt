"""LSTM-based anomaly detector (baseline)."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class LSTMConfig:
    input_dim: int = 3
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    seq_len: int = 50
    pred_horizon: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 50
    detection_mode: str = "prediction"
    threshold_percentile: float = 95.0


class LSTMEncoder(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(config.input_dim, config.hidden_dim, config.num_layers,
                            batch_first=True, dropout=config.dropout if config.num_layers > 1 else 0)
        if config.detection_mode == "prediction":
            self.output_layer = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(), nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.input_dim * config.pred_horizon),
            )
        else:
            self.output_layer = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(), nn.Linear(config.hidden_dim, config.input_dim),
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, (hidden, _) = self.lstm(x)
        if self.config.detection_mode == "prediction":
            out = self.output_layer(lstm_out[:, -1, :])
            out = out.view(x.size(0), self.config.pred_horizon, self.config.input_dim)
        else:
            out = self.output_layer(lstm_out)
        return out, hidden[-1]


class LSTMAnomalyDetector:
    def __init__(self, config: LSTMConfig = None, device: str = None):
        self.config = config or LSTMConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.threshold = None

    def fit(self, normal_data: np.ndarray):
        if normal_data.ndim == 1:
            normal_data = normal_data.reshape(-1, 1)
        n, nf = normal_data.shape
        assert nf == self.config.input_dim, f"Expected {self.config.input_dim} features, got {nf}"

        X, Y = self._create_sequences(normal_data)
        loader = DataLoader(TensorDataset(
            torch.FloatTensor(X).to(self.device),
            torch.FloatTensor(Y).to(self.device),
        ), batch_size=self.config.batch_size, shuffle=True)

        self.model = LSTMEncoder(self.config).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(self.config.num_epochs):
            total_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                out, _ = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            Y_t = torch.FloatTensor(Y).to(self.device)
            out, _ = self.model(X_t)
            errors = (out - Y_t).abs().mean(dim=(1, 2)).cpu().numpy()
            self.threshold = np.percentile(errors, self.config.threshold_percentile)
        return self

    def detect(self, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        if self.model is None:
            raise ValueError("Call fit() first")
        if test_data.ndim == 1:
            test_data = test_data.reshape(-1, 1)
        X, Y_true = self._create_sequences(test_data)
        X_t = torch.FloatTensor(X).to(self.device)
        Y_t = torch.FloatTensor(Y_true).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out, _ = self.model(X_t)
            errors = (out - Y_t).abs().mean(dim=(1, 2)).cpu().numpy()
        flags = (errors > self.threshold).astype(int)
        full_scores = np.zeros(len(test_data))
        full_scores[self.config.seq_len:self.config.seq_len + len(errors)] = errors
        return flags, full_scores, {"threshold": self.threshold, "raw_errors": errors}

    def _create_sequences(self, data):
        s, h = self.config.seq_len, self.config.pred_horizon
        X, Y = [], []
        for i in range(len(data) - s - h + 1):
            X.append(data[i:i + s])
            Y.append(data[i + s:i + s + h] if self.config.detection_mode == "prediction" else data[i:i + s])
        return np.array(X), np.array(Y)
