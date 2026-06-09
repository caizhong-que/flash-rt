"""Transformer trainer with sliding windows and progressive loss."""

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from flash_rt.models.transformer import RTDetector, create_sliding_windows


class TransformerTrainer:
    def __init__(self, model: RTDetector, device: torch.device,
                 window_size: int = 10, batch_size: int = 128,
                 lr: float = 1e-5, weight_decay: float = 0.0,
                 scheduler_step: int = 5, scheduler_gamma: float = 0.9,
                 n_epochs: int = 15):
        self.model = model.to(device)
        self.device = device
        self.window_size = window_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        self.criterion = nn.MSELoss(reduction="none")
        self.history = []

    def fit(self, triple: torch.Tensor, save_dir: Optional[Path] = None) -> List[float]:
        windows = create_sliding_windows(triple.float(), self.window_size)
        loader = DataLoader(TensorDataset(windows, windows), self.batch_size, shuffle=True,
                            num_workers=2, pin_memory=True)
        for epoch in range(self.n_epochs):
            self.model.train()
            losses = []
            for batch_x, _ in loader:
                batch_x = batch_x.permute(1, 0, 2).to(self.device).float()
                target = batch_x[-1].unsqueeze(0)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x, target)
                if isinstance(outputs, tuple):
                    w1 = 1.0 / (epoch + 1)
                    loss = w1 * self.criterion(outputs[0], target).mean() + (1 - w1) * self.criterion(outputs[1], target).mean()
                else:
                    loss = self.criterion(outputs, target).mean()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            self.scheduler.step()
            self.history.append({"epoch": epoch, "loss": sum(losses) / len(losses)})
            if save_dir and (epoch + 1) % 5 == 0:
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict()},
                           save_dir / f"transformer_epoch{epoch}.pth")
        if save_dir:
            torch.save({"epoch": self.n_epochs - 1, "model_state_dict": self.model.state_dict()},
                       save_dir / "transformer_final.pth")
        return [h["loss"] for h in self.history]


def train_transformer(triple: torch.Tensor, n_feats: int = 3,
                      window_size: int = 10, device: Optional[torch.device] = None,
                      save_dir: Optional[str] = None) -> RTDetector:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RTDetector(n_feats=n_feats, n_window=window_size)
    trainer = TransformerTrainer(model, device, window_size=window_size)
    trainer.fit(triple, Path(save_dir) if save_dir else None)
    return model
