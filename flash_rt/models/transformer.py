"""Non-stationary Transformer for real-time anomaly detection."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_sliding_windows(tensor: torch.Tensor, window_size: int) -> torch.Tensor:
    """[n, d] -> [n-window+1, window, d]"""
    n, d = tensor.size()
    return torch.stack([tensor[i:i + window_size] for i in range(n - window_size + 1)])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:x.size(0)])


class _TransformerEncoderLayer(nn.Module):
    """Single encoder layer with LeakyReLU."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, src: torch.Tensor, **kwargs) -> torch.Tensor:
        src = src + self.dropout1(self.self_attn(src, src, src)[0])
        src = src + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src)))))
        return src


class _TransformerDecoderLayer(nn.Module):
    """Single decoder layer with LeakyReLU (self-attn + cross-attn)."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = self.dropout2 = self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, **kwargs) -> torch.Tensor:
        tgt = tgt + self.dropout1(self.self_attn(tgt, tgt, tgt)[0])
        tgt = tgt + self.dropout2(self.cross_attn(tgt, memory, memory)[0])
        tgt = tgt + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(tgt)))))
        return tgt


class Projector(nn.Module):
    """MLP from (series_conv(x) || stats) -> de-stationary factors (tau, delta)."""

    def __init__(self, enc_in: int, seq_len: int, hidden_dims: list,
                 hidden_layers: int, output_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.series_conv = nn.Conv1d(seq_len, 1, kernel_size, padding=padding,
                                     padding_mode="circular", bias=False)
        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        x = self.series_conv(x)
        x = torch.cat([x, stats], dim=1).view(x.size(0), -1)
        return self.backbone(x)


class RTDetector(nn.Module):
    """Real-Time Detector: non-stationary Transformer with trend calibration."""

    def __init__(self, n_feats: int, n_window: int = 10, d_model: Optional[int] = None):
        super().__init__()
        self.n_feats = n_feats
        self.n_window = n_window
        self.d_model = d_model or 2 * n_feats
        self.lr = 1e-3
        self.batch_size = 128

        self.pos_encoder = PositionalEncoding(self.d_model, dropout=0.1, max_len=n_window)
        enc_layer = _TransformerEncoderLayer(self.d_model, n_feats, 16, 0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, 1)

        dec_layer1 = _TransformerDecoderLayer(self.d_model, n_feats, 16, 0.1)
        self.decoder1 = nn.TransformerDecoder(dec_layer1, 1)
        dec_layer2 = _TransformerDecoderLayer(self.d_model, n_feats, 16, 0.1)
        self.decoder2 = nn.TransformerDecoder(dec_layer2, 1)

        self.fcn = nn.Sequential(nn.Linear(self.d_model, n_feats), nn.Sigmoid())

        self.tau_learner = Projector(self.d_model, n_window, [16], 1, 1)
        self.delta_learner = Projector(self.d_model, n_window, [16], 1, n_window)

        self.register_buffer("std_enc", torch.randn(1, 1, self.d_model), persistent=False)
        self.register_buffer("mean_enc", torch.randn(1, 1, self.d_model), persistent=False)

    def encode(self, src: torch.Tensor, c: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # src: [window, batch, feats]; c: [window, batch, feats]; tgt: [1, batch, feats]
        src = torch.cat([src, c], dim=2).permute(1, 0, 2)  # [batch, window, 2*feats]
        mean_enc = src.mean(1, keepdim=True).detach()
        src = src - mean_enc
        std_enc = torch.sqrt(torch.var(src, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        src = src / std_enc
        self.std_enc.copy_(std_enc)
        self.mean_enc.copy_(mean_enc)

        tau = self.tau_learner(src, std_enc).exp().unsqueeze(1)  # [batch, 1, 1]
        delta = self.delta_learner(src, mean_enc).unsqueeze(1)    # [batch, 1, window]

        src = self.pos_encoder(src.permute(1, 0, 2))  # [window, batch, 2*feats]
        memory = self.encoder(src)

        memory = memory.permute(1, 0, 2)
        memory = memory * tau + delta.permute(0, 2, 1)  # [batch, window, 2*feats]
        memory = memory.permute(1, 0, 2)

        tgt = tgt.repeat(1, 1, 2).permute(1, 0, 2)
        delta_mean = delta.mean(dim=1, keepdim=True)
        tgt = tgt * tau + delta_mean
        return tgt.permute(1, 0, 2), memory

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # First pass (coarse)
        c = torch.zeros_like(src)
        tgt_c, mem = self.encode(src, c, tgt)
        out = self._decode_norm(self.decoder1(tgt_c, mem))
        x1 = self.fcn(out)

        # Second pass (refined with residual)
        c = (x1 - src) ** 2
        tgt_c, mem = self.encode(src, c, tgt)
        out = self._decode_norm(self.decoder1(tgt_c, mem))
        x2 = self.fcn(out)
        return x1, x2

    def _decode_norm(self, dec_out: torch.Tensor) -> torch.Tensor:
        out = dec_out.squeeze(0)
        out = out * self.std_enc.squeeze(1) + self.mean_enc.squeeze(1)
        return out.unsqueeze(0)
