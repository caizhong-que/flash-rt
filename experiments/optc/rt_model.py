"""RTDetector model for OpTC pipeline (without PE bug)."""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class _EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff=16, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.LeakyReLU(inplace=True),
                                nn.Dropout(dropout), nn.Linear(dim_ff, d_model))
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, **kw):
        x = x + self.drop1(self.attn(x, x, x)[0])
        x = x + self.drop2(self.ff(x))
        return x


class _DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff=16, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.LeakyReLU(inplace=True),
                                nn.Dropout(dropout), nn.Linear(dim_ff, d_model))
        self.drop1 = self.drop2 = self.drop3 = nn.Dropout(dropout)

    def forward(self, tgt, mem, **kw):
        tgt = tgt + self.drop1(self.self_attn(tgt, tgt, tgt)[0])
        tgt = tgt + self.drop2(self.cross_attn(tgt, mem, mem)[0])
        tgt = tgt + self.drop3(self.ff(tgt))
        return tgt


class Projector(nn.Module):
    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super().__init__()
        pad = 1 if torch.__version__ >= "1.5.0" else 2
        self.conv = nn.Conv1d(seq_len, 1, kernel_size, padding=pad, padding_mode="circular", bias=False)
        layers = []
        dims = [2 * enc_in] + hidden_dims
        for i in range(hidden_layers):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(dims[-1], output_dim, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, stats):
        x = self.conv(x)
        x = torch.cat([x, stats], dim=1).view(x.size(0), -1)
        return self.net(x)


class RTDetector(nn.Module):
    """Real-Time Detector with non-stationary Transformer (trend calibration)."""

    def __init__(self, feats: int):
        super().__init__()
        self.name = "RTDetector"
        self.n_feats = feats
        self.n_window = 10
        d_model = 2 * feats

        self.pos_encoder = PositionalEncoding(d_model, 0.1, self.n_window)
        self.encoder = nn.TransformerEncoder(_EncoderLayer(d_model, feats, 16, 0.1), 1)
        self.decoder1 = nn.TransformerDecoder(_DecoderLayer(d_model, feats, 16, 0.1), 1)
        self.fcn = nn.Sequential(nn.Linear(d_model, feats), nn.Sigmoid())

        self.tau_learner = Projector(d_model, self.n_window, [16], 1, 1)
        self.delta_learner = Projector(d_model, self.n_window, [16], 1, self.n_window)

        self.register_buffer("std_enc", torch.randn(1, 1, d_model), persistent=False)
        self.register_buffer("mean_enc", torch.randn(1, 1, d_model), persistent=False)

    def encode(self, src, c, tgt):
        src = torch.cat([src, c], dim=2).permute(1, 0, 2)
        mean_enc = src.mean(1, keepdim=True).detach()
        src = src - mean_enc
        std_enc = torch.sqrt(torch.var(src, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        src = src / std_enc
        self.std_enc.copy_(std_enc)
        self.mean_enc.copy_(mean_enc)

        tau = self.tau_learner(src, std_enc).exp().unsqueeze(1)
        delta = self.delta_learner(src, mean_enc).unsqueeze(1)

        src = self.pos_encoder(src.permute(1, 0, 2))
        memory = self.encoder(src)

        memory = memory.permute(1, 0, 2) * tau + delta.permute(0, 2, 1)
        memory = memory.permute(1, 0, 2)

        tgt = tgt.repeat(1, 1, 2).permute(1, 0, 2)
        tgt = tgt * tau + delta.mean(1, keepdim=True)
        return tgt.permute(1, 0, 2), memory

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = torch.zeros_like(src)
        tgt_c, mem = self.encode(src, c, tgt)
        out = self.decoder1(tgt_c, mem).squeeze(0)
        out = out * self.std_enc.squeeze(1) + self.mean_enc.squeeze(1)
        x1 = self.fcn(out.unsqueeze(0))

        c = (x1 - src) ** 2
        tgt_c, mem = self.encode(src, c, tgt)
        out = self.decoder1(tgt_c, mem).squeeze(0)
        out = out * self.std_enc.squeeze(1) + self.mean_enc.squeeze(1)
        x2 = self.fcn(out.unsqueeze(0))
        return x1, x2
