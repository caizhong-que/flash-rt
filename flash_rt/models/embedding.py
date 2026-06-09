"""Node embedding: Word2Vec + sinusoidal positional encoding."""

import math
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from gensim.models import Word2Vec


class PositionalEncoder(torch.nn.Module):
    """Sinusoidal positional encoder for node attribute sequences."""

    def __init__(self, d_model: int, max_len: int = 100000):
        super().__init__()
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0)]


class NodeEmbedder:
    """Word2Vec-based node attribute embedder with optional positional encoding."""

    def __init__(self, model_path: Union[str, Path], embedding_dim: int = 30,
                 use_positional_encoding: bool = True, max_seq_len: int = 100000):
        self.model_path = Path(model_path)
        self.embedding_dim = embedding_dim
        self.use_positional_encoding = use_positional_encoding
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.w2v_model = Word2Vec.load(str(self.model_path))
        actual_dim = self.w2v_model.vector_size
        if actual_dim != embedding_dim:
            self.embedding_dim = actual_dim
        self.pos_encoder = PositionalEncoder(self.embedding_dim, max_seq_len) if use_positional_encoding else None

    def infer(self, document: List[str]) -> np.ndarray:
        word_vecs = [self.w2v_model.wv[w] for w in document if w in self.w2v_model.wv]
        if not word_vecs:
            return np.zeros(self.embedding_dim)
        emb = torch.from_numpy(np.array(word_vecs, dtype=np.float32))
        if self.pos_encoder is not None and len(document) < 100000:
            emb = self.pos_encoder(emb)
        return emb.detach().cpu().numpy().mean(axis=0)

    def infer_batch(self, documents: List[List[str]]) -> np.ndarray:
        return np.array([self.infer(doc) for doc in documents])


def create_embedder(dataset_name: str = "cadets", checkpoint_dir: str = "checkpoints") -> NodeEmbedder:
    model_path = Path(checkpoint_dir) / dataset_name / f"word2vec_{dataset_name}_E3.model"
    return NodeEmbedder(model_path=model_path, embedding_dim=30, use_positional_encoding=True)
