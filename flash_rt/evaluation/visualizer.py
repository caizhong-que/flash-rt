"""Visualization: t-SNE, loss curves, confusion matrices."""

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE


def visualize_embeddings(embeddings: Union[torch.Tensor, np.ndarray],
                         labels: Union[torch.Tensor, np.ndarray, List[int]],
                         title: str = "Node Embeddings", figsize=(10, 10),
                         save_path: Optional[Path] = None) -> plt.Figure:
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    z = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(z[:, 0], z[:, 1], s=70, c=np.asarray(labels), cmap="Set2", alpha=0.7,
               edgecolors="white", linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_training_loss(losses: List[float], title: str = "Training Loss",
                       save_path: Optional[Path] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, linewidth=2, color="#2E86AB")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          normalize: bool = False, title: str = "Confusion Matrix",
                          save_path: Optional[Path] = None) -> plt.Figure:
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
