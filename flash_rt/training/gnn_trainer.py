"""GNN trainer with iterative self-training."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from flash_rt.models.gnn import ProvenanceGNN
from flash_rt.models.embedding import create_embedder


class GNNTrainer:
    """Iterative self-training for GNN with confidence-based sample filtering."""

    def __init__(self, model: ProvenanceGNN, device: torch.device,
                 lr: float = 0.01, weight_decay: float = 5e-4,
                 confidence_threshold: float = 0.9, max_iterations: int = 22,
                 batch_size: int = 5000, num_neighbors: List[int] = None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors or [-1, -1]
        self.criterion = None
        self.training_history = []

    def train_iterative(self, graph: Data, labels: np.ndarray,
                        save_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
        weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
        self.criterion = CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(self.device))
        num_nodes = graph.num_nodes
        mask = torch.ones(num_nodes, dtype=torch.bool)
        raw_scores = np.zeros(num_nodes)
        confidences = np.zeros(num_nodes)
        entropies = np.zeros(num_nodes)
        graph.n_id = torch.arange(num_nodes)

        for it in range(self.max_iterations):
            remaining = mask.sum().item()
            if remaining == 0:
                break
            self._train_epoch(graph, mask)
            m = self._evaluate_and_filter(graph, mask)
            confidences[m["evaluated_nodes"]] = m["confidence"]
            raw_scores[m["evaluated_nodes"]] = m["raw_score"]
            entropies[m["evaluated_nodes"]] = m["entropy"]
            mask[m["filtered_nodes"]] = False
            if save_dir:
                torch.save({"iteration": it, "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict()},
                           save_dir / f"gnn_iter{it}.pth")
        return {"raw_score": raw_scores, "confidence": confidences, "entropy": entropies, "final_mask": mask.numpy()}

    def _train_epoch(self, graph, mask):
        self.model.train()
        for subg in NeighborLoader(graph, self.num_neighbors, self.batch_size, input_nodes=mask):
            subg = subg.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(subg.x, subg.edge_index), subg.y)
            loss.backward()
            self.optimizer.step()

    def _confidence(self, logits):
        sp = logits.sort(1, descending=True)[0]
        conf = (sp[:, 0] - sp[:, 1]) / (sp[:, 0] + 1e-8)
        return (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)

    def _entropy(self, logits):
        probs = F.softmax(logits, 1)
        return -(probs * torch.log(probs + 1e-12)).sum(1)

    def _evaluate_and_filter(self, graph, mask):
        self.model.eval()
        results = {k: [] for k in ("confidence", "raw_score", "entropy", "evaluated_nodes", "filtered_nodes")}
        with torch.no_grad():
            for subg in NeighborLoader(graph, self.num_neighbors, self.batch_size, input_nodes=mask):
                subg = subg.to(self.device)
                out = self.model(subg.x, subg.edge_index)
                conf = self._confidence(out)
                entropy = self._entropy(out)
                correct = out.argmax(1) == subg.y
                to_filter = correct | (conf >= self.confidence_threshold)
                results["confidence"].append(conf.cpu())
                results["raw_score"].append((1.0 - conf).cpu())
                results["entropy"].append(entropy.cpu())
                results["evaluated_nodes"].append(subg.n_id.cpu())
                results["filtered_nodes"].append(subg.n_id[to_filter].cpu())
        return {k: torch.cat(v).numpy() for k, v in results.items()}


def train_gnn(features, labels, edges, embedding_dim=30, num_classes=6,
              device=None, save_dir=None) -> Tuple[ProvenanceGNN, Dict]:
    """End-to-end: embed -> build graph -> iterative GNN training."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = create_embedder()
    node_emb = embedder.infer_batch(features)
    graph = Data(x=torch.tensor(node_emb, dtype=torch.float),
                 y=torch.tensor(labels, dtype=torch.long),
                 edge_index=torch.tensor(edges, dtype=torch.long)).to(device)
    model = ProvenanceGNN(embedding_dim, num_classes)
    trainer = GNNTrainer(model, device)
    metrics = trainer.train_iterative(graph, np.array(labels), Path(save_dir) if save_dir else None)
    return model, metrics
