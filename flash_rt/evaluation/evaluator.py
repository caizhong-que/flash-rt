"""End-to-end evaluation pipeline: GNN + Transformer inference."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric import utils

from flash_rt.models.gnn import ProvenanceGNN
from flash_rt.models.transformer import RTDetector, create_sliding_windows
from flash_rt.models.embedding import create_embedder
from flash_rt.data.preprocessor import build_provenance_graph, filter_by_node_types
from flash_rt.evaluation.metrics import evaluate_detection


class FlashEvaluator:
    """End-to-end evaluator for Flash APT detection system."""

    def __init__(self, dataset_name: str, device: torch.device,
                 checkpoint_dir: Path, data_dir: Path,
                 node_type_map: Optional[Dict[str, int]] = None,
                 gnn_iterations: int = 22):
        self.dataset_name = dataset_name
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.data_dir = Path(data_dir)
        self.node_type_map = node_type_map
        self.gnn_iterations = gnn_iterations
        self.transformer_window = 10
        self.batch_size = 5000
        self.mapping: List[str] = []
        self.gt_malicious: Set[str] = set()
        self.all_ids: Set[str] = set()
        self.edges: List[List[int]] = []

    def load_and_filter_data(self, test_file: str) -> pd.DataFrame:
        test_path = self.data_dir / test_file
        df = pd.read_csv(test_path, compression="gzip")
        df.sort_values("timestamp", ascending=True, inplace=True)
        df = filter_by_node_types(df, set(self.node_type_map.keys()))
        self.all_ids = set(df["actorID"]) | set(df["objectID"])
        return df

    def build_graph(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        features, labels, edge_index, mapping = build_provenance_graph(df, node_type_map=self.node_type_map)
        self.edges = edge_index
        self.mapping = mapping
        embedder = create_embedder(self.dataset_name)
        emb = embedder.infer_batch(features)
        return (torch.tensor(emb, dtype=torch.float),
                torch.tensor(labels, dtype=torch.long),
                torch.tensor(edge_index, dtype=torch.long), mapping)

    def load_ground_truth(self, gt_file: str) -> Set[str]:
        gt_path = self.data_dir / gt_file
        with open(gt_path) as f:
            gt = set(json.load(f))
        gt &= self.all_ids
        self.gt_malicious = gt
        return gt

    def _compute_confidence(self, logits):
        sp = logits.sort(1, descending=True)[0]
        conf = (sp[:, 0] - sp[:, 1]) / (sp[:, 0] + 1e-8)
        return (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)

    def _compute_entropy(self, logits):
        probs = F.softmax(logits, 1)
        return -(probs * torch.log(probs + 1e-12)).sum(1)

    def gnn_inference(self, nodes, labels, edge_index) -> Tuple[torch.Tensor, ...]:
        graph = Data(x=nodes, y=labels, edge_index=edge_index).to(self.device)
        graph.n_id = torch.arange(graph.num_nodes)
        num = graph.num_nodes
        raw_scores = torch.zeros(num)
        confidences = torch.zeros(num)
        entropies = torch.zeros(num)
        flag = torch.ones(num, dtype=torch.bool)

        model = ProvenanceGNN(nodes.size(1), int(labels.max().item()) + 1).to(self.device)
        for it in range(self.gnn_iterations):
            ckpt = self.checkpoint_dir / f"gnn_iter{it}.pth"
            if not ckpt.exists():
                continue
            state = torch.load(ckpt, map_location=self.device)
            model.load_state_dict(state.get("model_state_dict", state))
            model.eval()
            with torch.no_grad():
                for subg in NeighborLoader(graph, [-1, -1], self.batch_size):
                    subg = subg.to(self.device)
                    out = model(subg.x, subg.edge_index)
                    conf = self._compute_confidence(out)
                    ent = self._compute_entropy(out)
                    confidences[subg.n_id] = conf.cpu()
                    raw_scores[subg.n_id] = (1.0 - conf).cpu()
                    entropies[subg.n_id] = ent.cpu()
                    correct = out.argmax(1) == subg.y
                    flag[subg.n_id[correct.cpu()]] = False
        return raw_scores, confidences, entropies, flag

    def transformer_inference(self, triple: torch.Tensor, flag: torch.Tensor) -> torch.Tensor:
        windows = create_sliding_windows(triple, self.transformer_window)
        loader = DataLoader(TensorDataset(windows, windows), 128, shuffle=False)
        model = RTDetector(triple.size(1), n_window=self.transformer_window)
        ckpt = self.checkpoint_dir / "transformer_final.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Transformer checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location=self.device)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
        model.to(self.device).eval()
        criterion = torch.nn.MSELoss(reduction="none")
        all_losses = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.permute(1, 0, 2).to(self.device)
                elem = batch_x[-1].unsqueeze(0)
                z = model(batch_x, elem)
                if isinstance(z, tuple):
                    z = z[1]
                all_losses.append(criterion(z, elem).cpu())
        all_losses = torch.cat(all_losses, dim=1).squeeze(0)
        scores = torch.zeros(len(triple))
        scores[self.transformer_window - 1:] = all_losses.mean(dim=1)
        return scores

    def evaluate_at(self, scores: torch.Tensor, flag: torch.Tensor,
                    percentile: float, name: str) -> Dict:
        thresh = np.percentile(scores.numpy(), percentile)
        pred_mask = (scores > thresh) & flag
        pred_ids = {self.mapping[i] for i in utils.mask_to_index(pred_mask).tolist()}
        return evaluate_detection(pred_ids, self.all_ids, self.gt_malicious, self.edges, self.mapping)

    def run(self, test_file: str, gt_file: str, gnn_only: bool = False) -> Dict[str, Dict]:
        df = self.load_and_filter_data(test_file)
        nodes, labels, edge_index, mapping = self.build_graph(df)
        self.load_ground_truth(gt_file)
        rs, conf, entropy, flag = self.gnn_inference(nodes, labels, edge_index)

        results = {}
        for pct in [30, 53, 70, 90, 95]:
            results[f"gnn_p{pct}"] = self.evaluate_at(1.0 - conf, flag, pct, "GNN")
        if not gnn_only:
            triple = torch.stack([conf, entropy], dim=1)
            rt_scores = self.transformer_inference(triple, flag)
            for pct in [90, 92, 95]:
                results[f"flash_p{pct}"] = self.evaluate_at(rt_scores, flag, pct, "FLASH")
        return results
