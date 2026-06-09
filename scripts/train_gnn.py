"""GNN training entry point."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from flash_rt.data.preprocessor import filter_by_node_types, build_provenance_graph
from flash_rt.config import NODE_TYPE_MAP
from flash_rt.models.embedding import create_embedder
from flash_rt.training.gnn_trainer import GNNTrainer
from flash_rt.models.gnn import ProvenanceGNN
from flash_rt.training.callbacks import EpochLogger, EpochSaver

DATA_FILE = {
    "cadets": "ta1-cadets-e3-official-2.json.1.csv.gz",
    "fivedirections": "ta1-fivedirections-e3-official-2.json.csv.gz",
    "trace": "ta1-trace-e3-official-1.json.csv.gz",
    "theia": "ta1-theia-e3-official-6r.json.csv.gz",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", "-c", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iterations", type=int, default=22)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--embedding", action="store_true", help="Train Word2Vec first")
    args = parser.parse_args()

    data_dir = args.data_dir or Path(f"data/processed/{args.dataset}")
    ckpt_dir = args.checkpoint_dir or Path(f"checkpoints/{args.dataset}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(data_dir / DATA_FILE[args.dataset], compression="gzip")
    df.sort_values("timestamp", ascending=True, inplace=True)
    df = filter_by_node_types(df, set(NODE_TYPE_MAP[args.dataset].keys()))
    features, labels, edge_index, _ = build_provenance_graph(df, NODE_TYPE_MAP[args.dataset])

    if args.embedding:
        from gensim.models import Word2Vec
        Word2Vec(sentences=features, vector_size=30, window=5, min_count=1,
                 workers=16, epochs=100,
                 callbacks=[EpochLogger(10), EpochSaver(str(ckpt_dir), f"word2vec_{args.dataset}_E3")])

    embedder = create_embedder(args.dataset)
    node_emb = embedder.infer_batch(features)
    graph = Data(x=torch.tensor(node_emb, dtype=torch.float),
                 y=torch.tensor(labels, dtype=torch.long),
                 edge_index=torch.tensor(edge_index, dtype=torch.long)).to(device)

    model = ProvenanceGNN(node_emb.shape[1], int(max(labels)) + 1)
    trainer = GNNTrainer(model, device, max_iterations=args.max_iterations,
                         confidence_threshold=args.confidence_threshold)
    trainer.train_iterative(graph, labels, save_dir=ckpt_dir)
    print(f"Done. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
