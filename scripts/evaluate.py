"""Evaluation entry point for FLASH APT detection."""

import argparse
from pathlib import Path

import torch

from flash_rt.evaluation.evaluator import FlashEvaluator
from flash_rt.config import NODE_TYPE_MAP


def get_dataset_config(dataset: str):
    configs = {
        "cadets": {"test_file": "ta1-cadets-e3-official-2.json.csv.gz",
                   "gt_file": "cadets.json",
                   "node_types": NODE_TYPE_MAP[dataset]},
        "fivedirections": {"test_file": "ta1-fivedirections-e3-official-2.json.23.csv.gz",
                           "gt_file": "fivedirections.json",
                           "node_types": NODE_TYPE_MAP[dataset]},
        "theia": {"test_file": "ta1-theia-e3-official-6r.json.8.csv.gz",
                  "gt_file": "theia.json",
                  "node_types": NODE_TYPE_MAP[dataset]},
        "trace": {"test_file": "ta1-trace-e3-official-1.json.4.csv.gz",
                  "gt_file": "trace.json",
                  "node_types": NODE_TYPE_MAP[dataset]},
    }
    return configs.get(dataset)


def main():
    parser = argparse.ArgumentParser(description="FLASH APT Detection Evaluation")
    parser.add_argument("--dataset", "-d", default="cadets",
                        choices=["cadets", "fivedirections", "theia", "trace", "optc"])
    parser.add_argument("--checkpoint-dir", "-c", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gnn-only", action="store_true")
    parser.add_argument("--gnn-iterations", type=int, default=22)
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir or Path(f"checkpoints/{args.dataset}")
    data_dir = args.data_dir or Path(f"data/processed/{args.dataset}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = get_dataset_config(args.dataset)

    evaluator = FlashEvaluator(args.dataset, device, ckpt_dir, data_dir,
                               cfg["node_types"], gnn_iterations=args.gnn_iterations)
    results = evaluator.run(cfg["test_file"], cfg["gt_file"], gnn_only=args.gnn_only)

    for name, m in results.items():
        print(f"\n{name}: Precision={m.get('precision',0):.4f}, "
              f"Recall={m.get('recall',0):.4f}, F1={m.get('fscore',0):.4f}, "
              f"FPR={m.get('FPR',0):.4f}")


if __name__ == "__main__":
    main()
