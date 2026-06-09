"""Transformer anomaly detector training entry point."""

import argparse
from pathlib import Path

import torch

from flash_rt.models.transformer import RTDetector
from flash_rt.training.transformer_trainer import TransformerTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--checkpoint-dir", "-c", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-feats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--triple-path", type=Path, default=None,
                        help="Path to pre-computed triple.pt (raw_score, confidence, entropy)")
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir or Path(f"checkpoints/{args.dataset}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.triple_path:
        triple = torch.load(args.triple_path)
    else:
        # Placeholder: user should provide triple via --triple-path
        raise RuntimeError("Provide --triple-path to pre-computed GNN output features")

    model = RTDetector(n_feats=args.n_feats, n_window=args.window_size)
    trainer = TransformerTrainer(model, device, window_size=args.window_size, n_epochs=args.epochs)
    trainer.fit(triple, save_dir=ckpt_dir)
    print(f"Done. Transformer model saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
