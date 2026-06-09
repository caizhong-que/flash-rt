# FLASH-RT: Real-Time APT Detection in Provenance Graphs

**FLASH** (**F**ast, **L**ightweight **A**nomaly Detection via **S**elf-Training and Transformer-based **H**euristics) is a real-time Advanced Persistent Threat (APT) detection system for provenance graphs. It combines Graph Neural Networks (GNN) with a non-stationary Transformer for high-accuracy streaming anomaly detection.

## Architecture

```
Raw CDM Logs (tar.gz) → CDMParser → Edge CSV
    → Graph Construction (filter + build)
    → Word2Vec Node Embedding (semantic + positional encoding)
    → GNN Iterative Self-Training (GraphSAGE, 22 iterations)
        Output: raw_score, confidence, entropy per node
    → Non-stationary Transformer (trend calibration)
        Anomaly scoring via reconstruction error
    → Evaluation (2-hop relaxation metrics)
```

## Project Structure

```
FLASH-RT/
├── flash_rt/                # Main package (pip install -e .)
│   ├── config.py            # NODE_TYPE_MAP, constants
│   ├── data/                # CDM parser, graph construction, downloader
│   ├── models/              # GNN, Transformer, LSTM, MovingAverage
│   ├── training/            # Iterative GNN trainer, Transformer trainer
│   └── evaluation/          # Evaluator, metrics, visualizer
├── experiments/
│   ├── optc/                # OpTC streaming benchmark pipeline
│   │   ├── pipeline.py      # Entry point
│   │   ├── rt_model.py      # RTDetector (trend calibration)
│   │   ├── gnn_model.py     # GNN for OpTC
│   │   └── ...              # Feature store, streaming, monitoring
├── scripts/                 # CLI entry points
│   ├── train_gnn.py
│   ├── train_transformer.py
│   └── evaluate.py
├── data/
│   └── labels/              # Ground truth (small, version-controlled)
├── results/
│   └── figures/
├── checkpoints/             # Model weights (not tracked)
├── setup.py                 # pip-installable
├── requirements.txt
└── .gitignore
```

## Datasets

Supports **DARPA E3** (Cadets, FiveDirections, Theia, Trace) and **OpTC** (real enterprise telemetry).

### Data Download

Raw data is large (~GB), download separately:

```bash
python flash_rt/data/downloader.py --dataset cadets
python flash_rt/data/downloader.py --dataset optc
```

## Installation

```bash
# Create environment (conda recommended)
conda create -n flash python=3.10
conda activate flash

# Install PyTorch (CUDA 11.8)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install FLASH-RT and remaining deps
pip install -e .
pip install -r requirements.txt
```

## Usage

### 1. Parse raw data

```bash
python flash_rt/data/parser.py \
  --input data/raw/cadets --output data/processed/cadets
```

### 2. Train GNN (iterative self-training)

```bash
python scripts/train_gnn.py --dataset cadets --device cuda
python scripts/train_gnn.py --dataset cadets --embedding  # train Word2Vec first
```

### 3. Train Transformer

```bash
python scripts/train_transformer.py --dataset cadets --triple-path data/triple.pt
```

### 4. Evaluate

```bash
python scripts/evaluate.py --dataset cadets --device cuda
python scripts/evaluate.py --dataset cadets --gnn-only  # skip Transformer
```

### 5. OpTC streaming benchmark

```bash
cd experiments/optc
python pipeline.py
```

## Typical Results (Cadets E3)

| Method | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| GNN (p50) | 0.907 | 0.999 | 0.951 |
| GNN (p95) | 0.988 | 0.999 | 0.993 |
| FLASH (p50) | 0.989 | 0.999 | 0.994 |
| FLASH (p95) | 0.996 | 0.999 | 0.997 |

## Citation

```bibtex
@article{flash-rt,
  title={FLASH-RT: Real-Time APT Detection in Provenance Graphs via
         Self-Training and Non-Stationary Transformers},
  author={Caizhong Que and Guangfang Dong},
  journal={...},
  year={2024}
}
```

This project is provided for academic research purposes.

