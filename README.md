# FLASH-RT: A Framework for Online Node Anomaly Detection in Provenance Graphs via Trend Calibration

**FLASH-RT** (Framework for online node anomaly detection in provenance graphs via trend calibration) is a lightweight anomaly detection framework for streaming provenance graphs. It addresses two key challenges in host-based intrusion detection:

- **Progressive APT attacks & behavioral drift** – APT activities often span multiple time windows and exhibit weak, cumulative anomalies. FLASH-RT introduces a **trend calibration module** that models cross‑window node state sequences (anomaly score, confidence, entropy) to refine base detection results.
- **Online inference efficiency** – Repeated graph encoding across consecutive windows is computationally expensive. FLASH-RT incorporates an **embedding reuse mechanism** via persistent node identifiers (PNI) and Jaccard‑based neighborhood matching, significantly reducing latency.

This repository is the official implementation of the paper  
*"FLASH-RT: A framework for online node anomaly detection in provenance graphs via trend calibration"* (Que & Dong, 2025, under review).

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

**Key innovations:**
- **Time‑aware sequential encoding** – Adds sinusoidal positional encoding to node semantic sequences to capture local event order.
- **Trend calibration** – Models low‑dimensional node state sequences and uses reconstruction residuals to correct the base anomaly score, improving detection of progressive and subtle anomalies.
- **Embedding reuse** – Reuses GraphSAGE embeddings for structurally stable nodes across windows via PNI matching and Jaccard index (only when $J_v = 1$).

---

## Project Structure



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

## Typical Results (from paper, Table 1)

| Dataset (E3)       | Method       | Precision | Recall  | F1-Score |
|--------------------|--------------|-----------|---------|----------|
| Cadets             | ThreaTrace   | 0.9000    | 0.9900  | 0.9500   |
|                    | FLASH        | 0.9276    | 0.9995  | 0.9622   |
|                    | **FLASH-RT** | **1.0000**| **0.9993**| **0.9996**|
| Trace              | **FLASH-RT** | 0.9427    | 1.0000  | 0.9705   |
| Theia              | **FLASH-RT** | 0.9998    | 0.9989  | 0.9994   |

On OpTC, FLASH‑RT achieves F1‑scores of 0.9290, 0.9242, 0.9613 across three attack scenarios.  
See paper Table 1 for full comparison.

## Citation

```bibtex
@article{que2025flashrt,
  title={FLASH-RT: A framework for online node anomaly detection in provenance graphs via trend calibration},
  author={Caizhong Que and Guangfang Dong},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Notes

- This repository is the official implementation of the paper. The code and results are fully reproducible.
- The trend calibration module operates on low‑dimensional node state sequences (anomaly score, confidence, entropy) – **not** on raw graph embeddings – making it lightweight for online deployment.
- The embedding reuse mechanism requires **Persistent Node Identifiers (PNI)** as defined in Eq. (17) of the paper. See `flash_rt/models/reuse.py` for implementation details.
- For any questions, please open an issue or contact the authors.
