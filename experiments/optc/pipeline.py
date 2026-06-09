"""OpTC streaming benchmark pipeline."""

import gc
import json
from collections import defaultdict

import psutil
import torch

from .config import BATCH_SIZES, DATA_DIR


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt_nodes = load_ground_truth(DATA_DIR / "optc.txt")
    groups = defaultdict(list)

    for batch_size in BATCH_SIZES:
        _, all_metrics, batch_results, hit_results = _run_batch_loop(batch_size, gt_nodes, device)

        avg_m = {k: sum(m[k] for m in all_metrics) / len(all_metrics)
                 for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))}
        avg_m["batch_size"] = batch_size
        avg_m["num_batches"] = len(all_metrics)

        avg_prf = {k: sum(r[k] for r in batch_results) / len(batch_results)
                   for k in ("precision", "recall", "f1")}
        avg_hit = sum(hit_results) / len(hit_results)

        groups["batch_size"].append(batch_size)
        groups["avg_metrics"].append(avg_m)
        groups["avg_PRF"].append(avg_prf)
        groups["avg_hitrate"].append(avg_hit)

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    _print_summary(groups)
    return groups


def _run_batch_loop(batch_size, gt_nodes, device):
    from .event_processing import process_dataframe
    from .stream_loader import stream_events
    from .monitor import ResourceMonitor

    from .evaluation_utils import load_pkl, evaluate_frame
    from .config import XGBOOST_WEIGHTS

    model = load_pkl(XGBOOST_WEIGHTS)
    all_metrics, batch_results, hit_results = [], [], []
    monitor = ResourceMonitor(device=device)
    process = psutil.Process()

    for i, frame in enumerate(stream_events(batch_size, 250), 1):
        monitor.start()
        result, hitrate = evaluate_frame(frame, model, gt_nodes)
        metrics = monitor.stop()
        metrics["net_ram_mb"] = process.memory_info().rss / 1_048_576 - 0
        batch_results.append(result)
        hit_results.append(hitrate)
        all_metrics.append(metrics)
        if i >= 1000:
            break
    return i, all_metrics, batch_results, hit_results


def _print_summary(groups):
    print("\n" + "=" * 130)
    header = f"{'Batch':<10} {'Num':<6} {'Time(s)':<10} {'RAM(MB)':<12} {'P':<8} {'R':<8} {'F1':<8}"
    print(header)
    print("-" * 130)
    for i, bs in enumerate(groups["batch_size"]):
        m = groups["avg_metrics"][i]
        prf = groups["avg_PRF"][i]
        print(f"{bs:<10} {m['num_batches']:<6} {m['time_s']:<10.4f} "
              f"{m.get('ram_mb', 0):<12.1f} {prf['precision']:<8.4f} "
              f"{prf['recall']:<8.4f} {prf['f1']:<8.4f}")


def load_ground_truth(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


if __name__ == "__main__":
    run()
