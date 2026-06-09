import gc
import psutil
import torch
from collections import defaultdict

from .config import batch_sizes
from .stream_loader import stream_events, load_ground_truth
from .evaluation_utils import analyze_events
from .monitor import ResourceMonitor

def run_stream_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    gt_nodes = load_ground_truth(".\\data\\OpTC\\optc.txt")
    groups = defaultdict(list)

    for batch_size in batch_sizes:
        print(f"\nTesting batch_size={batch_size}")
        batch_results, hit_results, all_metrics = [], [], []

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        monitor = ResourceMonitor(device=device)
        process = psutil.Process()

        for i, data_frame in enumerate(stream_events(batch_size, 250), 1):
            ram_before = process.memory_info().rss / 1024 ** 2

            monitor.start()
            result, hitrate = analyze_events(data_frame, gt_nodes)
            metrics = monitor.stop()

            current_ram = process.memory_info().rss / 1024 ** 2
            metrics["net_ram_mb"] = current_ram - ram_before

            batch_results.append(result)
            hit_results.append(hitrate)
            all_metrics.append(metrics)

            if i % 10 == 0:
                print(
                    f"  Batch {i}: Time={metrics['time_s']:.3f}s, "
                    f"GPU={metrics.get('gpu_peak_mb', 0):.1f}MB"
                )

            if i == 1000:
                break

        avg_metrics = {
            "batch_size": batch_size,
            "num_batches": i,

            "avg_time_s": sum(m["time_s"] for m in all_metrics) / len(all_metrics),
            "avg_time_ms": sum(m["time_ms"] for m in all_metrics) / len(all_metrics),
            "total_time_s": sum(m["time_s"] for m in all_metrics),

            "avg_ram_mb": sum(m["ram_mb"] for m in all_metrics) / len(all_metrics),
            "max_ram_mb": max(m["ram_mb"] for m in all_metrics),

            "avg_gpu_mb": sum(m["gpu_peak_mb"] for m in all_metrics) / len(all_metrics),
            "max_gpu_mb": max(m["gpu_peak_mb"] for m in all_metrics),

            "avg_cpu_percent": sum(m["cpu_percent"] for m in all_metrics) / len(all_metrics),

            "avg_net_ram_mb": sum(m["net_ram_mb"] for m in all_metrics) / len(all_metrics),
            "max_net_ram_mb": max(m["net_ram_mb"] for m in all_metrics),
        }

        avg_prf = {
            "avg_precision": sum(r[0] for r in batch_results) / len(batch_results),
            "avg_recall": sum(r[1] for r in batch_results) / len(batch_results),
            "avg_f1": sum(r[2] for r in batch_results) / len(batch_results),
        }

        avg_hitrate = sum(hit_results) / len(hit_results)

        groups["batch_size"].append(batch_size)
        groups["avg_metrics"].append(avg_metrics)
        groups["avg_PRF"].append(avg_prf)
        groups["avg_hitrate"].append(avg_hitrate)

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print("\n" + "=" * 130)
    print(
        f"{'Batch':<10} {'Num':<6} {'Time(s)':<10} {'RAM-max(MB)':<14} {'GPU-max(MB)':<14} "
        f"{'P':<8} {'R':<8} {'F1':<8} {'Hit':<8} {'CPU%':<8}"
    )
    print("-" * 130)

    for i, bs in enumerate(groups["batch_size"]):
        m = groups["avg_metrics"][i]
        prf = groups["avg_PRF"][i]
        hit = groups["avg_hitrate"][i]

        print(
            f"{bs:<10} {m['num_batches']:<6} {m['avg_time_s']:<10.4f} "
            f"{m['avg_ram_mb']:<14.1f} {m['max_gpu_mb']:<14.4f} "
            f"{prf['avg_precision']:<8.4f} {prf['avg_recall']:<8.4f} "
            f"{prf['avg_f1']:<8.4f} {hit:<8.4f} {m['avg_cpu_percent']:<8.4f}"
        )

    return groups
