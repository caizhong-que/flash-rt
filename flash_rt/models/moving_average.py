"""Moving average anomaly detector (baseline)."""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class MADetectorConfig:
    window_sizes: tuple = (5, 10, 20)
    threshold_percentile: float = 95.0
    score_aggregation: str = "mean"


class MovingAverageAnomalyDetector:
    def __init__(self, config: MADetectorConfig = None):
        self.config = config or MADetectorConfig()
        self.normal_stats = {}
        self.threshold = None

    def fit(self, normal_data: np.ndarray):
        if normal_data.ndim == 1:
            normal_data = normal_data.reshape(-1, 1)
        n, nf = normal_data.shape
        for fi in range(nf):
            fd = normal_data[:, fi]
            self.normal_stats[fi] = {}
            for w in self.config.window_sizes:
                ma = self._moving_avg(fd, w)
                res = fd - ma
                self.normal_stats[fi][w] = {"mean": float(res.mean()), "mad": float(np.median(np.abs(res - np.median(res))))}
        _, scores, _ = self._compute_scores(normal_data)
        self.threshold = np.percentile(scores, self.config.threshold_percentile)
        return self

    def detect(self, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        if self.threshold is None:
            raise ValueError("Call fit() first")
        if test_data.ndim == 1:
            test_data = test_data.reshape(-1, 1)
        _, scores, details = self._compute_scores(test_data)
        return (scores > self.threshold).astype(int), scores, details

    def _compute_scores(self, data):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        all_feat = np.zeros((len(data), data.shape[1]))
        for fi in range(data.shape[1]):
            fd = data[:, fi]
            scale = []
            for w, st in self.normal_stats.get(fi, {}).items():
                ma = self._moving_avg(fd, w)
                res = fd - ma
                std_est = 1.4826 * st["mad"] if st["mad"] > 0 else np.std(res)
                scale.append(np.abs((res - st["mean"]) / (std_est + 1e-8)))
            all_feat[:, fi] = np.max(scale, axis=0) if scale else 0
        agg = all_feat.mean(axis=1) if self.config.score_aggregation == "mean" else all_feat.max(axis=1)
        return None, agg, {"per_feature": all_feat}

    @staticmethod
    def _moving_avg(data, window):
        n = len(data)
        kernel = np.ones(window) / window
        ma = np.convolve(data, kernel, mode="same")
        pad = window // 2
        for i in range(pad):
            ma[i] = data[:i + pad + 1].mean()
            ma[-(i + 1)] = data[-(i + pad + 1):].mean()
        return ma
