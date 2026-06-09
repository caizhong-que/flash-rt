import os
import time
import psutil
import torch

class ResourceMonitor:
    def __init__(self, device="cuda"):
        self.device = device
        self.process = psutil.Process(os.getpid())
        self.cpu_count = psutil.cpu_count()

    def start(self):
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        self.time_start = time.perf_counter()
        self.cpu_start_time = time.time()
        self.mem_start = self.process.memory_info().rss / 1024 ** 2

        if torch.cuda.is_available() and self.device == "cuda":
            self.gpu_mem_start = torch.cuda.memory_allocated() / 1024 ** 2

        self.process.cpu_percent(interval=None)

    def stop(self):
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.synchronize()

        elapsed_s = time.perf_counter() - self.time_start
        cpu_percent = self.process.cpu_percent(interval=None) / self.cpu_count
        mem_info = self.process.memory_info()
        mem_current = mem_info.rss / 1024 ** 2

        result = {
            "time_s": elapsed_s,
            "time_ms": elapsed_s * 1000,
            "cpu_percent": cpu_percent,
            "ram_mb": mem_current,
            "ram_start_mb": self.mem_start,
            "ram_delta_mb": mem_current - self.mem_start,
        }

        if torch.cuda.is_available() and self.device == "cuda":
            gpu_mem_current = torch.cuda.memory_allocated() / 1024 ** 2
            gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024 ** 2

            result.update({
                "gpu_mb": gpu_mem_current,
                "gpu_peak_mb": gpu_mem_peak,
                "gpu_start_mb": self.gpu_mem_start,
                "gpu_delta_mb": gpu_mem_current - self.gpu_mem_start,
            })
        else:
            result.update({
                "gpu_mb": 0,
                "gpu_peak_mb": 0,
                "gpu_start_mb": 0,
                "gpu_delta_mb": 0,
            })

        return result