import time
from contextlib import contextmanager
import torch

def cuda_sync_if_available():
    if  torch.cuda.is_available():
        torch.cuda.synchronize()

@contextmanager
def timer(name: str, cuda_sync: bool = False, enabled: bool = True):
    if not enabled:
        yield
        return

    if cuda_sync:
        cuda_sync_if_available()

    start = time.perf_counter()
    try:
        yield
    finally:
        if cuda_sync:
            cuda_sync_if_available()
        cost = time.perf_counter() - start
        print(f"[TIME] {name}: {cost:.6f}s")