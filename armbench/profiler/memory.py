import os
import torch


def get_peak_memory_mb(device="cpu"):
    """Return peak memory in MB. Uses GPU VRAM for CUDA, system RSS for CPU."""
    if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()
        return round(peak, 2)

    try:
        import psutil
        process = psutil.Process(os.getpid())
        return round(process.memory_info().peak_wset / (1024 * 1024), 2)
    except ImportError:
        return -1.0