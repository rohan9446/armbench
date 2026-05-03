import time
import torch
import numpy as np


def profile_latency(model, input_tensor, warmup=5, runs=50):
    """Run inference multiple times and return p50/p95/p99 latencies in ms."""
    use_cuda = input_tensor.is_cuda

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(input_tensor)
            if use_cuda:
                torch.cuda.synchronize()

    # timed runs
    times = []
    with torch.no_grad():
        if use_cuda:
            for _ in range(runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(input_tensor)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
        else:
            for _ in range(runs):
                start = time.perf_counter()
                model(input_tensor)
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)

    times = np.array(times)
    return {
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "p99_ms": round(float(np.percentile(times, 99)), 3),
        "runs": runs,
    }