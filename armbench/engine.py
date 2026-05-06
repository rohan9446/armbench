import copy
import torch
from armbench.profiler.timer import profile_latency
from armbench.profiler.memory import get_peak_memory_mb
from armbench.profiler.model_info import get_model_info


def run_benchmark(model, input_tensor, backends, warmup=5, runs=50):
    """Profile a model across multiple backends. Returns a list of result dicts."""
    results = []

    for backend in backends:
        print(f"  Profiling [{backend.name}]...")

        try:
            model_copy = copy.deepcopy(model)
            optimized = backend.prepare(model_copy)
        except RuntimeError as e:
            print(f"    SKIPPED: {e}")
            continue

        model_info = get_model_info(optimized)

        # match input to backend device and dtype
        if backend.name == "fp16":
            inp = input_tensor.half()
        elif backend.name == "cuda_fp32":
            inp = input_tensor.cuda()
        elif backend.name == "cuda_fp16":
            inp = input_tensor.half().cuda()
        else:
            inp = input_tensor.float()

        latency = profile_latency(optimized, inp, warmup=warmup, runs=runs)
        device = "cuda" if backend.name.startswith("cuda") else "cpu"
        peak_mem = get_peak_memory_mb(device=device)
        throughput = round(1000.0 / latency["p50_ms"], 2) if latency["p50_ms"] > 0 else 0

        results.append({
            "backend": backend.name,
            "latency": latency,
            "peak_memory_mb": peak_mem,
            "throughput_ops_sec": throughput,
            "param_count": model_info["param_count"],
            "model_size_mb": model_info["model_size_mb"],
            "disk_size_mb": model_info["disk_size_mb"],
        })

    return results