import argparse
import torch
import torchvision.models as models
from armbench.backends.fp32 import FP32Backend
from armbench.backends.fp16 import FP16Backend
from armbench.backends.int8 import INT8Backend
from armbench.engine import run_benchmark
from armbench.report.json_report import save_json_report
from armbench.report.html_report import save_html_report
from armbench.backends.pruned import PrunedBackend
from armbench.backends.cuda_fp32 import CudaFP32Backend
from armbench.backends.cuda_fp16 import CudaFP16Backend
from armbench.backends.onnx_fp32 import OnnxFP32Backend

BACKENDS = {
    "fp32": FP32Backend,
    "fp16": FP16Backend,
    "int8": INT8Backend,
    "pruned": PrunedBackend,
    "cuda_fp32": CudaFP32Backend,
    "cuda_fp16": CudaFP16Backend,
    "onnx_fp32": OnnxFP32Backend,
}

MODELS = {
    "resnet18": lambda: models.resnet18(weights=None),
    "resnet50": lambda: models.resnet50(weights=None),
    "mobilenet_v2": lambda: models.mobilenet_v2(weights=None),
}


def main():
    parser = argparse.ArgumentParser(description="ArmBench — ML Inference Profiler")
    parser.add_argument("command", choices=["run"], help="Command to execute")
    parser.add_argument("--model", required=True, choices=MODELS.keys(), help="Model to benchmark")
    parser.add_argument("--backends", nargs="+", default=["fp32"], choices=BACKENDS.keys(), help="Backends to profile")
    parser.add_argument("--runs", type=int, default=50, help="Number of timed runs")

    args = parser.parse_args()

    print(f"\nArmBench — Profiling {args.model}")
    print(f"  Backends: {', '.join(args.backends)}")
    print(f"  Runs: {args.runs}\n")

    model = MODELS[args.model]()
    input_tensor = torch.randn(1, 3, 224, 224)
    backend_instances = [BACKENDS[b]() for b in args.backends]

    results = run_benchmark(model, input_tensor, backend_instances, runs=args.runs)
    save_json_report(results, args.model)
    save_html_report(results, args.model)

    print("\nDone.")


if __name__ == "__main__":
    main()