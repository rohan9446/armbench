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
from armbench.backends.onnx_armnn import OnnxArmNNBackend

BACKENDS = {
    "fp32": FP32Backend,
    "fp16": FP16Backend,
    "int8": INT8Backend,
    "pruned": PrunedBackend,
    "cuda_fp32": CudaFP32Backend,
    "cuda_fp16": CudaFP16Backend,
    "onnx_fp32": OnnxFP32Backend,
    "onnx_armnn": OnnxArmNNBackend,
}

MODELS = {
    "resnet18": lambda: models.resnet18(weights=None),
    "resnet50": lambda: models.resnet50(weights=None),
    "mobilenet_v2": lambda: models.mobilenet_v2(weights=None),
}

def load_model(model_arg):
    """Load a built-in model or a custom .pt/.onnx file."""
    if model_arg in MODELS:
        return MODELS[model_arg](), model_arg

    if model_arg.endswith(".pt") or model_arg.endswith(".pth"):
        import os
        if not os.path.exists(model_arg):
            raise FileNotFoundError(f"Model file not found: {model_arg}")
        model = torch.load(model_arg, map_location="cpu", weights_only=False)
        if isinstance(model, dict):
            raise ValueError("File contains a state_dict, not a full model. Save with torch.save(model, path) instead of torch.save(model.state_dict(), path).")
        model.eval()
        name = os.path.splitext(os.path.basename(model_arg))[0]
        return model, name

    if model_arg.endswith(".onnx"):
        import os
        import onnxruntime as ort
        from armbench.backends.onnx_fp32 import OnnxWrapper
        if not os.path.exists(model_arg):
            raise FileNotFoundError(f"Model file not found: {model_arg}")
        with open(model_arg, "rb") as f:
            onnx_bytes = f.read()
        session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
        name = os.path.splitext(os.path.basename(model_arg))[0]
        return OnnxWrapper(session, onnx_bytes), name

    raise ValueError(f"Unknown model: {model_arg}. Use a built-in name ({', '.join(MODELS.keys())}) or a path to a .pt/.onnx file.")


def main():
    parser = argparse.ArgumentParser(description="ArmBench — ML Inference Profiler")
    parser.add_argument("command", choices=["run"], help="Command to execute")
    parser.add_argument("--model", required=True, help="Built-in model name or path to .pt/.onnx file")
    parser.add_argument("--backends", nargs="+", default=["fp32"], choices=BACKENDS.keys(), help="Backends to profile")
    parser.add_argument("--runs", type=int, default=50, help="Number of timed runs")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None, help="Batch sizes to profile (e.g. 1 4 8 16 32)")

    args = parser.parse_args()

    print(f"\nArmBench — Profiling {args.model}")
    print(f"  Backends: {', '.join(args.backends)}")
    print(f"  Runs: {args.runs}\n")

    model, model_name = load_model(args.model)
    backend_instances = [BACKENDS[b]() for b in args.backends]

    if args.batch_sizes:
        all_results = []
        for bs in args.batch_sizes:
            print(f"--- Batch size: {bs} ---")
            input_tensor = torch.randn(bs, 3, 224, 224)
            results = run_benchmark(model, input_tensor, backend_instances, runs=args.runs)
            for r in results:
                r["batch_size"] = bs
            all_results.extend(results)
        save_json_report(all_results, args.model)
        save_html_report(all_results, args.model)
    else:
        input_tensor = torch.randn(1, 3, 224, 224)
        results = run_benchmark(model, input_tensor, backend_instances, runs=args.runs)
        for r in results:
            r["batch_size"] = 1
        save_json_report(results, args.model)
        save_html_report(results, args.model)

    print("\nDone.")


if __name__ == "__main__":
    main()