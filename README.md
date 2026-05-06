# ArmBench

A CLI tool for profiling PyTorch model inference across different optimization backends and hardware targets. Give it a model, pick your backends, and get back hard numbers on latency, memory, throughput, and model size.

I built this to understand the real trade-offs between precision levels (FP32, FP16, INT8), runtime optimizations (ONNX Runtime), pruning, and CPU vs GPU execution — the kind of analysis that matters when you're deploying models on resource-constrained hardware.

## Quick Start

```bash
git clone https://github.com/rohan9446/armbench.git
cd armbench
pip install -e .

# basic run
armbench run --model resnet18 --backends fp32 fp16 int8 --runs 50

# batch size scaling — critical for serving capacity planning
armbench run --model resnet50 --backends fp32 int8 onnx_fp32 --runs 20 --batch-sizes 1 4 8 16 32

# GPU profiling (auto-skips if no CUDA)
armbench run --model resnet50 --backends fp32 cuda_fp32 cuda_fp16 --runs 30

# custom model from disk
armbench run --model my_model.pt --backends fp32 int8 --runs 20

# custom ONNX model
armbench run --model my_model.onnx --backends fp32 --runs 20
```

## What It Measures

- **p50 / p95 / p99 latency** — tail latency is what actually matters in production SLAs, not averages
- **Peak memory** — system RSS on CPU, VRAM on GPU
- **Throughput** — ops/sec derived from median latency
- **Model size** — actual weight memory in MB
- **Disk size** — serialized model size on disk
- **Parameter count** — total params per backend (quantization can reduce this)

## Backends

| Backend | What it does | When it helps |
|---------|-------------|---------------|
| `fp32` | Baseline float32 | Reference point for all comparisons |
| `fp16` | Float16 weights | ~2x smaller model, fast on GPUs with FP16 cores |
| `int8` | Dynamic quantization | CPU-friendly, targets Linear + Conv2d layers |
| `pruned` | L1 unstructured pruning (30%) | Zeros out weights, reduces size on disk |
| `onnx_fp32` | ONNX Runtime inference | Graph optimizations + operator fusion, ~35% faster on CPU |
| `cuda_fp32` | FP32 on GPU | GPU baseline (auto-skips if no CUDA) |
| `cuda_fp16` | FP16 on GPU | Where FP16 actually shines — 16x faster than CPU FP32 |

CUDA backends detect GPU availability automatically. On CPU-only machines they skip with a message, no crashes.

## What I Found

These are real numbers from benchmarking ResNet50, not theoretical claims:

**ONNX Runtime is ~35% faster than PyTorch on CPU** — graph-level optimizations like operator fusion make a real difference even on the same hardware.

**FP16 is ~100x slower on x86 CPU** — there are no native FP16 compute units on x86, so every operation requires conversion overhead. On a GPU with dedicated FP16 cores, it's the fastest backend.

**CUDA FP16 is 16x faster than CPU FP32** — and half the model size. That's the whole case for GPU deployment in one benchmark.

**INT8 scales better than FP32 at higher batch sizes** — at batch 8, INT8 was 21% faster. Integer arithmetic has better throughput scaling on modern CPUs.

**Unstructured pruning doesn't speed things up** — it zeros weights without changing tensor shapes, so computation stays the same. You need structured pruning or a sparse runtime to see speedups.

## Batch Size Scaling

Profiling across batch sizes shows how models behave under load — critical for capacity planning:

```bash
armbench run --model resnet18 --backends fp32 int8 onnx_fp32 --runs 20 --batch-sizes 1 4 8 16 32
```

This produces results for every backend x batch size combination, so you can see exactly where throughput plateaus and memory becomes a bottleneck.

## Custom Models

ArmBench isn't limited to built-in models. Point it at any saved PyTorch model or ONNX file:

```bash
# PyTorch model (must be saved with torch.save(model, path), not state_dict)
armbench run --model my_model.pt --backends fp32 int8 --runs 20

# ONNX model
armbench run --model my_model.onnx --backends fp32 --runs 20
```

Built-in models: `resnet18`, `resnet50`, `mobilenet_v2`.

## Output

Every run produces two files in `results/`:

- `<model>_report.json` — raw structured data, easy to diff or pipe into other tools
- `<model>_report.html` — dark-themed dashboard with comparison table and bar charts for latency + model size

## Try It on GPU

No GPU? Run it on Google Colab for free:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohan9446/armbench/blob/main/notebooks/demo.ipynb)

## Project Structure

```
armbench/
├── backends/          # one file per optimization strategy
│   ├── base.py        # abstract Backend class (strategy pattern)
│   ├── fp32.py
│   ├── fp16.py
│   ├── int8.py
│   ├── pruned.py
│   ├── onnx_fp32.py
│   ├── cuda_fp32.py
│   └── cuda_fp16.py
├── profiler/
│   ├── timer.py       # warmup + timed runs, CUDA events for GPU
│   ├── memory.py      # RSS for CPU, VRAM for GPU
│   └── model_info.py  # param count, weight size, disk size
├── report/
│   ├── json_report.py
│   └── html_report.py
├── engine.py          # runs model x backends, collects results
└── cli.py             # entry point
```

## Adding a Backend

Subclass `Backend`, implement `prepare()`, register in `cli.py`:

```python
from armbench.backends.base import Backend

class MyBackend(Backend):
    name = "custom"

    def prepare(self, model):
        model.eval()
        # your optimization here
        return model
```

## CI

Every push runs benchmarks automatically via GitHub Actions — ResNet18 across FP32, INT8, ONNX Runtime, and pruned backends. Reports are uploaded as artifacts.

## What's Next

- Arm NN backend for real Arm hardware profiling
- Comparison mode — detect regressions between runs
- Warm-up analysis — prove stabilization over first N runs
- Structured pruning (actual channel removal for real speedup)
- CI regression detection with threshold alerts

## License

MIT