# ArmBench

A CLI tool for profiling PyTorch model inference across different optimization backends. Give it a model, pick your backends, and get back hard numbers on latency, memory, throughput, and model size.

I built this to understand the real trade-offs between precision levels (FP32, FP16, INT8) and optimizations like pruning — the kind of analysis that matters when you're deploying models on resource-constrained hardware.

## Quick Start

```bash
git clone https://github.com/rohan9446/armbench.git
cd armbench
pip install -e .

# basic run
armbench run --model resnet18 --backends fp32 fp16 int8 --runs 50

# batch size scaling — critical for serving capacity planning
armbench run --model resnet50 --backends fp32 int8 onnx_fp32 --runs 20 --batch-sizes 1 4 8 16 32

# custom model from disk
armbench run --model my_model.pt --backends fp32 int8 --runs 20

# custom ONNX model
armbench run --model my_model.onnx --backends fp32 --runs 20
```

## What It Measures

- **p50 / p95 / p99 latency** — not averages, because tail latency is what actually matters in production
- **Peak memory** — system RSS on CPU, VRAM on GPU
- **Throughput** — ops/sec derived from median latency
- **Model size** — actual weight memory in MB
- **Parameter count** — total trainable params per backend

## Backends

| Backend | What it does | When it helps |
|---------|-------------|---------------|
| `fp32` | Baseline float32 | Reference point |
| `fp16` | Float16 weights | ~2x smaller model, fast on GPUs with FP16 cores |
| `int8` | Dynamic quantization | CPU-friendly, targets Linear + Conv2d layers |
| `pruned` | L1 unstructured pruning (30%) | Zeros out weights, reduces size on disk |
| `cuda_fp32` | FP32 on GPU | GPU baseline (auto-skips if no CUDA) |
| `cuda_fp16` | FP16 on GPU | Where FP16 actually shines |

The CUDA backends detect GPU availability automatically — they skip with a message on CPU-only machines, no crashes.

## Interesting Findings

Running on CPU (x86), FP16 is roughly **100x slower** than FP32. That's because x86 chips don't have native FP16 compute units — every operation requires conversion overhead. On a GPU with dedicated FP16 cores, you'd see the opposite. This is one of those things that's obvious in hindsight but easy to miss if you've only read about quantization without actually benchmarking it.

INT8 matches or beats FP32 on CPU since integer arithmetic is natively fast.

## Models

Currently supports `resnet18`, `resnet50`, and `mobilenet_v2` out of the box. Adding more is just a one-liner in `cli.py`.

## Output

Every run produces two files in `results/`:

- `<model>_report.json` — raw structured data, easy to diff or pipe into other tools
- `<model>_report.html` — dark-themed dashboard with a comparison table and bar charts for latency + model size

## Project Structure

```
armbench/
├── backends/          # one file per optimization strategy
│   ├── base.py        # abstract Backend class (strategy pattern)
│   ├── fp32.py
│   ├── fp16.py
│   ├── int8.py
│   ├── pruned.py
│   ├── cuda_fp32.py
│   └── cuda_fp16.py
├── profiler/
│   ├── timer.py       # warmup + timed runs, CUDA events for GPU
│   ├── memory.py      # RSS for CPU, VRAM for GPU
│   └── model_info.py  # param count + weight size
├── report/
│   ├── json_report.py
│   └── html_report.py
├── engine.py          # runs model × backends, collects results
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

## What's Next

- Arm NN backend for real Arm hardware profiling
- Comparison mode — detect regressions between runs
- Warm-up analysis — prove stabilization over first N runs
- Structured pruning (actual channel removal for real speedup)
- CI regression detection with threshold alerts

## License

MIT
