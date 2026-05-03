import torch
from armbench.backends.base import Backend


class CudaFP32Backend(Backend):
    """Runs model inference on CUDA GPU in float32."""

    name = "cuda_fp32"

    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available — skipping cuda_fp32 backend")
        model.eval()
        return model.cuda()