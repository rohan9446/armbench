import torch
from armbench.backends.base import Backend


class CudaFP16Backend(Backend):
    """Runs model inference on CUDA GPU in float16."""

    name = "cuda_fp16"

    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available — skipping cuda_fp16 backend")
        model.eval()
        return model.half().cuda()