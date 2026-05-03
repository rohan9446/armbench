import torch
from armbench.backends.base import Backend


class FP32Backend(Backend):
    """Baseline backend — runs the model as-is in float32."""

    name = "fp32"

    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        model.eval()
        return model.float()