import torch
from armbench.backends.base import Backend


class FP16Backend(Backend):
    """Converts model weights to float16."""

    name = "fp16"

    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        model.eval()
        return model.half()