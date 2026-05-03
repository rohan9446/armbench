import torch
from armbench.backends.base import Backend


class INT8Backend(Backend):
    """Applies PyTorch dynamic quantization (INT8)."""

    name = "int8"

    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        model.eval()
        return torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )