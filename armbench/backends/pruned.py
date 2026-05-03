import torch
import torch.nn.utils.prune as prune
from armbench.backends.base import Backend


class PrunedBackend(Backend):
    """Applies unstructured L1 pruning to Linear and Conv2d layers."""

    name = "pruned"

    def __init__(self, amount=0.3):
        self.amount = amount

    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        model.eval()
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                prune.l1_unstructured(module, name="weight", amount=self.amount)
                prune.remove(module, "weight")
        return model