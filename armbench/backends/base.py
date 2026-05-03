from abc import ABC, abstractmethod
import torch


class Backend(ABC):
    """Base class all optimization backends must implement."""

    name: str = "base"

    @abstractmethod
    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        """Take a model and return its optimized variant."""
        ...

    def __repr__(self):
        return f"<Backend: {self.name}>"