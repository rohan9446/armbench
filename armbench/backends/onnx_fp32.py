import io
import numpy as np
import torch
import onnxruntime as ort
from armbench.backends.base import Backend


class OnnxWrapper(torch.nn.Module):
    """Wraps an ONNX Runtime session to behave like a PyTorch model."""

    def __init__(self, session, onnx_bytes):
        super().__init__()
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self._onnx_bytes = onnx_bytes

    def forward(self, x):
        result = self.session.run(None, {self.input_name: x.numpy()})
        return torch.from_numpy(result[0])

    def parameters(self):
        return iter([])

    def onnx_size_mb(self):
        return round(len(self._onnx_bytes) / (1024 * 1024), 2)


class OnnxFP32Backend(Backend):
    """Exports model to ONNX and runs inference via ONNX Runtime."""

    name = "onnx_fp32"

    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        buf = io.BytesIO()
        torch.onnx.export(model, dummy, buf, input_names=["input"], output_names=["output"])
        buf.seek(0)
        onnx_bytes = buf.read()
        session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
        return OnnxWrapper(session, onnx_bytes)
    