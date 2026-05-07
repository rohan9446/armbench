import io
import torch
import onnxruntime as ort
from armbench.backends.onnx_fp32 import OnnxWrapper
from armbench.backends.base import Backend


class OnnxArmNNBackend(Backend):
    """Runs inference via ONNX Runtime with Arm NN execution provider (ACL)."""

    name = "onnx_armnn"

    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        available = ort.get_available_providers()
        if "ArmNNExecutionProvider" not in available:
            raise RuntimeError(
                f"Arm NN execution provider not available. "
                f"Available providers: {', '.join(available)}. "
                f"Run on Arm hardware (Graviton, Raspberry Pi) with "
                f"onnxruntime built with ACL support."
            )

        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        buf = io.BytesIO()
        torch.onnx.export(model, dummy, buf, input_names=["input"], output_names=["output"])
        onnx_bytes = buf.read()

        session = ort.InferenceSession(
            onnx_bytes,
            providers=["ArmNNExecutionProvider", "CPUExecutionProvider"],
        )

        active = session.get_providers()
        print(f"    Active providers: {', '.join(active)}")

        return OnnxWrapper(session, onnx_bytes)