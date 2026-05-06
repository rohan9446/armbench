import os
import tempfile
import torch


def get_model_info(model):
    """Return parameter count, weight memory, and disk size."""
    # Handle ONNX wrapper
    if hasattr(model, "onnx_size_mb"):
        size = model.onnx_size_mb()
        return {
            "param_count": 0,
            "model_size_mb": size,
            "disk_size_mb": size,
        }

    total_params = 0
    total_bytes = 0
    for param in model.parameters():
        total_params += param.numel()
        total_bytes += param.numel() * param.element_size()

    tmp = os.path.join(tempfile.gettempdir(), "armbench_tmp.pt")
    torch.save(model.state_dict(), tmp)
    disk_mb = round(os.path.getsize(tmp) / (1024 * 1024), 2)
    os.remove(tmp)

    return {
        "param_count": total_params,
        "model_size_mb": round(total_bytes / (1024 * 1024), 2),
        "disk_size_mb": disk_mb,
    }