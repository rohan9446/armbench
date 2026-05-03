import torch


def get_model_info(model):
    """Return parameter count and estimated model size in MB."""
    total_params = 0
    total_bytes = 0

    for param in model.parameters():
        total_params += param.numel()
        total_bytes += param.numel() * param.element_size()

    size_mb = round(total_bytes / (1024 * 1024), 2)

    return {
        "param_count": total_params,
        "model_size_mb": size_mb,
    }