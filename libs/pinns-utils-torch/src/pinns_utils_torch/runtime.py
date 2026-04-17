import os

import torch


def _resolve_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")

    preferred_index = int(os.getenv("RELAXNN_DEFAULT_CUDA_DEVICE", "1"))
    cuda_count = torch.cuda.device_count()
    if 0 <= preferred_index < cuda_count:
        return torch.device(f"cuda:{preferred_index}")
    return torch.device("cuda:0")


DEVICE = _resolve_device()
