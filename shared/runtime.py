import os

import torch


def _resolve_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device("cuda:0")


DEVICE = _resolve_device()
