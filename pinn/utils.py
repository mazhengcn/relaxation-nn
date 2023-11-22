import numpy as np
import torch


def to_numpy(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.detach().cpu().numpy()
    elif isinstance(inputs, np.ndarray):
        return inputs
    else:
        raise TypeError(
            "Unknown type of input, expected torch.Tensor or "
            "np.ndarray, but got {}".format(type(input))
        )
