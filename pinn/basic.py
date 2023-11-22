import math

import torch
import torch.nn as nn
import torch.nn.init as init

device = torch.device("cuda:0")


class Linear(nn.Linear):
    """xavier uniform initializer"""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features)

    def reset_parameters(self) -> None:
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.weight)
        bound = math.sqrt(6 / (fan_in + fan_out))
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)


class Net(nn.Module):
    def __init__(self, layer_sizes: list, activation: str, configuration: str) -> None:
        super().__init__()
        self._layer_sizes = layer_sizes
        if activation == "tanh":
            self._activation = nn.Tanh()
        elif activation == "relu":
            self._activation = nn.ReLU()
        self._configuration = configuration
        self.layers = nn.ModuleList(
            nn.Linear(*i) for i in zip(layer_sizes[:-1], layer_sizes[1:])
        )

    def forward(self, x):
        if self._configuration == "resnet10":
            y = self._activation(self.layers[0](x))
            block1 = self._activation(self.layers[1](y))
            y = self._activation(self.layers[2](block1))
            for layer in self.layers[3:6]:
                y = self._activation(layer(y))
            block2 = y + block1
            y = self._activation(self.layers[6](block2))
            for layer in self.layers[7:-2]:
                y = self._activation(layer(y))
            y = self._activation(self.layers[-2](y)) + block2
            y = self.layers[-1](y)
        elif self._configuration == "DNN":
            for layer in self.layers[:-1]:
                x = self._activation(layer(x))
            y = self.layers[-1](x)
        return y
