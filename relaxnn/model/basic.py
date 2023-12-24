import torch
import torch.nn as nn

device = torch.device("cuda:0")

class Net(nn.Module):
    def __init__(self, layer_sizes: list, activation: str, configuration: str) -> None:
        super().__init__()
        self._layer_sizes = layer_sizes
        if activation == "tanh":
            self._activation = nn.Tanh()
        self._configuration = configuration
        self.layers = nn.ModuleList(
            nn.Linear(*i) for i in zip(layer_sizes[:-1], layer_sizes[1:])
        )

    def forward(self, x):
        if self._configuration == "resnet":
            y = self._activation(self.layers[0](x))
            for layer in self.layers[1:-1]:
                y = self._activation(layer(y)) + y
            y = self.layers[-1](y)
        elif self._configuration == "DNN":
            for layer in self.layers[:-1]:
                x = self._activation(layer(x))
            y = self.layers[-1](x)
        return y
