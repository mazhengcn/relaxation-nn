import math

import torch
import torch.nn as nn

from pinns_utils_torch.runtime import DEVICE

device = DEVICE


class PDElossfn(torch.nn.MSELoss):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.square(input - target).mean(dim=0).sum()


class Net(nn.Module):
    def __init__(
        self,
        layer_sizes: list,
        activation: str,
        configuration: str,
        initialization: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        self._layer_sizes = layer_sizes
        if activation == "tanh":
            self._activation = nn.Tanh()
        elif activation == "relu":
            self._activation = nn.ReLU()
        else:
            raise ValueError("Unsupported activation {}".format(activation))
        self._activation_name = activation
        self._configuration = configuration
        self.layers = nn.ModuleList(
            nn.Linear(*i) for i in zip(layer_sizes[:-1], layer_sizes[1:])
        )
        self._initialize_layers(initialization)

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

    def _initialize_layers(self, initialization: str):
        if initialization == "pytorch_default":
            return

        # gain = nn.init.calculate_gain(self._activation_name)
        gain = 1.0
        for layer in self.layers:
            if initialization == "kaiming_uniform":
                if self._activation_name == "relu":
                    nn.init.kaiming_uniform_(
                        layer.weight,
                        mode="fan_in",
                        nonlinearity="relu",
                    )
                else:
                    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            elif initialization == "kaiming_normal":
                if self._activation_name != "relu":
                    raise ValueError(
                        "kaiming_normal is only supported with relu activation"
                    )
                nn.init.kaiming_normal_(
                    layer.weight,
                    mode="fan_in",
                    nonlinearity="relu",
                )
            elif initialization == "xavier_uniform":
                nn.init.xavier_uniform_(layer.weight, gain=gain)
            elif initialization == "xavier_normal":
                nn.init.xavier_normal_(layer.weight, gain=gain)
            elif initialization == "kaiming_uniform_tanh":
                nn.init.kaiming_uniform_(
                    layer.weight,
                    mode="fan_in",
                    nonlinearity=self._activation_name,
                )
            elif initialization == "kaiming_normal_tanh":
                nn.init.kaiming_normal_(
                    layer.weight,
                    mode="fan_in",
                    nonlinearity=self._activation_name,
                )
            elif initialization == "orthogonal":
                nn.init.orthogonal_(layer.weight, gain=gain)
            else:
                raise ValueError("Unsupported initialization {}".format(initialization))

            if layer.bias is None:
                continue
            if initialization == "kaiming_uniform" and self._activation_name != "relu":
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(layer.bias, -bound, bound)
            else:
                nn.init.zeros_(layer.bias)
