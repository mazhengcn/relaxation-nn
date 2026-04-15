import dataclasses

from flax import nnx


@dataclasses.dataclass
class MLPConfig:
    in_dims: int
    out_dims: int
    hidden_dims: int
    num_hidden_layers: int
    activation: str = "relu"
    init_fn: str = "glorot_uniform"


def get_init_fn(init_fn: str):
    if init_fn == "glorot_uniform":
        return nnx.initializers.glorot_uniform()
    elif init_fn == "glorot_normal":
        return nnx.initializers.glorot_normal()
    elif init_fn == "he_uniform":
        return nnx.initializers.he_uniform()
    elif init_fn == "he_normal":
        return nnx.initializers.he_normal()
    else:
        raise ValueError(f"Unsupported init_fn: {init_fn}")


def get_activation_fn(activation: str):
    if activation == "relu":
        return nnx.relu
    elif activation == "tanh":
        return nnx.tanh
    elif activation == "sigmoid":
        return nnx.sigmoid
    else:
        raise ValueError(f"Unsupported activation: {activation}")


class MLP(nnx.Module):
    def __init__(self, config: MLPConfig, *, rngs: nnx.Rngs):

        self.activation_fn = get_activation_fn(config.activation)
        self.init_fn = get_init_fn(config.init_fn)

        self.layers = nnx.List(
            [
                nnx.Linear(
                    config.in_dims,
                    config.hidden_dims,
                    kernel_init=self.init_fn,
                    rngs=rngs,
                )
            ]
        )
        for _ in range(config.num_hidden_layers - 1):
            self.layers.append(
                nnx.Linear(
                    config.hidden_dims,
                    config.hidden_dims,
                    kernel_init=self.init_fn,
                    rngs=rngs,
                )
            )
        self.layers.append(
            nnx.Linear(
                config.hidden_dims,
                config.out_dims,
                kernel_init=self.init_fn,
                use_bias=False,
                rngs=rngs,
            )
        )

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = self.activation_fn(layer(x))
            else:
                x = layer(x)
        return x
