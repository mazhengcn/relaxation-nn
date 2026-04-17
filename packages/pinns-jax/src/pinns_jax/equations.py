from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import nnx

from .net import MLP, MLPConfig


class Solution(nnx.Module):
    def __init__(self, config: MLPConfig, rngs: nnx.Rngs):
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x: jax.Array, t: jax.Array):
        if x.ndim > 1 or t.ndim > 1:
            assert x.shape[:-1] == t.shape[:-1], "x and t must have the same batch size"
        inputs = jnp.concat([jnp.atleast_1d(x), jnp.atleast_1d(t)], axis=-1)
        outputs = self.mlp(inputs)
        return jnp.squeeze(outputs, axis=-1)  # Output shape: (batch_size,)


class BaseEquation(ABC):
    @abstractmethod
    def residual(self, u: Solution, x: jax.Array, t: jax.Array):
        raise NotImplementedError

    @abstractmethod
    def initial_condition(self, u: Solution, x: jax.Array, t: jax.Array):
        raise NotImplementedError

    @abstractmethod
    def boundary_condition(self, u: Solution, x: jax.Array, t: jax.Array):
        raise NotImplementedError

    @abstractmethod
    def loss(self, u: Solution, batch: dict[str, tuple[jax.Array, jax.Array]]):
        raise NotImplementedError


class Burgers1D(BaseEquation):
    def __init__(
        self,
        nu: float = 0,
        initial_condition=lambda x: -jnp.sin(jnp.pi * x),
        boundary_condition=lambda t: 0.0,
        loss_fn: str = "mse",
        weights=(1.0, 1.0, 1.0),
    ):
        self.nu = nu
        self.initial_condition_fn = initial_condition
        self.boundary_condition_fn = boundary_condition
        self.weights = weights

        if loss_fn == "mse":
            self.loss_fn = lambda x: jnp.mean(x**2)

    def residual(self, u: Solution, x: jax.Array, t: jax.Array):
        u_x_fn = jax.grad(u, argnums=0)
        u_t_fn = jax.grad(u, argnums=1)
        # u_x = u_x_fn(x, t)
        flux_x = jax.grad(lambda x, t: 0.5 * u(x, t) ** 2, argnums=0)(x, t)
        u_xx = jax.grad(lambda x, t: u_x_fn(x, t).squeeze(-1), argnums=0)(x, t)
        u_t = u_t_fn(x, t)
        return u_t + flux_x - self.nu * u_xx

    def initial_condition(self, u: Solution, x: jax.Array, t: jax.Array):
        return u(x, t) - self.initial_condition_fn(x)

    def boundary_condition(self, u: Solution, x: jax.Array, t: jax.Array):
        return u(x, t) - self.boundary_condition_fn(t)

    def loss(self, u: Solution, batch):
        res = jax.vmap(self.residual, in_axes=(None, 0, 0))(u, *batch["interior"])
        ic = jax.vmap(self.initial_condition, in_axes=(None, 0, 0))(
            u, *batch["initial"]
        )
        bc = jax.vmap(self.boundary_condition, in_axes=(None, 0, 0))(
            u, *batch["boundary"]
        )

        res_loss, ic_loss, bc_loss = jax.tree.map(self.loss_fn, (res, ic, bc))
        total_loss = (
            self.weights[0] * res_loss
            + self.weights[1] * ic_loss
            + self.weights[2] * bc_loss
        )
        return total_loss, {
            "res_loss": res_loss,
            "ic_loss": ic_loss,
            "bc_loss": bc_loss,
        }
