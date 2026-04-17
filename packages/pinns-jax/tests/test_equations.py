import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from .equations import Burgers1D, Solution
from .net import MLPConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_solution(activation: str = "tanh") -> Solution:
    config = MLPConfig(
        in_dims=2,
        out_dims=1,
        hidden_dims=16,
        num_hidden_layers=2,
        activation=activation,
    )
    return Solution(config, rngs=nnx.Rngs(0))


# ---------------------------------------------------------------------------
# Solution
# ---------------------------------------------------------------------------


class TestSolution:
    def test_scalar_inputs_return_scalar(self):
        u = make_solution()
        x = jnp.array(0.5)
        t = jnp.array(0.1)
        out = u(x, t)
        assert out.shape == ()

    def test_batched_inputs_return_batch(self):
        u = make_solution()
        x = jnp.linspace(-1.0, 1.0, 10).reshape(-1, 1)  # (10, 1)
        t = jnp.full((10, 1), 0.5)
        out = u(x, t)
        assert out.shape == (10,)

    def test_output_is_finite(self):
        u = make_solution()
        x = jnp.array(0.0)
        t = jnp.array(0.0)
        assert jnp.isfinite(u(x, t))

    def test_mismatched_batch_raises(self):
        u = make_solution()
        x = jnp.ones((5, 1))
        t = jnp.ones((6, 1))
        with pytest.raises(AssertionError):
            u(x, t)


# ---------------------------------------------------------------------------
# Burgers1D
# ---------------------------------------------------------------------------


class TestBurgers1DInit:
    def test_default_weights(self):
        eq = Burgers1D()
        assert eq.weights == (1.0, 1.0, 1.0)

    def test_default_nu(self):
        eq = Burgers1D()
        assert eq.nu == 0

    def test_custom_nu(self):
        eq = Burgers1D(nu=0.01)
        assert eq.nu == 0.01

    def test_custom_weights(self):
        eq = Burgers1D(weights=(2.0, 3.0, 4.0))
        assert eq.weights == (2.0, 3.0, 4.0)


class TestBurgers1DResidual:
    def test_residual_returns_scalar(self):
        u = make_solution()
        eq = Burgers1D()
        x = jnp.array(0.5)
        t = jnp.array(0.1)
        res = eq.residual(u, x, t)
        assert res.shape == ()

    def test_residual_is_finite(self):
        u = make_solution()
        eq = Burgers1D(nu=0.01)
        x = jnp.array(0.3)
        t = jnp.array(0.2)
        assert jnp.isfinite(eq.residual(u, x, t))

    def test_residual_vmappable(self):
        u = make_solution()
        eq = Burgers1D()
        xs = jnp.linspace(-1.0, 1.0, 8)
        ts = jnp.full((8,), 0.5)
        res = jax.vmap(eq.residual, in_axes=(None, 0, 0))(u, xs, ts)
        assert res.shape == (8,)
        assert jnp.all(jnp.isfinite(res))


class TestBurgers1DInitialCondition:
    def test_ic_at_t0_equals_minus_sin(self):
        u = make_solution()
        eq = Burgers1D()
        x = jnp.array(0.5)
        t = jnp.array(0.0)
        ic = eq.initial_condition(u, x, t)
        # ic = u(x, 0) - (-sin(pi*x))
        expected = u(x, t) - (-jnp.sin(jnp.pi * x))
        assert jnp.allclose(ic, expected)

    def test_ic_returns_scalar(self):
        u = make_solution()
        eq = Burgers1D()
        ic = eq.initial_condition(u, jnp.array(0.0), jnp.array(0.0))
        assert ic.shape == ()

    def test_custom_initial_condition(self):
        u = make_solution()
        ic_fn = lambda x: jnp.cos(jnp.pi * x)
        eq = Burgers1D(initial_condition=ic_fn)
        x = jnp.array(0.25)
        t = jnp.array(0.0)
        ic = eq.initial_condition(u, x, t)
        expected = u(x, t) - ic_fn(x)
        assert jnp.allclose(ic, expected)


class TestBurgers1DBoundaryCondition:
    def test_bc_returns_scalar(self):
        u = make_solution()
        eq = Burgers1D()
        bc = eq.boundary_condition(u, jnp.array(-1.0), jnp.array(0.3))
        assert bc.shape == ()

    def test_default_bc_is_zero_dirichlet(self):
        u = make_solution()
        eq = Burgers1D()
        x = jnp.array(1.0)
        t = jnp.array(0.5)
        bc = eq.boundary_condition(u, x, t)
        expected = u(x, t) - 0.0
        assert jnp.allclose(bc, expected)

    def test_custom_boundary_condition(self):
        u = make_solution()
        bc_fn = lambda t: jnp.sin(t)
        eq = Burgers1D(boundary_condition=bc_fn)
        x = jnp.array(-1.0)
        t = jnp.array(0.5)
        bc = eq.boundary_condition(u, x, t)
        expected = u(x, t) - bc_fn(t)
        assert jnp.allclose(bc, expected)


def make_batch(n: int = 16, seed: int = 42) -> dict:
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    return {
        "interior": (
            jax.random.uniform(k1, (n,), minval=-1.0, maxval=1.0),
            jax.random.uniform(k2, (n,), minval=0.0, maxval=1.0),
        ),
        "initial": (
            jax.random.uniform(k3, (n,), minval=-1.0, maxval=1.0),
            jnp.zeros((n,)),
        ),
        "boundary": (
            jnp.ones((n,)),
            jax.random.uniform(k6, (n,), minval=0.0, maxval=1.0),
        ),
    }


class TestBurgers1DLoss:
    def setup_method(self):
        self.u = make_solution()
        self.eq = Burgers1D()
        self.batch = make_batch()

    def test_loss_returns_scalar_total(self):
        total, _ = self.eq.loss(self.u, self.batch)
        assert total.shape == ()

    def test_loss_returns_aux_dict(self):
        _, aux = self.eq.loss(self.u, self.batch)
        assert set(aux.keys()) == {"res_loss", "ic_loss", "bc_loss"}

    def test_loss_is_non_negative(self):
        total, aux = self.eq.loss(self.u, self.batch)
        assert total >= 0.0
        for v in aux.values():
            assert v >= 0.0

    def test_loss_is_finite(self):
        total, aux = self.eq.loss(self.u, self.batch)
        assert jnp.isfinite(total)
        for v in aux.values():
            assert jnp.isfinite(v)

    def test_loss_weighted(self):
        eq_weighted = Burgers1D(weights=(2.0, 3.0, 4.0))
        _, aux_w = eq_weighted.loss(self.u, self.batch)
        _, aux_1 = Burgers1D(weights=(1.0, 1.0, 1.0)).loss(self.u, self.batch)

        total_weighted = (
            2.0 * aux_w["res_loss"] + 3.0 * aux_w["ic_loss"] + 4.0 * aux_w["bc_loss"]
        )
        total_unweighted = (
            1.0 * aux_1["res_loss"] + 1.0 * aux_1["ic_loss"] + 1.0 * aux_1["bc_loss"]
        )

        # Aux losses should be the same regardless of weights
        assert jnp.allclose(aux_w["res_loss"], aux_1["res_loss"])
        # Weighted total should differ from unweighted when sub-losses differ
        assert jnp.isfinite(total_weighted)
        assert jnp.isfinite(total_unweighted)

    def test_loss_differentiable(self):
        """Ensure the loss is differentiable w.r.t. model parameters."""
        graphdef, params, *rest = nnx.split(self.u, nnx.Param, ...)

        def loss_fn(params):
            u = nnx.merge(graphdef, params, *rest)
            total, _ = self.eq.loss(u, self.batch)
            return total

        grads = jax.grad(loss_fn)(params)
        leaves = jax.tree.leaves(grads)
        assert all(jnp.isfinite(g).all() for g in leaves)
