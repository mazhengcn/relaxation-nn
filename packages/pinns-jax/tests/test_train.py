import dataclasses
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
from flax import nnx
from orbax.checkpoint import v1 as ocp

from .equations import Burgers1D, Solution
from .net import MLPConfig
from .train import eval_step, init_or_restore, init_training_states, train_step


@dataclasses.dataclass
class TrainingConfig(MLPConfig):
    """Minimal config combining MLPConfig and optimizer settings for tests."""

    optimizer: str = "adam"
    scheduler: str = "constant"
    learning_rate: float = 1e-3


def make_config():
    return TrainingConfig(
        in_dims=2,
        out_dims=1,
        hidden_dims=16,
        num_hidden_layers=2,
        activation="tanh",
    )


class TestInitTrainingStates:
    def test_returns_model_and_optimizer(self):
        config = make_config()
        model, optimizer = init_training_states(config, jax.random.key(0))
        assert isinstance(model, Solution)
        assert isinstance(optimizer, nnx.Optimizer)

    def test_model_output_shape(self):
        config = make_config()
        model, _ = init_training_states(config, jax.random.key(0))
        out = model(jnp.array(0.5), jnp.array(0.1))
        assert out.shape == ()

    def test_optimizer_state_is_concrete(self):
        config = make_config()
        _, optimizer = init_training_states(config, jax.random.key(0))
        opt_state = nnx.state(optimizer, nnx.optimizer.OptState)
        leaves = jax.tree.leaves(nnx.to_pure_dict(opt_state))
        # concrete JAX arrays have a 'devices' attribute (unlike ShapeDtypeStruct)
        assert any(hasattr(leaf, "devices") for leaf in leaves)

    def test_same_key_gives_same_params(self):
        config = make_config()
        model1, _ = init_training_states(config, jax.random.key(42))
        model2, _ = init_training_states(config, jax.random.key(42))
        leaves1 = jax.tree.leaves(nnx.to_pure_dict(nnx.state(model1)))
        leaves2 = jax.tree.leaves(nnx.to_pure_dict(nnx.state(model2)))
        assert all(jnp.array_equal(a, b) for a, b in zip(leaves1, leaves2))

    def test_different_keys_give_different_params(self):
        config = make_config()
        model1, _ = init_training_states(config, jax.random.key(0))
        model2, _ = init_training_states(config, jax.random.key(1))
        leaves1 = jax.tree.leaves(nnx.to_pure_dict(nnx.state(model1)))
        leaves2 = jax.tree.leaves(nnx.to_pure_dict(nnx.state(model2)))
        assert any(not jnp.array_equal(a, b) for a, b in zip(leaves1, leaves2))


# ---------------------------------------------------------------------------
# Helpers for checkpoint tests
# ---------------------------------------------------------------------------


def _save_checkpoint(tmp_path, config, step=0):
    """Save a fresh training state as an orbax checkpoint, return the model."""
    model, optimizer = init_training_states(config, init_key=0)
    state = nnx.state({"params": model, "optimizer": optimizer})
    with ocp.training.Checkpointer(tmp_path) as ckptr:
        ckptr.save_pytree(step, state, force=True)
    return model


def _param_leaves(model):
    return jax.tree.leaves(nnx.to_pure_dict(nnx.state(model)))


class TestInitOrRestore:
    def test_fresh_init_returns_zero_step(self):
        config = make_config()
        mock_ckptr = MagicMock()
        mock_ckptr.latest = None

        model, optimizer, last_step = init_or_restore(
            mock_ckptr, None, config, init_key=0
        )

        assert last_step == 0
        assert isinstance(model, Solution)
        assert isinstance(optimizer, nnx.Optimizer)

    def test_fresh_init_model_produces_finite_output(self):
        config = make_config()
        mock_ckptr = MagicMock()
        mock_ckptr.latest = None

        model, _, _ = init_or_restore(mock_ckptr, None, config, init_key=0)
        out = model(jnp.array(0.5), jnp.array(0.1))
        assert jnp.isfinite(out)

    def test_restore_from_ckptr_latest_matches_saved_params(self, tmp_path):
        config = make_config()
        original_model = _save_checkpoint(tmp_path, config)

        with ocp.training.Checkpointer(tmp_path) as ckptr:
            model, optimizer, last_step = init_or_restore(
                ckptr, None, config, init_key=0
            )

        assert isinstance(model, Solution)
        assert isinstance(optimizer, nnx.Optimizer)
        assert all(
            jnp.array_equal(a, b)
            for a, b in zip(_param_leaves(original_model), _param_leaves(model))
        )

    def test_restore_from_ckptr_latest_returns_correct_step(self, tmp_path):
        config = make_config()
        # Manually advance step counter before saving so we can verify it's restored.
        model, optimizer = init_training_states(config, init_key=0)
        optimizer.step[...] = jnp.array(7, dtype=jnp.uint32)
        state = nnx.state({"params": model, "optimizer": optimizer})
        with ocp.training.Checkpointer(tmp_path) as ckptr:
            ckptr.save_pytree(0, state, force=True)

        with ocp.training.Checkpointer(tmp_path) as ckptr:
            _, _, last_step = init_or_restore(ckptr, None, config, init_key=0)

        assert last_step == 7

    def test_restore_from_ckpt_path_matches_saved_params(self, tmp_path):
        config = make_config()
        save_dir = tmp_path / "saved"
        original_model = _save_checkpoint(save_dir, config, step=0)

        # Determine the exact step directory orbax created
        step_dirs = sorted(save_dir.iterdir())
        ckpt_path = str(step_dirs[0])

        # Use a separate empty directory so ckptr.latest is None
        fresh_dir = tmp_path / "fresh"
        fresh_dir.mkdir()
        with ocp.training.Checkpointer(fresh_dir) as ckptr:
            model, optimizer, _ = init_or_restore(ckptr, ckpt_path, config, init_key=0)

        assert isinstance(model, Solution)
        assert all(
            jnp.array_equal(a, b)
            for a, b in zip(_param_leaves(original_model), _param_leaves(model))
        )


# ---------------------------------------------------------------------------
# Helpers for train_step / eval_step tests
# ---------------------------------------------------------------------------

N = 8  # small batch size for tests


def _make_batch():
    """Create a minimal batch matching the Burgers1D loss signature.

    The residual uses jax.grad so x and t must be scalars when vmapped;
    pass 1-D arrays of shape (N,) so each vmapped element is a scalar.
    """
    x = jnp.linspace(-1.0, 1.0, N)
    t = jnp.linspace(0.0, 1.0, N)
    return {
        "interior": (x, t),
        "initial": (x, jnp.zeros(N)),
        "boundary": (jnp.full(N, -1.0), t),
    }


def _make_train_metrics():
    return nnx.MultiMetric(
        total_loss=nnx.metrics.Average("total_loss"),
        res_loss=nnx.metrics.Average("res_loss"),
        bc_loss=nnx.metrics.Average("bc_loss"),
        ic_loss=nnx.metrics.Average("ic_loss"),
    )


def _make_eval_metrics():
    return nnx.MultiMetric(
        mse=nnx.metrics.Average("mse"),
        relative_mse=nnx.metrics.Average("relative_mse"),
    )


class TestTrainStep:
    def test_metrics_are_updated(self):
        config = make_config()
        model, optimizer = init_training_states(config, jax.random.key(0))
        metrics = _make_train_metrics()
        equation = Burgers1D()
        batch = _make_batch()

        train_step(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            equation=equation,
            batch=batch,
        )
        result = metrics.compute()

        assert set(result.keys()) == {"total_loss", "res_loss", "bc_loss", "ic_loss"}
        assert all(jnp.isfinite(v) for v in result.values())

    def test_losses_are_non_negative(self):
        config = make_config()
        model, optimizer = init_training_states(config, jax.random.key(0))
        metrics = _make_train_metrics()
        equation = Burgers1D()
        batch = _make_batch()

        train_step(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            equation=equation,
            batch=batch,
        )
        result = metrics.compute()

        assert all(v >= 0 for v in result.values())

    def test_params_change_after_step(self):
        config = make_config()
        model, optimizer = init_training_states(config, jax.random.key(0))
        params_before = jax.tree.leaves(nnx.to_pure_dict(nnx.state(model, nnx.Param)))
        metrics = _make_train_metrics()
        equation = Burgers1D()
        batch = _make_batch()

        train_step(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            equation=equation,
            batch=batch,
        )
        params_after = jax.tree.leaves(nnx.to_pure_dict(nnx.state(model, nnx.Param)))

        assert any(
            not jnp.array_equal(a, b) for a, b in zip(params_before, params_after)
        )

    def test_multiple_steps_accumulate_metrics(self):
        config = make_config()
        model, optimizer = init_training_states(config, jax.random.key(0))
        metrics = _make_train_metrics()
        equation = Burgers1D()
        batch = _make_batch()

        train_step(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            equation=equation,
            batch=batch,
        )
        train_step(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            equation=equation,
            batch=batch,
        )
        result = metrics.compute()

        # After two steps, averages should still be finite scalars
        assert all(jnp.isfinite(v) for v in result.values())


class TestEvalStep:
    def test_metrics_are_updated(self):
        config = make_config()
        model, _ = init_training_states(config, jax.random.key(0))
        metrics = _make_eval_metrics()
        x = jnp.linspace(-1.0, 1.0, N).reshape(N, 1)
        t = jnp.linspace(0.0, 1.0, N).reshape(N, 1)
        labels = model(x, t)
        data = {"inputs": (x, t), "labels": labels}

        eval_step(model, metrics, data)
        result = metrics.compute()

        assert set(result.keys()) == {"mse", "relative_mse"}
        assert all(jnp.isfinite(v) for v in result.values())

    def test_perfect_prediction_gives_zero_mse(self):
        config = make_config()
        model, _ = init_training_states(config, jax.random.key(0))
        metrics = _make_eval_metrics()
        x = jnp.linspace(-1.0, 1.0, N).reshape(N, 1)
        t = jnp.linspace(0.0, 1.0, N).reshape(N, 1)
        # Labels equal model output → MSE should be 0
        labels = model(x, t)
        data = {"inputs": (x, t), "labels": labels}

        eval_step(model, metrics, data)
        result = metrics.compute()

        assert jnp.allclose(result["mse"], 0.0, atol=1e-6)

    def test_mse_is_non_negative(self):
        config = make_config()
        model, _ = init_training_states(config, jax.random.key(0))
        metrics = _make_eval_metrics()
        x = jnp.linspace(-1.0, 1.0, N).reshape(N, 1)
        t = jnp.linspace(0.0, 1.0, N).reshape(N, 1)
        labels = jnp.ones(N)
        data = {"inputs": (x, t), "labels": labels}

        eval_step(model, metrics, data)
        result = metrics.compute()

        assert result["mse"] >= 0.0
        assert result["relative_mse"] >= 0.0

    def test_multiple_eval_steps_accumulate(self):
        config = make_config()
        model, _ = init_training_states(config, jax.random.key(0))
        metrics = _make_eval_metrics()
        x = jnp.linspace(-1.0, 1.0, N).reshape(N, 1)
        t = jnp.linspace(0.0, 1.0, N).reshape(N, 1)
        labels = jnp.ones(N)
        data = {"inputs": (x, t), "labels": labels}

        eval_step(model, metrics, data)
        eval_step(model, metrics, data)
        result = metrics.compute()

        assert all(jnp.isfinite(v) for v in result.values())
