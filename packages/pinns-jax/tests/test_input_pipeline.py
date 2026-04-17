import types

import numpy as np
import pytest

from .input_pipeline import (
    CollocationPoints1D,
    get_datasets,
    lhs_sampler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_config(**kwargs):
    """Return a SimpleNamespace config with sensible defaults for CollocationPoints1D."""
    defaults = dict(
        domain=(0.0, 1.0),
        time_interval=(0.0, 2.0),
        num_interior_points=10,
        num_boundary_points=6,
        num_initial_points=3,
        sampler="lhs",
    )
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# lhs_sampler
# ---------------------------------------------------------------------------


class TestLhsSampler:
    def test_output_shape_1d(self):
        pts = lhs_sampler(0.0, 1.0, n=20, d=1, rng=make_rng())
        assert pts.shape == (20, 1)

    def test_output_shape_2d(self):
        pts = lhs_sampler(0.0, 1.0, n=15, d=2, rng=make_rng())
        assert pts.shape == (15, 2)

    def test_values_within_bounds(self):
        low, high = -3.0, 5.0
        pts = lhs_sampler(low, high, n=50, d=3, rng=make_rng())
        assert np.all(pts >= low)
        assert np.all(pts <= high)

    def test_values_within_bounds_custom_range(self):
        low, high = 0.5, 2.5
        pts = lhs_sampler(low, high, n=30, d=1, rng=make_rng())
        assert np.all(pts >= low)
        assert np.all(pts <= high)

    def test_lhs_stratification(self):
        """Each stratum [k/n, (k+1)/n] must contain exactly one sample per dimension."""
        n = 20
        pts = lhs_sampler(0.0, 1.0, n=n, d=2, rng=make_rng())
        for dim in range(2):
            col = pts[:, dim]
            strata = (col * n).astype(int)
            assert len(np.unique(strata)) == n, (
                f"Dimension {dim} does not satisfy LHS stratification"
            )

    def test_different_seeds_give_different_results(self):
        pts1 = lhs_sampler(0.0, 1.0, n=10, d=1, rng=make_rng(0))
        pts2 = lhs_sampler(0.0, 1.0, n=10, d=1, rng=make_rng(99))
        assert not np.allclose(pts1, pts2)

    def test_same_seed_gives_same_results(self):
        pts1 = lhs_sampler(0.0, 1.0, n=10, d=1, rng=make_rng(7))
        pts2 = lhs_sampler(0.0, 1.0, n=10, d=1, rng=make_rng(7))
        np.testing.assert_array_equal(pts1, pts2)


# ---------------------------------------------------------------------------
# CollocationPoints1D – construction
# ---------------------------------------------------------------------------


class TestCollocationPoints1DInit:
    def test_lhs_sampler_accepted(self):
        cfg = make_config(sampler="lhs")
        sampler = CollocationPoints1D(cfg)
        assert sampler.sample_fn is not None

    def test_uniform_sampler_accepted(self):
        cfg = make_config(sampler="uniform")
        sampler = CollocationPoints1D(cfg)
        assert sampler.sample_fn is not None

    def test_invalid_sampler_raises(self):
        cfg = make_config(sampler="sobol")
        with pytest.raises(ValueError, match="Unsupported sampler"):
            CollocationPoints1D(cfg)

    def test_domain_and_time_stored(self):
        cfg = make_config(domain=(-2.0, 3.0), time_interval=(1.0, 5.0))
        s = CollocationPoints1D(cfg)
        assert s.xmin == -2.0
        assert s.xmax == 3.0
        assert s.tmin == 1.0
        assert s.tmax == 5.0


# ---------------------------------------------------------------------------
# CollocationPoints1D – random_map (lhs)
# ---------------------------------------------------------------------------


class TestCollocationPoints1DRandomMapLhs:
    @pytest.fixture
    def sampler(self):
        cfg = make_config(
            domain=(0.0, 1.0),
            time_interval=(0.0, 2.0),
            num_interior_points=16,
            num_boundary_points=8,
            num_initial_points=4,
            sampler="lhs",
        )
        return CollocationPoints1D(cfg), cfg

    def test_returns_required_keys(self, sampler):
        s, _ = sampler
        result = s.random_map(None, make_rng())
        assert set(result.keys()) == {"interior", "boundary", "initial"}

    def test_interior_shape(self, sampler):
        s, cfg = sampler
        x, t = s.random_map(None, make_rng())["interior"]
        assert x.shape == (cfg.num_interior_points, 1)
        assert t.shape == (cfg.num_interior_points, 1)

    def test_interior_bounds(self, sampler):
        s, cfg = sampler
        x, t = s.random_map(None, make_rng())["interior"]
        assert np.all(x >= cfg.domain[0]) and np.all(x <= cfg.domain[1])
        assert np.all(t >= cfg.time_interval[0]) and np.all(t <= cfg.time_interval[1])

    def test_boundary_shape(self, sampler):
        s, cfg = sampler
        x_b, t_b = s.random_map(None, make_rng())["boundary"]
        half = cfg.num_boundary_points // 2
        assert x_b.shape == (half * 2, 1)
        assert t_b.shape == (half * 2, 1)

    def test_boundary_x_values(self, sampler):
        s, cfg = sampler
        x_b, _ = s.random_map(None, make_rng())["boundary"]
        half = cfg.num_boundary_points // 2
        # First half must equal xmin, second half must equal xmax
        np.testing.assert_array_equal(x_b[:half], cfg.domain[0])
        np.testing.assert_array_equal(x_b[half:], cfg.domain[1])

    def test_boundary_t_within_time_interval(self, sampler):
        s, cfg = sampler
        _, t_b = s.random_map(None, make_rng())["boundary"]
        assert np.all(t_b >= cfg.time_interval[0])
        assert np.all(t_b <= cfg.time_interval[1])

    def test_initial_shape(self, sampler):
        s, cfg = sampler
        x_i, t_i = s.random_map(None, make_rng())["initial"]
        assert x_i.shape == (cfg.num_initial_points, 1)
        assert t_i.shape == (cfg.num_initial_points, 1)

    def test_initial_t_equals_tmin(self, sampler):
        s, cfg = sampler
        _, t_i = s.random_map(None, make_rng())["initial"]
        np.testing.assert_array_equal(t_i, cfg.time_interval[0])

    def test_initial_x_within_domain(self, sampler):
        s, cfg = sampler
        x_i, _ = s.random_map(None, make_rng())["initial"]
        assert np.all(x_i >= cfg.domain[0]) and np.all(x_i <= cfg.domain[1])

    def test_different_calls_give_different_points(self, sampler):
        s, _ = sampler
        r1 = s.random_map(None, make_rng(0))["interior"][0]
        r2 = s.random_map(None, make_rng(1))["interior"][0]
        assert not np.allclose(r1, r2)


# ---------------------------------------------------------------------------
# CollocationPoints1D – random_map (uniform)
# ---------------------------------------------------------------------------


class TestCollocationPoints1DRandomMapUniform:
    @pytest.fixture
    def sampler(self):
        cfg = make_config(
            domain=(0.0, 1.0),
            time_interval=(0.0, 2.0),
            num_interior_points=12,
            num_boundary_points=6,
            num_initial_points=3,
            sampler="uniform",
        )
        return CollocationPoints1D(cfg), cfg

    def test_interior_shape(self, sampler):
        s, cfg = sampler
        x, t = s.random_map(None, make_rng())["interior"]
        assert x.shape == (cfg.num_interior_points, 1)
        assert t.shape == (cfg.num_interior_points, 1)

    def test_interior_bounds(self, sampler):
        s, cfg = sampler
        x, t = s.random_map(None, make_rng())["interior"]
        assert np.all(x >= cfg.domain[0]) and np.all(x <= cfg.domain[1])
        assert np.all(t >= cfg.time_interval[0]) and np.all(t <= cfg.time_interval[1])

    def test_initial_t_equals_tmin(self, sampler):
        s, cfg = sampler
        _, t_i = s.random_map(None, make_rng())["initial"]
        np.testing.assert_array_equal(t_i, cfg.time_interval[0])


# ---------------------------------------------------------------------------
# get_datasets
# ---------------------------------------------------------------------------


class TestGetDatasets:
    @pytest.fixture
    def eval_file(self, tmp_path):
        """Create a small numpy eval file: columns are [t, x, u]."""
        rng = make_rng()
        data = rng.random((48, 3)).astype(np.float32)
        path = tmp_path / "eval.npy"
        np.save(path, data)
        return path, data

    @pytest.fixture
    def config(self, eval_file):
        path, _ = eval_file
        return make_config(
            domain=(0.0, 1.0),
            time_interval=(0.0, 2.0),
            num_interior_points=10,
            num_boundary_points=6,
            num_initial_points=3,
            sampler="lhs",
            num_train_steps=4,
            seed=0,
            eval_data_path=str(path),
        )

    def test_returns_two_objects(self, config):
        result = get_datasets(config)
        assert len(result) == 2

    def test_train_ds_has_correct_length(self, config):
        train_ds, _ = get_datasets(config)
        assert len(train_ds) == config.num_train_steps

    def test_train_ds_element_keys(self, config):
        train_ds, _ = get_datasets(config)
        element = train_ds[0]
        assert set(element.keys()) == {"interior", "boundary", "initial"}

    def test_eval_ds_has_required_keys(self, config):
        _, eval_ds = get_datasets(config)
        assert set(eval_ds.keys()) == {"inputs", "labels"}

    def test_eval_ds_inputs_are_tuple_of_two(self, config):
        _, eval_ds = get_datasets(config)
        inputs = eval_ds["inputs"]
        assert len(inputs) == 2

    def test_eval_ds_inputs_shape(self, config, eval_file):
        _, data = eval_file
        _, eval_ds = get_datasets(config)
        x, t = eval_ds["inputs"]
        assert x.shape == (data.shape[0], 1)
        assert t.shape == (data.shape[0], 1)

    def test_eval_ds_labels_shape(self, config, eval_file):
        _, data = eval_file
        _, eval_ds = get_datasets(config)
        assert eval_ds["labels"].shape == (data.shape[0],)

    def test_eval_ds_inputs_match_file_columns(self, config, eval_file):
        path, data = eval_file
        _, eval_ds = get_datasets(config)
        x, t = eval_ds["inputs"]
        # column 1 -> x input, column 0 -> t input, last column -> label
        np.testing.assert_array_equal(x, data[..., 1:2])
        np.testing.assert_array_equal(t, data[..., 0:1])

    def test_eval_ds_labels_match_file_last_column(self, config, eval_file):
        path, data = eval_file
        _, eval_ds = get_datasets(config)
        np.testing.assert_array_equal(eval_ds["labels"], data[..., -1])
