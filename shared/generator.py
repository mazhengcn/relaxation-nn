"""Generate the training data."""

from numbers import Number

import numpy as np
import torch
from ml_collections import ConfigDict
from smt.sampling_methods import LHS
from shared import cartesian
from shared.runtime import DEVICE
from torch import nan
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all


class Uniform(torch.distributions.Distribution):
    arg_constraints = {
        "low": constraints.dependent(is_discrete=False, event_dim=0),
        "high": constraints.dependent(is_discrete=False, event_dim=0),
    }
    has_rsample = True

    @property
    def mean(self):
        return (self.high + self.low) / 2

    @property
    def mode(self):
        return nan * self.high

    @property
    def stddev(self):
        return (self.high - self.low) / 12**0.5

    @property
    def variance(self):
        return (self.high - self.low).pow(2) / 12

    def __init__(self, low, high, validate_args=None):
        self.low, self.high = broadcast_all(low, high)

        if isinstance(low, Number) and isinstance(high, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.low.size()
        super().__init__(batch_shape, validate_args=validate_args)

        if self._validate_args and not torch.le(self.low, self.high).all():
            raise ValueError("Uniform is not defined when low> high")

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.low.dtype, device=self.low.device)
        return self.low + rand * (self.high - self.low)


class BCsampler:
    def __init__(self, low, high):
        self.tlow = low[0]
        self.thigh = high[0]
        self.tsampler = Uniform(self.tlow, self.thigh)
        self.xrange = torch.tensor([[low[1]], [high[1]]]).to(DEVICE)

    def rsample(self, sample_shape=torch.Size()):
        total_samples = sample_shape[0]
        if total_samples % 2 != 0:
            raise ValueError("Boundary batch size must be even")

        t = self.tsampler.rsample((total_samples // 2, 1))
        txy = cartesian.cartesian_prod(t, self.xrange).reshape(-1, 2)
        return txy


class LHSSampler:
    def __init__(self, low, high, criterion, seed=None):
        xlimits = np.stack([low.detach().cpu().numpy(), high.detach().cpu().numpy()], axis=-1)
        self.sampler = LHS(xlimits=xlimits, criterion=criterion, seed=seed)

    def rsample(self, sample_shape=torch.Size()):
        num_samples = sample_shape[0]
        samples = self.sampler(num_samples)
        return torch.tensor(samples, dtype=torch.float32, device=DEVICE)


class LHSBoundarySampler:
    def __init__(self, low, high, criterion, seed=None):
        self.xrange = torch.tensor([[low[1]], [high[1]]], dtype=torch.float32, device=DEVICE)
        xlimits = np.array([[low[0].item(), high[0].item()]], dtype=np.float64)
        self.tsampler = LHS(xlimits=xlimits, criterion=criterion, seed=seed)

    def rsample(self, sample_shape=torch.Size()):
        total_samples = sample_shape[0]
        if total_samples % 2 != 0:
            raise ValueError("Boundary batch size must be even")

        t = self.tsampler(total_samples // 2)
        t = torch.tensor(t, dtype=torch.float32, device=DEVICE)
        return cartesian.cartesian_prod(t, self.xrange).reshape(-1, 2)


class Generator:
    def __init__(self, config: ConfigDict):
        self.config = config
        self.load_path = config.testdata_path
        self.intbatch = config.num_samples[0]
        self.icbatch = config.num_samples[1]
        self.bcbatch = config.num_samples[2]
        self.intlow = torch.tensor(config.range_L).to(DEVICE)
        self.inthigh = torch.tensor(config.range_R).to(DEVICE)
        self.iclow = torch.tensor(config.range_L).to(DEVICE)
        self.ichigh = torch.tensor([config.range_L[0], config.range_R[1]]).to(DEVICE)
        self.sampling_strategy = getattr(config, "sampling_strategy", "monte_carlo")
        self.lhs_criterion = getattr(config, "lhs_criterion", None)
        self.sampling_seed = getattr(config, "sampling_seed", None)
        self.eval_time_window = tuple(getattr(config, "eval_time_window", ()))
        self.interior_grid_shape = tuple(getattr(config, "interior_grid_shape", ()))
        self._fixed_samples = None
        self._validate_sampling_strategy(self.sampling_strategy, "sampling_strategy")
        self._validate_boundary_batch_size()
        self.intsampler, self.icsampler, self.bcsampler = self._build_samplers()
        if self.sampling_strategy == "fixed_grid":
            self._fixed_samples = self._build_fixed_samples()

    def samples(self):
        if self.sampling_strategy == "fixed_grid":
            return tuple(samples.clone().detach() for samples in self._fixed_samples)

        intsamples = self.intsampler.rsample((self.intbatch,))
        icsamples = self.icsampler.rsample((self.icbatch,))
        bcsamples = self.bcsampler.rsample((self.bcbatch,))
        return intsamples, icsamples, bcsamples

    def export_reference_samples(self, save_path):
        if self._fixed_samples is None:
            return

        intsamples, icsamples, bcsamples = self._fixed_samples
        np.savez(
            save_path,
            interior=intsamples.detach().cpu().numpy(),
            initial=icsamples.detach().cpu().numpy(),
            boundary=bcsamples.detach().cpu().numpy(),
        )

    def _build_fixed_samples(self):
        intsamples = self._build_interior_grid()
        icsamples = self._build_initial_grid()
        bcsamples = self._build_boundary_grid()
        return intsamples, icsamples, bcsamples

    def _build_samplers(self):
        if self.sampling_strategy == "monte_carlo":
            return (
                Uniform(self.intlow, self.inthigh),
                Uniform(self.iclow, self.ichigh),
                BCsampler(self.intlow, self.inthigh),
            )

        if self.sampling_strategy == "lhs":
            if self.lhs_criterion is None:
                raise ValueError(
                    "DataConfig.lhs_criterion must be provided when sampling_strategy='lhs'"
                )
            interior_seed, initial_seed, boundary_seed = self._spawn_lhs_generators()
            return (
                LHSSampler(
                    self.intlow,
                    self.inthigh,
                    criterion=self.lhs_criterion,
                    seed=interior_seed,
                ),
                LHSSampler(
                    self.iclow,
                    self.ichigh,
                    criterion=self.lhs_criterion,
                    seed=initial_seed,
                ),
                LHSBoundarySampler(
                    self.intlow,
                    self.inthigh,
                    criterion=self.lhs_criterion,
                    seed=boundary_seed,
                ),
            )

        if self.sampling_strategy == "fixed_grid":
            return (
                Uniform(self.intlow, self.inthigh),
                Uniform(self.iclow, self.ichigh),
                BCsampler(self.intlow, self.inthigh),
            )

        raise ValueError("Unknown sampling strategy {}".format(self.sampling_strategy))

    def _build_interior_grid(self):
        nt, nx = self._resolve_interior_grid_shape()
        tgrid = self._linspace_column(self.intlow[0], self.inthigh[0], nt)
        xgrid = self._linspace_column(self.intlow[1], self.inthigh[1], nx)
        return cartesian.cartesian_prod(tgrid, xgrid).reshape(-1, 2)

    def _build_initial_grid(self):
        tgrid = torch.full(
            (self.icbatch, 1),
            self.iclow[0].item(),
            device=DEVICE,
            dtype=torch.float32,
        )
        xgrid = self._linspace_column(self.iclow[1], self.ichigh[1], self.icbatch)
        return torch.cat([tgrid, xgrid], dim=-1)

    def _build_boundary_grid(self):
        tgrid = self._linspace_column(
            self.intlow[0], self.inthigh[0], self.bcbatch // 2
        )
        return cartesian.cartesian_prod(tgrid, self.bcsampler.xrange).reshape(-1, 2)

    def _spawn_lhs_generators(self):
        if self.sampling_seed is None:
            return None, None, None

        seed_sequence = np.random.SeedSequence(int(self.sampling_seed))
        child_sequences = seed_sequence.spawn(3)
        return [np.random.default_rng(child) for child in child_sequences]

    def _validate_boundary_batch_size(self):
        if self.bcbatch % 2 != 0:
            raise ValueError("Boundary batch size must be even")

    def _resolve_interior_grid_shape(self):
        if self.interior_grid_shape:
            if len(self.interior_grid_shape) != 2:
                raise ValueError("interior_grid_shape must have exactly two entries")
            nt, nx = self.interior_grid_shape
        else:
            side = int(round(self.intbatch**0.5))
            nt, nx = side, side

        if nt * nx != self.intbatch:
            raise ValueError(
                "Interior grid shape {} does not match {} collocation points".format(
                    (nt, nx), self.intbatch
                )
            )
        return nt, nx

    @staticmethod
    def _linspace_column(low, high, steps):
        return torch.linspace(
            low.item(), high.item(), steps, device=DEVICE, dtype=torch.float32
        ).reshape(-1, 1)

    @staticmethod
    def _validate_sampling_strategy(strategy, name):
        if strategy not in {"monte_carlo", "lhs", "fixed_grid"}:
            raise ValueError(
                "Unknown {} {}, expected 'monte_carlo', 'lhs', or 'fixed_grid'".format(
                    name, strategy
                )
            )

    def load_testdata(self):
        data = np.load(self.load_path)
        x_test, q_test = data[:, 0:2], data[:, 2 : data.shape[1]]
        if self.eval_time_window:
            if len(self.eval_time_window) != 2:
                raise ValueError("eval_time_window must have exactly two entries")
            t_low, t_high = self.eval_time_window
            # Numerical grids often store values like 0.6000000000000001, so use
            # a small tolerance when filtering by time window.
            tol = 1e-12 * max(1.0, abs(float(t_low)), abs(float(t_high)))
            mask = (x_test[:, 0] >= t_low - tol) & (x_test[:, 0] <= t_high + tol)
            x_test = x_test[mask]
            q_test = q_test[mask]
        return x_test, q_test
