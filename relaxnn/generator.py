"""Generate the training data """

from numbers import Number
from pathlib import Path

import cartesian
import numpy as np
import torch
from ml_collections import ConfigDict
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
        self.xrange = torch.tensor([[low[1]], [high[1]]]).to("cuda:0")

    def rsample(self, sample_shape=torch.Size()):
        s = sample_shape[0]
        t = self.tsampler.rsample((s, 1))
        txy = cartesian.cartesian_prod(t, self.xrange).reshape(-1, 2)
        return txy


class Generator:
    def __init__(self, config: ConfigDict):
        self.config = config
        self.load_path = config.testdata_path
        self.intbatch = config.num_samples[0]
        self.icbatch = config.num_samples[1]
        self.bcbatch = config.num_samples[2]
        self.intlow = torch.tensor(config.range_L).to("cuda:0")
        self.inthigh = torch.tensor(config.range_R).to("cuda:0")
        self.iclow = torch.tensor(config.range_L).to("cuda:0")
        self.ichigh = torch.tensor([config.range_L[0], config.range_R[1]]).to("cuda:0")
        self.intsampler = Uniform(self.intlow, self.inthigh)
        self.icsampler = Uniform(self.iclow, self.ichigh)
        self.bcsampler = BCsampler(self.intlow, self.inthigh)

    def samples(self):
        intsamples = self.intsampler.rsample((self.intbatch,))
        icsamples = self.icsampler.rsample((self.icbatch,))
        bcsamples = self.bcsampler.rsample((self.bcbatch,))
        return intsamples, icsamples, bcsamples

    def load_testdata(self):
        data = np.load(self.load_path)
        x_test, q_test = data[:, 0:2], data[:, 2 : data.shape[1]]
        return x_test, q_test
