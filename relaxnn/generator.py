"""Generate the training data """
from pathlib import Path

import numpy as np
from ml_collections import ConfigDict


class DataGenerator:
    def __init__(self, config: ConfigDict):
        assert config.range_L <= config.range_R
        self.sampler = np.random.default_rng(seed=config.seed)
        self.range_L = config.range_L  # List[float]
        self.range_R = config.range_R  # List[float]
        self.num_samples = config.num_samples  # List[int]
        self.sample = config.sample
        self.dim = len(self.range_L)

        self.load_path = Path(config.testdata_path)
        g = np.random.default_rng(seed=config.seed)
        if config.distribution == "uniform":
            self._sampler = g.uniform
        elif config.distribution == "normal":
            self._sampler = g.normal

    def get_iter(self):
        for _ in range(self.sample):
            range_int = [self.range_L, self.range_R, (self.num_samples[0], self.dim)]
            range_ic = [
                [self.range_L[0], self.range_L[1]],
                [self.range_L[0], self.range_R[1]],
                (self.num_samples[1], self.dim),
            ]
            range_bcl = [
                [self.range_L[0], self.range_L[1]],
                [self.range_R[0], self.range_L[1]],
                (self.num_samples[2] // 2, self.dim),
            ]
            range_bcr = [
                [self.range_L[0], self.range_R[1]],
                [self.range_R[0], self.range_R[1]],
                (self.num_samples[2] // 2, self.dim),
            ]
            yield {
                "interior": self._sampler(*range_int),
                "initial": self._sampler(*range_ic),
                "boundary": np.vstack(
                    [self._sampler(*range_bcl), self._sampler(*range_bcr)]
                ),
            }

    def load_testdata(self):
        data = np.load(self.load_path)
        x_test, q_test = data[:, 0 : self.dim], data[:, self.dim : data.shape[1]]
        return x_test, q_test

