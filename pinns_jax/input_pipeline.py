import pathlib

import grain
import numpy as np


def lhs_sampler(low: float, high: float, n: int, d: int, rng: np.random.Generator):
    # 1. Create a grid of points
    grid = np.linspace(0, 1, n + 1)
    lower = grid[:-1]
    upper = grid[1:]

    # 2. Randomly sample within each interval
    points = rng.uniform(lower, upper, (d, n)).T

    # 3. Shuffle each dimension independently to maintain LHS property
    for i in range(d):
        rng.shuffle(points[:, i])

    return low + (high - low) * points


class CollocationPoints1D(grain.transforms.RandomMap):
    def __init__(self, config):
        self.config = config

        self.xmin, self.xmax = config.domain
        self.tmin, self.tmax = config.time_interval

        self.num_interior_points = config.num_interior_points
        self.num_boundary_points = config.num_boundary_points
        self.num_initial_points = config.num_initial_points

        if config.sampler == "lhs":
            self.sample_fn = lambda low, high, n, rng: lhs_sampler(low, high, n, 1, rng)
        elif config.sampler == "uniform":
            self.sample_fn = lambda low, high, n, rng: rng.uniform(
                low, high, size=(n, 1)
            )
        else:
            raise ValueError(f"Unsupported sampler: {config.sampler}")

    def random_map(self, element, rng: np.random.Generator):
        # Sample interior points
        x_interior = self.sample_fn(self.xmin, self.xmax, self.num_interior_points, rng)
        t_interior = self.sample_fn(self.tmin, self.tmax, self.num_interior_points, rng)

        # Sample boundary points
        x_boundary = np.concatenate(
            [
                np.full((self.num_boundary_points // 2, 1), self.xmin),
                np.full((self.num_boundary_points // 2, 1), self.xmax),
            ],
            axis=0,
        )
        t_boundary = np.concatenate(
            [
                self.sample_fn(
                    self.tmin, self.tmax, self.num_boundary_points // 2, rng
                ),
                self.sample_fn(
                    self.tmin, self.tmax, self.num_boundary_points // 2, rng
                ),
            ],
            axis=0,
        )

        # Sample initial points
        x_initial = self.sample_fn(self.xmin, self.xmax, self.num_initial_points, rng)
        t_initial = np.full((self.num_initial_points, 1), self.tmin)

        return {
            "interior": (x_interior, t_interior),
            "boundary": (x_boundary, t_boundary),
            "initial": (x_initial, t_initial),
        }


def get_datasets(config):
    collocation_points_sampler = CollocationPoints1D(config)
    train_ds = (
        grain.MapDataset.range(config.num_train_steps)
        .seed(config.seed)
        .random_map(collocation_points_sampler)
    )

    eval_numpy_ds = np.load(pathlib.Path(config.eval_data_path).resolve())
    eval_ds = {
        "inputs": (eval_numpy_ds[..., 1:2], eval_numpy_ds[..., 0:1]),
        "labels": eval_numpy_ds[..., -1],
    }

    return train_ds, eval_ds
