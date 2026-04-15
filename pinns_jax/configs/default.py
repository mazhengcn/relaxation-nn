"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # -------------------------------------------------------------------------
    # PDE / equation parameters
    # -------------------------------------------------------------------------
    # Viscosity coefficient for the Burgers equation (0 = inviscid).
    config.nu = 0.0

    # Spatial domain [xmin, xmax].
    config.domain = (-1.0, 1.0)

    # Time interval [tmin, tmax].
    config.time_interval = (0.0, 1.0)

    # Loss weights: (residual, initial condition, boundary condition).
    config.loss_weights = (1.0, 1.0, 1.0)

    # -------------------------------------------------------------------------
    # Collocation / data sampling
    # -------------------------------------------------------------------------
    config.num_interior_points = 2000
    config.num_initial_points = 200
    config.num_boundary_points = 200

    # Sampler type: "lhs" (Latin Hypercube) or "uniform".
    config.sampler = "lhs"

    # Path to the .npy evaluation data file.
    config.eval_data_path = "data/clawpack_data/burgers_sine.npy"

    # -------------------------------------------------------------------------
    # Network architecture (MLP)
    # -------------------------------------------------------------------------
    # Input dims: 2 (x, t); output dims: 1 (u).
    config.in_dims = 2
    config.out_dims = 1
    config.hidden_dims = 64
    config.num_hidden_layers = 4
    config.activation = "tanh"
    config.init_fn = "glorot_normal"

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------
    config.optimizer = "adam"
    config.learning_rate = 1e-3

    # Scheduler: "exponential_decay", "cosine_decay", or "constant".
    config.scheduler = "exponential_decay"
    config.scheduler_transition_steps = 5_000
    config.scheduler_decay_rate = 0.9
    config.scheduler_staircase = False
    # Used only when scheduler == "cosine_decay".
    config.scheduler_decay_steps = 50_000

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    config.num_train_steps = 50_000

    # Frequency of evaluation during training.
    config.eval_every_steps = 1_000

    # Whether to save model checkpoints.
    config.save_checkpoints = True

    # Save a checkpoint every these number of steps.
    config.checkpoint_every_steps = 10_000

    # Optional path to restore a specific checkpoint (empty string = disabled).
    config.restore_checkpoint = ""

    # Integer for PRNG random seed.
    config.seed = 0

    return config
