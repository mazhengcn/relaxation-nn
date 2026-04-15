from pathlib import Path

from ml_collections import ConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path=str(_REPO_ROOT / "data" / "clawpack_data" / "burgers_sine.npy"),
        sampling_strategy="monte_carlo",
        range_L=[0.0, -1.0],
        range_R=[1.0, 1.0],
        num_samples=[2500, 100, 100],
    )
    config.NetConfig = dict(
        layer_sizes=[2, 20, 20, 20, 20, 1],
        configuration="DNN",
        activation="tanh",
        ibc_type=["sine", "sine"],
        loss="MSE",
        initialization="xavier_uniform",
    )
    config.TrainConfig = dict(
        epochs=30000,
        loss_weights=dict(
            res_loss=1.0,
            u_ic=1.0,
            u_bc=1.0,
        ),
        history_terms=["res_loss", "u_ic", "u_bc"],
        optimizer="LBFGS",
        lr=0.1,
        lbfgs_max_iter=1,
        lbfgs_history_size=100,
        lbfgs_line_search_fn="strong_wolfe",
        history_every=1,
        log_every=100,
        checkpoint_every=1000,
    )
    config.model = "burgers_viscid"
    config.plot_label = "PINN"
    config.train_mode = "train"
    config.torch_seed = 42
    config.root_dir = ""
    config.timestamp = ""
    return config
