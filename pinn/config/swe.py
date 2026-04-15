from pathlib import Path

from ml_collections import ConfigDict, config_dict

_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path=str(_REPO_ROOT / "data" / "clawpack_data" / "swe_2shock.npy"),
        sampling_strategy="monte_carlo",
        lhs_criterion="center",
        sampling_seed=None,
        eval_time_window=[],
        range_L=[0.0, -1.0],
        range_R=[1.0, 1.0],
        num_samples=[10000, 1000, 1000],
        interior_grid_shape=[10, 1000],
    )
    config.NetConfig = dict(
        layer_sizes=[2, 64, 64, 64, 64, 2],
        configuration="DNN",
        activation="tanh",
        ibc_type=["2shock", "2shock"],
        loss="MSE",
        initialization="xavier_uniform",
        gravity=1.0,
    )
    config.TrainConfig = dict(
        epochs=600000,
        loss_weights=dict(
            res_loss=1.0,
            u_ic=10.0,
            u_bc=10.0,
        ),
        history_terms=["res_loss", "u_ic", "u_bc"],
        optimizer="Adam",
        lr=1e-3,
        decay="CosineAnnealing",
        scheduler_every=1,
        cosine_eta_min=1e-6,
        history_every=1000,
        log_every=100,
        checkpoint_every=1000,
    )
    config.model = "swe"
    config.plot_label = "PINN"
    config.train_mode = "train"
    config.torch_seed = config_dict.placeholder(int)
    config.output_root = str(_REPO_ROOT / "_output" / "pinn" / "swe" / "2shock")
    config.experiment_name = ""
    config.root_dir = ""
    config.timestamp = ""
    return config
