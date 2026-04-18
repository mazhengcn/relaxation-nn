from pathlib import Path

from ml_collections import ConfigDict, config_dict

_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path=str(
            _REPO_ROOT / "data" / "clawpack_data" / "burgers_riemann.npy"
        ),
        sampling_strategy="monte_carlo",
        range_L=[0.0, -0.6],
        range_R=[1.0, 0.6],
        num_samples=[10000, 1000, 1000],
    )
    config.NetConfig = dict(
        layer_sizes=[
            [2, 64, 64, 64, 64, 1],
            [2, 64, 64, 64, 64, 1],
        ],
        configuration=["DNN", "DNN"],
        activation=["tanh", "tanh"],
        initialization=["xavier_uniform", "xavier_uniform"],
        ibc_type=["riemann", "riemann"],
        loss="MSE",
    )
    config.TrainConfig = dict(
        epochs=300000,
        ratio=[1.0, 10.0, 10.0, 10.0],
        int_weights=[1.0, 1.0],
        history_terms=["res_loss", "flux_loss", "u_ic", "F_ic", "u_bc", "F_bc"],
        optimizer="Adam",
        lr=1e-3,
        decay="CosineAnnealing",
        scheduler_every=1,
        cosine_eta_min=1e-6,
        history_every=1000,
        log_every=100,
        checkpoint_every=1000,
    )
    config.model = "burgers"
    config.plot_label = "RelaxNN"
    config.train_mode = "train"
    config.torch_seed = config_dict.placeholder(int)
    # By default, runs are saved to output_root/experiment_name/timestamp.
    config.output_root = str(_REPO_ROOT / "_output" / "relaxnn" / "burgers" / "riemann")
    config.experiment_name = (
        "adam_cosine_eta1e6_300000_mc_10000_1000_1000_"
        "tanh_xavier_uniform_w1_10_10_10n64"
    )
    config.root_dir = ""
    config.timestamp = ""
    return config
