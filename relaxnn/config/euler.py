from pathlib import Path

from ml_collections import ConfigDict, config_dict

_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path=str(_REPO_ROOT / "data" / "clawpack_data" / "euler_laxtube.npy"),
        distribution="uniform",
        range_L=[0.0, -0.5],
        range_R=[0.16, 0.5],
        num_samples=[10000, 1000, 1000],
    )
    config.NetConfig = dict(
        layer_sizes=[
            [2, 128, 128, 128, 128, 1],
            [2, 128, 128, 128, 128, 1],
        ],
        configuration=["DNN", "DNN"],
        activation=["tanh", "tanh"],
        initialization=["xavier_uniform", "xavier_uniform"],
        ibc_type=["lax_tube", "lax_tube"],
        loss="MSE",
    )
    config.TrainConfig = dict(
        epochs=600000,
        ratio=[0.01, 1.0, 5.0, 5.0],
        optimizer="Adam",
        lr=1e-3,
        decay="CosineAnnealing",
        scheduler_every=1,
        cosine_eta_min=1e-6,
    )
    config.model = "euler_v3"
    config.train_mode = "train"
    config.torch_seed = config_dict.placeholder(int)
    config.output_root = str(
        _REPO_ROOT / "_output" / "relaxnn" / "euler_v3" / "lax_tube"
    )
    config.experiment_name = ""
    config.root_dir = ""
    config.timestamp = ""
    return config
