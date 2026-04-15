from pathlib import Path

from ml_collections import ConfigDict, config_dict

_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path=str(_REPO_ROOT / "data" / "clawpack_data" / "swe_dam_break.npy"),
        distribution="uniform",
        range_L=[0.0, -1.5],
        range_R=[1.0, 1.5],
        num_samples=[10000, 1000, 1000],
    )
    config.NetConfig = dict(
        layer_sizes=[
            [2, 64, 64, 64, 64, 2],
            [2, 64, 64, 64, 64, 2],
        ],
        configuration=["DNN", "DNN"],
        activation=["tanh", "tanh"],
        ibc_type=["dam-break", "dam-break"],
        loss="MSE",
    )
    config.TrainConfig = dict(
        epochs=600000,
        ratio=[0.1, 1.0, 1.0, 1.0],
        optimizer="Adam",
        lr=1e-3,
        decay="CosineAnnealing",
        scheduler_every=1,
        cosine_eta_min=1e-6,
    )
    config.model = "swe_v1"
    config.train_mode = "train"
    config.torch_seed = config_dict.placeholder(int)
    config.output_root = str(
        _REPO_ROOT / "_output" / "relaxnn" / "swe_v1" / "dam-break"
    )
    config.experiment_name = ""
    config.root_dir = ""
    config.timestamp = ""
    return config
