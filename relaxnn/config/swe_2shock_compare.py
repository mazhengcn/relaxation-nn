from pathlib import Path

from ml_collections import ConfigDict, config_dict

_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path=str(_REPO_ROOT / "data" / "clawpack_data" / "swe_2shock.npy"),
        sampling_strategy="monte_carlo",
        range_L=[0.0, -1.0],
        range_R=[1.0, 1.0],
        num_samples=[2540, 320, 160],
    )
    config.NetConfig = dict(
        layer_sizes=[
            [2, 128, 128, 128, 128, 128, 1],
            [2, 128, 128, 128, 128, 128, 1],
        ],
        configuration=["DNN", "DNN"],
        activation=["tanh", "tanh"],
        ibc_type=["2shock", "2shock"],
        loss="MSE",
    )
    config.TrainConfig = dict(
        epochs=50000,
        ratio=[0.01, 1.0, 1.0, 1.0],
        int_weights=[1.0, 1.0, 1.0],
        optimizer="Adam",
        lr=1e-3,
        decay="CosineAnnealing",
        scheduler_every=1,
        cosine_eta_min=0.0,
    )
    config.model = config_dict.placeholder(str)
    config.plot_label = "RelaxNN"
    config.train_mode = "train"
    config.torch_seed = config_dict.placeholder(int)
    config.root_dir = ""
    config.timestamp = ""
    return config
