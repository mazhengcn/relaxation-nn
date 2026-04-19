from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path="data/clawpack_data/swe_2shock.npy",
        distribution="uniform",
        range_L=[0.0, -1.0],
        range_R=[1.0, 1.0],
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
        ibc_type=["2shock", "2shock"],
        loss="MSE",
    )
    config.TrainConfig = dict(
        epochs=600000,
        ratio=[0.1, 1.0, 1.0, 1.0],
        int_weights=[1.0, 1.0, 1.0, 1.0],
        optimizer="Adam",
        lr=1e-3,
        decay="CosineAnnealing",
        scheduler_every=1,
        cosine_eta_min=1e-6,
    )
    config.model = "swe_v2"
    config.train_mode = "train"
    config.torch_seed = config_dict.placeholder(int)
    config.output_root = "_output/relaxnn/swe_v2/2shock"
    config.experiment_name = ""
    config.root_dir = ""
    config.timestamp = ""
    return config
