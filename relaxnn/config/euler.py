from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path="data/clawpack_data/euler_shocktube.npy",
        distribution="uniform",
        range_L=[0.0, -0.8],
        range_R=[0.4, 0.8],
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
        ibc_type=["shock_tube", "shock_tube"],
        loss="MSE",
    )
    config.TrainConfig = dict(
        epochs=600000,
        ratio=[1.0, 1000.0, 100.0, 100.0],
        int_weights=[1.0, 0.5, 0.1, 1.0],
        optimizer="Adam",
        lr=1e-3,
        decay="CosineAnnealing",
        scheduler_every=1000,
        cosine_eta_min=1e-6,
    )
    config.model = "euler_v3"
    config.train_mode = "train"
    config.torch_seed = config_dict.placeholder(int)
    config.output_root = "_output/relaxnn/euler_v3/shock_tube"
    config.experiment_name = "loss_weighting"
    config.root_dir = ""
    config.timestamp = ""
    return config
