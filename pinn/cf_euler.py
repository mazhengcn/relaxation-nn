from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path="/nfs/my/Origin/clawpack_data/euler_shocktube.npy",
        seed=config_dict.placeholder(int),
        distribution="uniform",
        range_L=[0.0, -0.8],
        range_R=[0.4, 0.8],
        num_samples=[2540, 320, 160],
        sample=700001,
    )
    config.TrainConfig = dict(
        epochs=600001,
        weights=[0.1, 1.0, 1.0],
        optimizer="Adam",
        lr=1e-3,
        decay="Exponential",
        decay_rate=0.99,
        metric="MAE",
    )
    config.NetConfig = dict(
        layer_sizes=[2, 384, 384, 384, 384, 384, 384, 3],
        configuration="DNN",
        activation="tanh",
        ibc_type=["lax_tube", "lax_tube"],
        loss="MSE",
    )
    config.model = "euler"
    config.train_mode = "train"
    config.torch_seed = config_dict.placeholder(int)
    config.root_dir = ""
    config.timestamp = ""
    return config
