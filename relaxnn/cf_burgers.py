from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path="/nfs/my/Origin/clawpack_data/burgers_sine.npy",
        seed=config_dict.placeholder(int),
        distribution="uniform",
        range_L=[0.0, -1.0],
        range_R=[1.0, 1.0],
        num_samples=[2540, 320, 160],
        sample=400000,
    )
    config.NetConfig = dict(
        layer_sizes=[
            [2, 32, 32, 32, 32, 32, 32, 32, 32, 1],
            [2, 64, 64, 64, 64, 1],
        ],
        configuration=["resnet", "resnet"],
        activation=["tanh", "tanh"],
        ibc_type=["sine", "sine"],
        loss="MSE",
    )
    config.TrainConfig = dict(
        epochs=300001,
        ratio=[0.5, 2.0, 5.0, 5.0],
        int_weights=[1.0, 1.0],
        optimizer="Adam",
        lr=1e-3,
        decay="Exponential",
        decay_rate=0.99,
    )
    config.model = config_dict.placeholder(str)
    config.train_mode = "train"
    config.torch_seed = config_dict.placeholder(int)
    config.root_dir = ""
    config.timestamp = ""
    return config
