from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path="/workspaces/relaxation-nn/data/clawpack_data/2d_euler_riemann.npy",
        seed=config_dict.placeholder(int),
        distribution="uniform",
        range_L=[0.0, 0.0, 0.0],
        range_R=[0.25, 1.0, 1.0],
        num_samples=[2540, 320, 40],
        sample=900002,
    )
    config.NetConfig = dict(
        layer_sizes=[
            [3, 64, 64, 64, 64, 1],
            [3, 64, 64, 64, 64, 1],
        ],
        configuration=["DNN", "DNN"],
        activation=["tanh", "tanh"],
        ibc_type=["riemann", "riemann"],
        loss="MSE",
    )
    config.TrainConfig = dict(
        epochs=300001,
        ratio=[1.0, 1.0, 10.0, 10.0],
        int_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
