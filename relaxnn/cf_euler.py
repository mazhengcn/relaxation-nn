from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path="/nfs/my/Origin/clawpack_data/euler_shocksine.npy",
        seed=config_dict.placeholder(int),
        distribution="uniform",
        range_L=[0.0, -4.5],
        range_R=[1.8, 2.7],
        num_samples=[2540, 320, 160],
        sample=900002,
    )
    config.NetConfig = dict(
        layer_sizes=[
            [2, 384, 384, 384, 384, 384, 3],
            [2, 256, 256, 256, 256, 256, 2],
        ],
        configuration=["DNN", "DNN"],
        activation=["tanh", "tanh"],
        ibc_type=["shu_osher", "shu_osher"],
        loss="MSE",
    )
    config.TrainConfig = dict(
        epochs=600001,
        ratio=[0.1, 1.0, 10.0, 10.0],
        int_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
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
