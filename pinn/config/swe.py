from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path="data/clawpack_data/swe_dam_break.npy",
        sampling_strategy="monte_carlo",
        range_L=[0.0, -1.5],
        range_R=[1.0, 1.5],
        num_samples=[10000, 1000, 1000],
    )
    config.NetConfig = dict(
        layer_sizes=[2, 64, 64, 64, 64, 1],
        configuration="DNN",
        activation="tanh",
        ibc_type=["dam_break", "dam_break"],
        loss="MSE",
        initialization="xavier_uniform",
    )
    config.TrainConfig = dict(
        epochs=600000,
        ratio=[0.1, 10.0, 10.0, 10.0],
        int_weights=[1.0, 1.0],
        optimizer="Adam",
        lr=1e-3,
        decay="CosineAnnealing",
        scheduler_every=1000,
        cosine_eta_min=1e-6,
    )
    config.model = "swe"
    config.plot_label = "PINN"
    config.train_mode = "train"
    config.torch_seed = config_dict.placeholder(int)
    config.output_root = "_output/pinn/swe/dam_break"
    config.experiment_name = ""
    config.root_dir = ""
    config.timestamp = ""
    return config
