from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path="data/clawpack_data/burgers_sine.npy",
        sampling_strategy="monte_carlo",
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
        ibc_type=["sine", "sine"],
        loss="MSE",
    )
    config.TrainConfig = dict(
        epochs=300000,
        ratio=[1.0, 10.0, 10.0, 10.0],
        int_weights=[1.0, 1.0],
        history_terms=["res_loss", "flux_loss", "u_ic", "F_ic", "u_bc", "F_bc"],
        optimizer="Adam",
        lr=1e-3,
        decay="CosineAnnealing",
        scheduler_every=1,
        cosine_eta_min=1e-6,
        history_every=1000,
        log_every=1000,
        checkpoint_every=1000,
    )
    config.model = "burgers"
    config.plot_label = "RelaxNN"
    config.train_mode = "train"
    config.torch_seed = config_dict.placeholder(int)
    config.output_root = "_output/relaxnn/burgers/sine"
    config.experiment_name = "different_seeds"
    config.root_dir = ""
    config.timestamp = ""
    return config
