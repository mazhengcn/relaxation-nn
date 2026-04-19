from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path="data/clawpack_data/euler_shocktube.npy",
        sampling_strategy="monte_carlo",
        lhs_criterion="center",
        sampling_seed=None,
        eval_time_window=[],
        range_L=[0.0, -0.8],
        range_R=[0.4, 0.8],
        num_samples=[10000, 1000, 1000],
        interior_grid_shape=[10, 1000],
    )
    config.NetConfig = dict(
        layer_sizes=[2, 64, 64, 64, 64, 3],
        configuration="DNN",
        activation="tanh",
        ibc_type=["shock_tube", "shock_tube"],
        loss="MSE",
        initialization="xavier_uniform",
        gamma=1.4,
        shu_osher_amplitude=0.2,
    )
    config.TrainConfig = dict(
        epochs=300000,
        loss_weights=dict(
            res_loss=1.0,
            u_ic=10.0,
            u_bc=10.0,
        ),
        history_terms=["res_loss", "u_ic", "u_bc"],
        optimizer="Adam",
        lr=1e-3,
        decay="CosineAnnealing",
        scheduler_every=1,
        cosine_eta_min=1e-6,
        history_every=1000,
        log_every=100,
        checkpoint_every=1000,
    )
    config.model = "euler"
    config.plot_label = "PINN"
    config.train_mode = "train"
    config.torch_seed = config_dict.placeholder(int)
    config.output_root = "_output/pinn/euler/shock_tube"
    config.experiment_name = ""
    config.root_dir = ""
    config.timestamp = ""
    return config
