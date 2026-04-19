from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path="data/clawpack_data/burgers_sine.npy",
        sampling_strategy="monte_carlo",
        lhs_criterion="center",
        sampling_seed=None,
        range_L=[0.0, -1.0],
        range_R=[1.0, 1.0],
        num_samples=[2500, 320, 160],
        interior_grid_shape=[10, 250],
    )
    config.NetConfig = dict(
        layer_sizes=[
            [2, 20, 20, 20, 20, 20, 1],
            [2, 20, 20, 20, 20, 20, 1],
        ],
        configuration=["DNN", "DNN"],
        activation=["tanh", "tanh"],
        ibc_type=["sine", "sine"],
        loss="MSE",
    )
    config.TrainConfig = dict(
        epochs=30000,
        ratio=[0.1, 2.0, 10.0, 10.0],
        int_weights=[1.0, 1.0],
        history_terms=["res_loss", "flux_loss", "u_ic", "F_ic", "u_bc", "F_bc"],
        optimizer="LBFGS",
        lr=1.0,
        lbfgs_max_iter=1,
        lbfgs_history_size=100,
        lbfgs_line_search_fn="strong_wolfe",
        history_every=1,
        log_every=100,
        checkpoint_every=1000,
    )
    config.model = "burgers"
    config.plot_label = "RelaxNN"
    config.train_mode = "train"
    config.torch_seed = config_dict.placeholder(int)
    config.root_dir = ""
    config.timestamp = ""
    return config
