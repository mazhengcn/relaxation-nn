from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path="data/clawpack_data/burgers_riemann.npy",
        sampling_strategy="monte_carlo",
        lhs_criterion="center",
        sampling_seed=None,
        eval_time_window=[],
        range_L=[0.0, -0.6],
        range_R=[1.0, 0.6],
        num_samples=[10000, 1000, 1000],
        interior_grid_shape=[10, 250],
    )
    config.NetConfig = dict(
        layer_sizes=[2, 64, 64, 64, 64, 1],
        configuration="DNN",
        activation="tanh",
        ibc_type=["riemann", "riemann"],
        loss="MSE",
        initialization="xavier_uniform",
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
    config.model = "burgers"
    config.plot_label = "PINN"
    config.train_mode = "train"
    config.torch_seed = config_dict.placeholder(int)
    # By default, runs are saved to output_root/experiment_name/timestamp.
    config.output_root = "_output/pinn/burgers/riemann"
    config.experiment_name = (
        "adam_cosine_eta1e6_300000_mc_10000_1000_1000_tanh_xavier_uniform_w1_10_10_n64"
    )
    config.root_dir = ""
    config.timestamp = ""
    return config
