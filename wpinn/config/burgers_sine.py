from ml_collections import ConfigDict, config_dict


def get_config() -> ConfigDict:
    config = ConfigDict()
    config.DataConfig = dict(
        testdata_path="data/clawpack_data/burgers_sine.npy",
        sampling_strategy="monte_carlo",
        range_L=[0.0, -1.0],
        range_R=[1.0, 1.0],
        num_samples=[4096, 512, 512],
    )
    config.NetConfig = dict(
        solution_layer_sizes=[2, 64, 64, 64, 64, 1],
        test_layer_sizes=[2, 32, 32, 32, 1],
        configuration="DNN",
        solution_configuration="DNN",
        test_configuration="DNN",
        activation="tanh",
        solution_activation="tanh",
        test_activation="tanh",
        initialization="xavier_uniform",
        test_initialization="xavier_uniform",
        ibc_type=["sine", "sine"],
        entropy_norm="H1",
        cutoff="def_max",
        weak_form="partial",
        c_mode="max",
        entropy_c_samples=101,
        c_range_factor=2.0,
        sign_smoothing=1e-2,
        use_relu=True,
        p=2,
        eps=1e-12,
        data_weight=10.0,
        residual_weight=1.0,
        solution_regularization=0.0,
        test_regularization=0.0,
        domain_min=[0.0, -1.0],
        domain_max=[1.0, 1.0],
    )
    config.TrainConfig = dict(
        epochs=20000,
        maximize_steps=1,
        minimize_steps=5,
        optimizer="Adam",
        solution_lr=1e-3,
        test_lr=1e-3,
        solution_decay="CosineAnnealing",
        test_decay="CosineAnnealing",
        solution_cosine_eta_min=1e-6,
        test_cosine_eta_min=1e-6,
        solution_amsgrad=True,
        test_amsgrad=True,
        test_reset_every=2000,
        history_every=100,
        log_every=100,
        checkpoint_every=1000,
    )
    config.model = "burgers"
    config.plot_label = "WPINN"
    config.train_mode = "train"
    config.torch_seed = config_dict.placeholder(int)
    config.output_root = "_output/wpinn/burgers/sine"
    config.experiment_name = (
        "adam_cosine_20000_mc_4096_512_512_partial_h1_w10_entropy101_sol64_test32"
    )
    config.root_dir = ""
    config.timestamp = ""
    return config
