import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from absl import app, flags, logging
from ml_collections import config_flags
from pinns_utils_torch import generator, train
from pinns_utils_torch.runtime import DEVICE

from relaxation_nn.model import burgers, euler_v1, euler_v2, euler_v3, swe_v1, swe_v2

_CONFIG = config_flags.DEFINE_config_file("config")

FLAGS = flags.FLAGS

model_dict = {
    "burgers": burgers.BurgersNet,
    "swe_v1": swe_v1.SweNet,
    "swe_v2": swe_v2.SweNet,
    "euler_v1": euler_v1.EulerNet,
    "euler_v2": euler_v2.EulerNet,
    "euler_v3": euler_v3.EulerNet,
}


def _format_elapsed_seconds(seconds):
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, secs)


def _log_training_summary(time_dir: Path, summary: dict):
    if not summary:
        return

    elapsed_seconds = float(summary.get("elapsed_seconds", 0.0))
    logging.info(
        "Training summary | status : {} | final_epoch : {} | final_total_loss : {} | "
        "best_MAE : {} @ {} | best_L2RE : {} @ {} | elapsed_seconds : {:.2f} | "
        "elapsed_hms : {} | output_dir : {}".format(
            summary.get("status"),
            summary.get("final_epoch"),
            summary.get("final_total_loss"),
            summary.get("best_mae"),
            summary.get("best_mae_epoch"),
            summary.get("best_l2re"),
            summary.get("best_l2re_epoch"),
            elapsed_seconds,
            _format_elapsed_seconds(elapsed_seconds),
            time_dir,
        )
    )


def _prepare_run_dirs(root_dir, timestamp):
    if not isinstance(root_dir, Path):
        root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    time_dir = root_dir / timestamp
    if time_dir.exists():
        existing_entries = [entry.name for entry in time_dir.iterdir()]
        if existing_entries:
            raise FileExistsError(
                "Run directory {} already exists and is not empty. "
                "Use a new root_dir/timestamp or delete it before rerunning.".format(
                    time_dir
                )
            )
    else:
        time_dir.mkdir()

    csv_path = time_dir / "history.csv"
    model_dir = time_dir / "model_state_dict"
    lr_dir = time_dir / "lr_state_dict"
    model_dir.mkdir(exist_ok=True)
    lr_dir.mkdir(exist_ok=True)
    return time_dir, csv_path, model_dir, lr_dir


def _resolve_root_dir(config):
    if "root_dir" in config and config.root_dir:
        return Path(config.root_dir)

    if "output_root" in config and config.output_root:
        root_dir = Path(config.output_root)
        if "experiment_name" in config and config.experiment_name:
            root_dir = root_dir / config.experiment_name
        return root_dir

    raise ValueError(
        "Config must set `root_dir` or set `output_root` "
        "and optionally `experiment_name`."
    )


def save_config(config, save_path):
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    # Serialize config as json
    logging.info("Saving config.")
    config_path = save_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_best_effort(indent=2))


def _prepare_config_for_run(config):
    with config.DataConfig.unlocked():
        if "sampling_strategy" not in config.DataConfig:
            config.DataConfig.sampling_strategy = "monte_carlo"

        if config.DataConfig.sampling_strategy == "lhs":
            if (
                "lhs_criterion" not in config.DataConfig
                or config.DataConfig.lhs_criterion is None
            ):
                config.DataConfig.lhs_criterion = "center"
            if (
                "sampling_seed" not in config.DataConfig
                or config.DataConfig.sampling_seed is None
            ):
                config.DataConfig.sampling_seed = config.torch_seed
            if "interior_grid_shape" in config.DataConfig:
                del config.DataConfig["interior_grid_shape"]
        else:
            if "lhs_criterion" in config.DataConfig:
                del config.DataConfig["lhs_criterion"]
            if "sampling_seed" in config.DataConfig:
                del config.DataConfig["sampling_seed"]

        if (
            config.DataConfig.sampling_strategy != "fixed_grid"
            and "interior_grid_shape" in config.DataConfig
        ):
            del config.DataConfig["interior_grid_shape"]


def run_with_config(config):
    torch.manual_seed(config.torch_seed)
    _prepare_config_for_run(config)
    root_dir = _resolve_root_dir(config)
    with config.unlocked():
        config.root_dir = str(root_dir)

    time_dir, csv_path, model_dir, lr_dir = _prepare_run_dirs(
        root_dir, config.timestamp
    )

    mygenerator = generator.Generator(config.DataConfig)
    save_config(config, time_dir)

    logging.get_absl_handler().use_absl_log_file("train", time_dir)
    mygenerator.export_reference_samples(time_dir / "reference_samples.npz")
    x_test, q_test = mygenerator.load_testdata()
    model = model_dict[config.model](config.NetConfig).to(DEVICE)
    # model_path = Path(
    #     "/nfs/my/OriginRela/_output/euler_v3/blast/2023-10-18T02-21-34/model_state_dict/model_600000"
    # )
    # model.load_state_dict(torch.load(model_path))

    if config.train_mode == "train":
        summary = train.train(
            device=DEVICE,
            datagenerator=mygenerator,
            x_test=x_test,
            q_test=q_test,
            model=model,
            config=config.TrainConfig,
            csv_path=csv_path,
            model_dir=model_dir,
            lr_dir=lr_dir,
        )
        _log_training_summary(time_dir, summary)
        return time_dir, summary
    else:
        raise ValueError("other mode have not been implemented")


@app.run
def main(argv):
    del argv
    run_with_config(FLAGS.config)


if __name__ == "__main__":
    app.run(main)
