from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from absl import app, flags, logging
from ml_collections import config_flags
from pinn.model import burgers
from shared import generator, train
from shared.runtime import DEVICE

_CONFIG = config_flags.DEFINE_config_file("config")

FLAGS = flags.FLAGS

model_dict = {
    "burgers": burgers.BurgersNet,
}


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


def save_config(config, save_path):
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    logging.info("Saving config.")
    config_path = save_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_best_effort(indent=2))


def run_with_config(config):
    torch.manual_seed(config.torch_seed)
    with config.DataConfig.unlocked():
        if config.DataConfig.sampling_strategy == "lhs":
            if "lhs_criterion" not in config.DataConfig or config.DataConfig.lhs_criterion is None:
                config.DataConfig.lhs_criterion = "center"
            if "sampling_seed" not in config.DataConfig or config.DataConfig.sampling_seed is None:
                config.DataConfig.sampling_seed = config.torch_seed
            if "interior_grid_shape" in config.DataConfig:
                del config.DataConfig["interior_grid_shape"]
        else:
            if "lhs_criterion" in config.DataConfig:
                del config.DataConfig["lhs_criterion"]
            if "sampling_seed" in config.DataConfig:
                del config.DataConfig["sampling_seed"]

        if config.DataConfig.sampling_strategy != "fixed_grid" and "interior_grid_shape" in config.DataConfig:
            del config.DataConfig["interior_grid_shape"]

    time_dir, csv_path, model_dir, lr_dir = _prepare_run_dirs(
        config.root_dir, config.timestamp
    )

    mygenerator = generator.Generator(config.DataConfig)
    save_config(config, time_dir)
    logging.get_absl_handler().use_absl_log_file("train", time_dir)
    mygenerator.export_reference_samples(time_dir / "reference_samples.npz")
    x_test, q_test = mygenerator.load_testdata()
    model = model_dict[config.model](config.NetConfig).to(DEVICE)

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
        return time_dir, summary
    else:
        raise ValueError("other mode have not been implemented")


def main(argv):
    del argv
    run_with_config(FLAGS.config)


if __name__ == "__main__":
    app.run(main)
