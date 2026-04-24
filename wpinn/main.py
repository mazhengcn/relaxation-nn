from pathlib import Path
import sys
import csv
import json

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from absl import app, flags, logging
from ml_collections import config_flags
from shared import generator
from shared.path_utils import repo_relative_path, resolve_repo_path
from shared.runtime import DEVICE
from wpinn import train
from wpinn.evaluate import load_model, resolve_checkpoint_epoch
from wpinn.model import burgers

_CONFIG = config_flags.DEFINE_config_file("config")

FLAGS = flags.FLAGS

MODEL_DICT = {
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


def _resolve_root_dir(config):
    if "root_dir" in config and config.root_dir:
        return resolve_repo_path(config.root_dir)

    if "output_root" in config and config.output_root:
        root_dir = resolve_repo_path(config.output_root)
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

    logging.info("Saving config.")
    config_path = save_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_best_effort(indent=2))


def _write_ensemble_info(config, save_path):
    info = getattr(config, "legacy_ensemble_info", None)
    if not info:
        return

    info_path = save_path / "EnsembleInfo.csv"
    with open(info_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for key, value in info.items():
            writer.writerow([key, value])


def _write_info_model(config, save_path, summary, x_test, q_test):
    model, _, epoch = load_model(save_path, checkpoint_epoch="best_total")
    x_tensor = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        prediction = model(x_tensor).detach().cpu().numpy()

    reference = np.asarray(q_test)
    mae = float(np.mean(np.abs(prediction - reference)))
    rel_mae = float(mae / np.mean(np.abs(reference)))

    data_config = config.DataConfig
    legacy_n_u = int(
        getattr(
            data_config,
            "num_u_total",
            int(data_config.num_samples[1]) + int(data_config.num_samples[2]),
        )
    )
    legacy_n_f = int(getattr(data_config, "num_f_total", int(data_config.num_samples[0])))
    legacy_n_int = int(getattr(data_config, "num_internal", 0))
    validation_size = float(getattr(config, "validation_size", 0.0))
    retrain_value = getattr(config, "torch_seed", "")

    info_rows = [
        ("Nu_train", legacy_n_u),
        ("Nf_train", legacy_n_f),
        ("Nint_train", legacy_n_int),
        ("validation_size", validation_size),
        ("train_time", float(summary["elapsed_seconds"])),
        ("L2_norm_test", mae),
        ("rel_L2_norm", rel_mae),
        ("loss_tot", float(summary["best_total"])),
        ("loss_vars", float(summary["best_total_data_loss"])),
        ("loss_pde", float(summary["best_total_res_loss"])),
        ("loss_pde_no_norm", float(summary["best_total_res_raw"])),
        (
            "loss_pde_no_norm_after_max",
            float(summary["best_total_res_raw_after_max"]),
        ),
        ("best_epoch", int(epoch)),
        ("retrain", retrain_value),
    ]

    with open(save_path / "InfoModel.txt", "w", encoding="utf-8", newline="") as f:
        for key, value in info_rows:
            f.write("{}, {}\n".format(key, value))

    with open(save_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _load_resume_state(config, model):
    resume_root_dir = getattr(config, "resume_root_dir", "")
    if not resume_root_dir:
        return None, 0

    resume_dir = resolve_repo_path(resume_root_dir)
    model_dir = resume_dir / "model_state_dict"
    lr_dir = resume_dir / "lr_state_dict"
    checkpoint_epoch = int(getattr(config, "resume_checkpoint_epoch", 0) or 0)
    epoch = resolve_checkpoint_epoch(
        model_dir,
        checkpoint_epoch if checkpoint_epoch > 0 else None,
    )
    model_path = model_dir / "model_{:02d}".format(epoch)
    lr_path = lr_dir / "model_{:02d}".format(epoch)
    logging.info("Resuming from %s at epoch %s", resume_dir, epoch)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    if not bool(getattr(config, "resume_load_optimizer_state", True)):
        return None, epoch

    return torch.load(lr_path, map_location=DEVICE), epoch


def run_with_config(config):
    torch.manual_seed(config.torch_seed)
    root_dir = _resolve_root_dir(config)
    with config.DataConfig.unlocked():
        config.DataConfig.testdata_path = repo_relative_path(
            config.DataConfig.testdata_path
        )
    with config.unlocked():
        if "output_root" in config and config.output_root:
            config.output_root = repo_relative_path(config.output_root)
        config.root_dir = repo_relative_path(root_dir)

    time_dir, csv_path, model_dir, lr_dir = _prepare_run_dirs(
        root_dir, config.timestamp
    )

    mygenerator = generator.Generator(config.DataConfig)
    save_config(config, time_dir)
    _write_ensemble_info(config, time_dir)
    logging.get_absl_handler().use_absl_log_file("train", time_dir)
    x_test, q_test = mygenerator.load_testdata()
    model = MODEL_DICT[config.model](config.NetConfig).to(DEVICE)
    optimizer_state, start_epoch = _load_resume_state(config, model)

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
            start_epoch=start_epoch,
            optimizer_state=optimizer_state,
        )
        _write_info_model(config, time_dir, summary, x_test, q_test)
        return time_dir, summary
    raise ValueError("other mode have not been implemented")


def main(argv):
    del argv
    run_with_config(FLAGS.config)


if __name__ == "__main__":
    app.run(main)
