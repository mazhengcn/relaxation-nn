from pathlib import Path

import torch
from absl import app, flags, logging
from ml_collections import config_flags

import advection
import burgers
import euler
import generator
import swe
import train

_CONFIG = config_flags.DEFINE_config_file("config")

FLAGS = flags.FLAGS
DEVICE = torch.device("cuda:0")

model_dict = {
    "advection": advection.AdvectionNet,
    "burgers": burgers.BurgersNet,
    "swe": swe.SweNet,
    "euler": euler.EulerNet,
}


def save_config(save_path):
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    # Serialize config as json
    logging.info("Saving config.")
    config_path = save_path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(FLAGS.config.to_json_best_effort(indent=2))


def main(argv):
    torch.manual_seed(FLAGS.config.torch_seed)
    root_dir = FLAGS.config.root_dir
    if not isinstance(root_dir, Path):
        root_dir = Path(root_dir)
    if not root_dir.exists():
        root_dir.mkdir()
    time_dir = root_dir / FLAGS.config.timestamp
    if not time_dir.exists():
        time_dir.mkdir()
    csv_path = time_dir / "history.csv"
    model_dir = time_dir / "model_state_dict"
    lr_dir = time_dir / "lr_state_dict"
    if not model_dir.exists():
        model_dir.mkdir()
    if not lr_dir.exists():
        lr_dir.mkdir()

    save_config(time_dir)

    logging.get_absl_handler().use_absl_log_file("train", time_dir)

    dataset = generator.DataGenerator(FLAGS.config.DataConfig)
    training_data = dataset.get_iter()
    x_test, q_test = dataset.load_testdata()
    if FLAGS.config.model not in model_dict:
        raise ValueError("Model not set up")
    else:
        model = model_dict[FLAGS.config.model](FLAGS.config.NetConfig).to(DEVICE)
    # model_path = Path(
    #     "./_output_sv/center_riemann/2023-06-17T04:08:37/model_state_dict/model_30000"
    # )
    # lr_path = Path(
    #     "./_output_sv/center_riemann/2023-06-17T04:08:37/lr_state_dict/model_30000"
    # )
    # model.load_state_dict(torch.load(model_path))
    if FLAGS.config.train_mode == "train":
        train.train(
            device=DEVICE,
            training_data=training_data,
            x_test=x_test,
            q_test=q_test,
            model=model,
            config=FLAGS.config.TrainConfig,
            csv_path=csv_path,
            model_dir=model_dir,
            lr_dir=lr_dir,
        )
    if FLAGS.config.train_mode == "hessian":
        train.hessian_analysis(
            device=DEVICE,
            training_data=training_data,
            x_test=x_test,
            q_test=q_test,
            model=model,
            config=FLAGS.config.TrainConfig,
            csv_path=csv_path,
            model_dir=model_dir,
            lr_dir=lr_dir,
        )
    elif FLAGS.config.train_mode == "sv":
        train.train_sv(
            device=DEVICE,
            training_data=training_data,
            x_test=x_test,
            q_test=q_test,
            model=model,
            config=FLAGS.config.TrainConfig,
            csv_path=csv_path,
            model_dir=model_dir,
            lr_dir=lr_dir,
        )
    elif FLAGS.config.train_mode == "svpinn":
        train.svpinn(
            device=DEVICE,
            training_data=training_data,
            x_test=x_test,
            q_test=q_test,
            model=model,
            config=FLAGS.config.TrainConfig,
            csv_path=csv_path,
            model_dir=model_dir,
            lr_dir=lr_dir,
        )
    else:
        raise ValueError("other mode have not been implemented")


if __name__ == "__main__":
    app.run(main)
