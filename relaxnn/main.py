from pathlib import Path

import generator
import model.burgers as burgers
import model.euler_v1 as euler_v1
import model.euler_v2 as euler_v2
import model.euler_v3 as euler_v3
import model.swe_v1 as swe_v1
import model.swe_v2 as swe_v2
import torch
import train
from absl import app, flags, logging
from ml_collections import config_flags

_CONFIG = config_flags.DEFINE_config_file("config")

FLAGS = flags.FLAGS
DEVICE = torch.device("cuda:0")

model_dict = {
    "burgers": burgers.BurgersNet,
    "swe_v1": swe_v1.SweNet,
    "swe_v2": swe_v2.SweNet,
    "euler_v1": euler_v1.EulerNet,
    "euler_v2": euler_v2.EulerNet,
    "euler_v3": euler_v3.EulerNet,
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
        root_dir.mkdir(parents=True, exist_ok=True)
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

    mygenerator = generator.Generator(FLAGS.config.DataConfig)
    x_test, q_test = mygenerator.load_testdata()
    model = model_dict[FLAGS.config.model](FLAGS.config.NetConfig).to(DEVICE)
    # model_path = Path(
    #     "/nfs/my/OriginRela/_output/euler_v3/blast/2023-10-18T02-21-34/model_state_dict/model_600000"
    # )
    # model.load_state_dict(torch.load(model_path))

    if FLAGS.config.train_mode == "train":
        train.train(
            device=DEVICE,
            datagenerator=mygenerator,
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
