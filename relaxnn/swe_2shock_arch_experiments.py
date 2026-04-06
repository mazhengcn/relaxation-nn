import argparse
import copy
import csv
import gc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from relaxnn.config.swe_2shock_compare import get_config
from relaxnn.main import run_with_config

DEFAULT_MODELS = ("swe_v1", "swe_v2")
DEFAULT_LAYERS = (3, 4, 5)
DEFAULT_NEURONS = (16, 32, 64, 128)


@dataclass(frozen=True)
class Variant:
    model: str
    hidden_layers: int
    neurons: int

    @property
    def name(self):
        return "{}_layers_{}_neurons_{}".format(
            self.model, self.hidden_layers, self.neurons
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RelaxNN SWE 2shock architecture comparison experiments."
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument(
        "--layers",
        default=",".join(str(value) for value in DEFAULT_LAYERS),
    )
    parser.add_argument(
        "--neurons",
        default=",".join(str(value) for value in DEFAULT_NEURONS),
    )
    parser.add_argument("--torch_seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--scheduler_every", type=int, default=1)
    parser.add_argument("--cosine_eta_min", type=float, default=0.0)
    parser.add_argument(
        "--root_dir",
        default="",
        help="Optional output root for all experiment runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = _resolve_root_dir(args.root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    models = _parse_models(args.models)
    hidden_layers = _parse_positive_ints(args.layers, "layers")
    neurons = _parse_positive_ints(args.neurons, "neurons")
    variants = [
        Variant(model=model, hidden_layers=layer_count, neurons=width)
        for model in models
        for layer_count in hidden_layers
        for width in neurons
    ]

    rows = []
    for variant in variants:
        config = copy.deepcopy(get_config())
        config.root_dir = str(root_dir)
        config.timestamp = variant.name
        config.model = variant.model
        config.plot_label = variant.model
        config.torch_seed = args.torch_seed
        config.TrainConfig.epochs = args.epochs
        config.TrainConfig.scheduler_every = args.scheduler_every
        config.TrainConfig.cosine_eta_min = args.cosine_eta_min
        layer_sizes = _build_layer_sizes(variant.hidden_layers, variant.neurons)
        config.NetConfig.layer_sizes = [layer_sizes, layer_sizes.copy()]

        result = _run_variant(config, variant)
        rows.append(result)
        _write_results(root_dir / "results.csv", rows)


def _resolve_root_dir(root_dir: str):
    if root_dir:
        return Path(root_dir)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return Path("_output") / "relaxnn" / "swe" / "2shock_arch_compare" / timestamp


def _parse_models(models_arg: str):
    models = [value.strip() for value in models_arg.split(",") if value.strip()]
    unknown_models = [model for model in models if model not in DEFAULT_MODELS]
    if unknown_models:
        raise ValueError("Unknown models: {}".format(unknown_models))
    return models


def _parse_positive_ints(values_arg: str, name: str):
    values = [int(value.strip()) for value in values_arg.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("{} must contain positive integers".format(name))
    return values


def _build_layer_sizes(hidden_layers: int, neurons: int):
    return [2, *([neurons] * hidden_layers), 1]


def _run_variant(config, variant: Variant):
    try:
        run_dir, summary = run_with_config(config)
        status = summary.get("status", "completed")
    except Exception as exc:  # pragma: no cover
        run_dir = Path(config.root_dir) / config.timestamp
        summary = {}
        status = "error: {}".format(exc)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "model": variant.model,
        "hidden_layers": variant.hidden_layers,
        "neurons": variant.neurons,
        "epochs": config.TrainConfig.epochs,
        "optimizer": config.TrainConfig.optimizer,
        "lr": config.TrainConfig.lr,
        "decay": config.TrainConfig.decay,
        "scheduler_every": config.TrainConfig.scheduler_every,
        "cosine_eta_min": config.TrainConfig.cosine_eta_min,
        "torch_seed": config.torch_seed,
        "status": status,
        "run_dir": str(run_dir),
        "final_epoch": summary.get("final_epoch"),
        "final_total_loss": summary.get("final_total_loss"),
        "final_mae": summary.get("final_mae"),
        "final_l2re": summary.get("final_l2re"),
        "best_mae": summary.get("best_mae"),
        "best_mae_epoch": summary.get("best_mae_epoch"),
        "best_l2re": summary.get("best_l2re"),
        "best_l2re_epoch": summary.get("best_l2re_epoch"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
    }


def _write_results(csv_path: Path, rows: list[dict]):
    fieldnames = [
        "model",
        "hidden_layers",
        "neurons",
        "epochs",
        "optimizer",
        "lr",
        "decay",
        "scheduler_every",
        "cosine_eta_min",
        "torch_seed",
        "status",
        "run_dir",
        "final_epoch",
        "final_total_loss",
        "final_mae",
        "final_l2re",
        "best_mae",
        "best_mae_epoch",
        "best_l2re",
        "best_l2re_epoch",
        "elapsed_seconds",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
