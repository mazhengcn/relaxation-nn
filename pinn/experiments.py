import argparse
import copy
import csv
import gc
import importlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from pinn.main import run_with_config
from shared.path_utils import repo_relative_path, resolve_repo_path


@dataclass(frozen=True)
class Variant:
    name: str
    description: str


@dataclass(frozen=True)
class SeedVariant:
    seed: int

    @property
    def name(self):
        return "seed_{}".format(self.seed)

    @property
    def description(self):
        return "Repeat the same configuration with torch_seed={}".format(self.seed)


OPTIMIZER_VARIANTS = [
    Variant("adam", "Adam with configured scheduler"),
    Variant("lbfgs", "Pure L-BFGS"),
    Variant("adam_lbfgs", "Adam warm-up followed by L-BFGS"),
]

INITIALIZATION_VARIANTS = [
    Variant("tanh_xavier_uniform", "tanh activation with xavier_uniform"),
    Variant("tanh_xavier_normal", "tanh activation with xavier_normal"),
    Variant("relu_kaiming_uniform", "relu activation with kaiming_uniform"),
    Variant("relu_kaiming_normal", "relu activation with kaiming_normal"),
]

LOSS_WEIGHT_VARIANTS = [
    Variant("w1_1_1", "lighter IC/BC penalties"),
    Variant("w1_5_5", "moderate IC/BC penalties"),
    Variant("w1_10_10", "heavier IC/BC penalties"),
]

SAMPLING_VARIANTS = [
    Variant("monte_carlo", "Uniform Monte Carlo sampling"),
    Variant("lhs", "Latin hypercube sampling with SMT default criterion"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run staged PINN experiments.")
    parser.add_argument(
        "--stage",
        choices=[
            "optimizer",
            "loss_weights",
            "initialization",
            "sampling_strategy",
            "random_seed",
        ],
        default="loss_weights",
    )
    parser.add_argument("--root_dir", default="", help="Optional output root.")
    parser.add_argument(
        "--config_module",
        default="pinn.config.burgers_sine_experiments",
        help="Python module that defines get_config().",
    )
    parser.add_argument(
        "--variants",
        default="",
        help="Optional comma-separated subset of variants for the selected stage.",
    )
    parser.add_argument("--torch_seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--adam_epochs", type=int, default=20000)
    parser.add_argument(
        "--lbfgs_lr",
        type=float,
        default=None,
        help="Optional override for pure/adam_lbfgs L-BFGS learning rate.",
    )
    parser.add_argument(
        "--base_optimizer",
        choices=[variant.name for variant in OPTIMIZER_VARIANTS],
        default="lbfgs",
    )
    parser.add_argument(
        "--base_initialization",
        choices=[variant.name for variant in INITIALIZATION_VARIANTS],
        default="tanh_xavier_uniform",
    )
    parser.add_argument(
        "--base_loss_weights",
        choices=[variant.name for variant in LOSS_WEIGHT_VARIANTS],
        default="w1_10_10",
    )
    parser.add_argument(
        "--base_sampling_strategy",
        choices=[variant.name for variant in SAMPLING_VARIANTS],
        default="monte_carlo",
    )
    parser.add_argument(
        "--seed_values",
        default="42,3407,27182,31415,2027",
        help="Comma-separated list of seeds for random_seed stage.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    get_config = _load_get_config(args.config_module)
    root_dir = _resolve_root_dir(args.stage, args.root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in _selected_variants(args.stage, args.variants, args.seed_values):
        config = copy.deepcopy(get_config())
        config.root_dir = repo_relative_path(root_dir)
        config.timestamp = variant.name
        config.torch_seed = args.torch_seed
        config.TrainConfig.epochs = args.epochs
        if args.lbfgs_lr is not None:
            config.TrainConfig.lbfgs_lr = args.lbfgs_lr

        if args.stage == "optimizer":
            _apply_optimizer_variant(config, variant.name)
            _apply_initialization_variant(config, "tanh_xavier_uniform")
            _apply_loss_weight_variant(config, "w1_10_10")
            _apply_sampling_variant(config, "monte_carlo")
        elif args.stage == "loss_weights":
            _apply_optimizer_variant(config, args.base_optimizer)
            _apply_initialization_variant(config, args.base_initialization)
            _apply_loss_weight_variant(config, variant.name)
            _apply_sampling_variant(config, args.base_sampling_strategy)
        elif args.stage == "initialization":
            _apply_optimizer_variant(config, args.base_optimizer)
            _apply_initialization_variant(config, variant.name)
            _apply_loss_weight_variant(config, args.base_loss_weights)
            _apply_sampling_variant(config, args.base_sampling_strategy)
        elif args.stage == "sampling_strategy":
            _apply_optimizer_variant(config, args.base_optimizer)
            _apply_initialization_variant(config, args.base_initialization)
            _apply_loss_weight_variant(config, args.base_loss_weights)
            _apply_sampling_variant(config, variant.name)
        elif args.stage == "random_seed":
            _apply_optimizer_variant(config, args.base_optimizer)
            _apply_initialization_variant(config, args.base_initialization)
            _apply_loss_weight_variant(config, args.base_loss_weights)
            _apply_sampling_variant(config, args.base_sampling_strategy)
            config.torch_seed = variant.seed
            config.timestamp = variant.name
        else:
            raise ValueError("Unknown stage {}".format(args.stage))

        if args.stage != "random_seed":
            _apply_sampling_seed_default(config)

        if args.stage == "random_seed":
            _apply_sampling_seed_default(config)

        _prune_unused_train_config_fields(config, adam_epochs=args.adam_epochs)
        result = _run_variant(config, variant, args.stage)
        rows.append(result)
        _write_stage_summary(root_dir / "results.csv", rows)


def _load_get_config(module_name: str):
    module = importlib.import_module(module_name)
    if not hasattr(module, "get_config"):
        raise AttributeError("Module {} does not define get_config()".format(module_name))
    return module.get_config


def _apply_sampling_seed_default(config):
    with config.DataConfig.unlocked():
        if config.DataConfig.sampling_strategy == "lhs":
            config.DataConfig.lhs_criterion = "center"
            config.DataConfig.sampling_seed = config.torch_seed
        else:
            if "lhs_criterion" in config.DataConfig:
                del config.DataConfig["lhs_criterion"]
            if "sampling_seed" in config.DataConfig:
                del config.DataConfig["sampling_seed"]


def _resolve_root_dir(stage: str, root_dir: str):
    if root_dir:
        return resolve_repo_path(root_dir)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return resolve_repo_path(Path("_output") / "pinn" / "burgers" / stage / timestamp)


def _selected_variants(stage: str, variants_arg: str, seed_values_arg: str):
    if stage == "random_seed":
        return _selected_seed_variants(variants_arg or seed_values_arg)

    variant_map = {
        "optimizer": OPTIMIZER_VARIANTS,
        "loss_weights": LOSS_WEIGHT_VARIANTS,
        "initialization": INITIALIZATION_VARIANTS,
        "sampling_strategy": SAMPLING_VARIANTS,
    }
    variants = variant_map[stage]
    if not variants_arg:
        return variants

    requested = [name.strip() for name in variants_arg.split(",") if name.strip()]
    known_names = {variant.name for variant in variants}
    unknown = [name for name in requested if name not in known_names]
    if unknown:
        raise ValueError("Unknown variants for stage {}: {}".format(stage, unknown))

    return [variant for variant in variants if variant.name in requested]


def _selected_seed_variants(variants_arg: str):
    if not variants_arg:
        return []
    return [SeedVariant(int(name.strip())) for name in variants_arg.split(",") if name.strip()]


def _apply_optimizer_variant(config, variant_name: str):
    train_config = config.TrainConfig

    if variant_name == "adam":
        train_config.optimizer = "Adam"
        if "lr" not in train_config:
            train_config.lr = 1e-3
        if "decay" not in train_config:
            train_config.decay = "Exponential"
        if "scheduler_every" not in train_config:
            train_config.scheduler_every = 1000
        if (
            str(train_config.decay).strip().lower() == "exponential"
            and "decay_rate" not in train_config
        ):
            train_config.decay_rate = 0.99
        return

    if variant_name == "lbfgs":
        train_config.optimizer = "LBFGS"
        train_config.lr = getattr(train_config, "lbfgs_lr", 1.0)
        return

    if variant_name == "adam_lbfgs":
        train_config.optimizer = "Adam_LBFGS"
        if "adam_lr" not in train_config:
            train_config.adam_lr = getattr(train_config, "lr", 1e-3)
        if "adam_decay" not in train_config:
            train_config.adam_decay = getattr(train_config, "decay", "Exponential")
        if "adam_scheduler_every" not in train_config:
            train_config.adam_scheduler_every = getattr(
                train_config, "scheduler_every", 1000
            )
        if (
            str(train_config.adam_decay).strip().lower() == "exponential"
            and "adam_decay_rate" not in train_config
        ):
            train_config.adam_decay_rate = getattr(train_config, "decay_rate", 0.99)
        if "lbfgs_lr" not in train_config:
            train_config.lbfgs_lr = 1.0
        return

    raise ValueError("Unknown optimizer variant {}".format(variant_name))


def _apply_initialization_variant(config, variant_name: str):
    mapping = {
        "tanh_xavier_uniform": ("tanh", "xavier_uniform"),
        "tanh_xavier_normal": ("tanh", "xavier_normal"),
        "relu_kaiming_uniform": ("relu", "kaiming_uniform"),
        "relu_kaiming_normal": ("relu", "kaiming_normal"),
    }
    if variant_name not in mapping:
        raise ValueError("Unknown initialization variant {}".format(variant_name))
    activation, initialization = mapping[variant_name]
    config.NetConfig.activation = activation
    config.NetConfig.initialization = initialization


def _apply_sampling_variant(config, variant_name: str):
    if variant_name not in {"monte_carlo", "lhs"}:
        raise ValueError("Unknown sampling variant {}".format(variant_name))
    config.DataConfig.sampling_strategy = variant_name


def _apply_loss_weight_variant(config, variant_name: str):
    mapping = {
        "w1_1_1": (1.0, 1.0),
        "w1_5_5": (5.0, 5.0),
        "w1_10_10": (10.0, 10.0),
    }
    if variant_name not in mapping:
        raise ValueError("Unknown loss-weight variant {}".format(variant_name))
    if "ratio" in config.TrainConfig:
        res_weight = config.TrainConfig.ratio[0]
        flux_weight = config.TrainConfig.ratio[1]
        u_ic_weight, u_bc_weight = mapping[variant_name]
        config.TrainConfig.ratio = [
            res_weight,
            flux_weight,
            u_ic_weight,
            u_bc_weight,
        ]
    else:
        config.TrainConfig.loss_weights = dict(
            res_loss=1.0,
            u_ic=mapping[variant_name][0],
            u_bc=mapping[variant_name][1],
        )


def _run_variant(config, variant, stage: str):
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
        "stage": stage,
        "variant": variant.name,
        "description": variant.description,
        "optimizer": config.TrainConfig.optimizer,
        "activation": config.NetConfig.activation,
        "initialization": config.NetConfig.initialization,
        "sampling_strategy": config.DataConfig.sampling_strategy,
        "num_interior": config.DataConfig.num_samples[0],
        "num_initial": config.DataConfig.num_samples[1],
        "num_boundary": config.DataConfig.num_samples[2],
        "loss_weights": _loss_weight_summary(config.TrainConfig),
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


def _write_stage_summary(csv_path: Path, rows: list[dict]):
    fieldnames = [
        "stage",
        "variant",
        "description",
        "optimizer",
        "activation",
        "initialization",
        "sampling_strategy",
        "num_interior",
        "num_initial",
        "num_boundary",
        "loss_weights",
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


def _dict_to_compact_string(values):
    return ";".join("{}={}".format(key, value) for key, value in dict(values).items())


def _loss_weight_summary(train_config):
    if "ratio" in train_config:
        return ",".join(str(value) for value in train_config.ratio)
    return _dict_to_compact_string(train_config.loss_weights)


def _prune_unused_train_config_fields(config, adam_epochs: int):
    train_config = config.TrainConfig
    optimizer = train_config.optimizer

    if optimizer == "Adam":
        _delete_keys(
            train_config,
            [
                "adam_epochs",
                "adam_lr",
                "adam_decay",
                "adam_decay_rate",
                "adam_scheduler_every",
                "lbfgs_lr",
                "lbfgs_max_iter",
                "lbfgs_history_size",
                "lbfgs_line_search_fn",
                "lbfgs_max_eval",
                "lbfgs_tolerance_grad",
                "lbfgs_tolerance_change",
            ],
        )
        return

    if optimizer == "LBFGS":
        _delete_keys(
            train_config,
            [
                "adam_epochs",
                "adam_lr",
                "adam_decay",
                "adam_decay_rate",
                "adam_scheduler_every",
                "decay",
                "decay_rate",
                "scheduler_every",
            ],
        )
        return

    if optimizer == "Adam_LBFGS":
        train_config.adam_epochs = min(adam_epochs, max(train_config.epochs - 1, 1))
        _delete_keys(train_config, ["lr", "decay", "decay_rate", "scheduler_every"])
        return

    raise ValueError("Unknown optimizer {}".format(optimizer))


def _delete_keys(config, keys: list[str]):
    for key in keys:
        if key in config:
            del config[key]


if __name__ == "__main__":
    main()
