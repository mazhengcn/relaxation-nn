import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from ml_collections import ConfigDict

from pinn.model import burgers
from shared.runtime import DEVICE

MODEL_DICT = {
    "burgers": burgers.BurgersNet,
}


def to_numpy(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.detach().cpu().numpy()
    if isinstance(inputs, np.ndarray):
        return inputs
    raise TypeError(
        "Unknown type of input, expected torch.Tensor or np.ndarray, but got {}".format(
            type(inputs)
        )
    )


def load_config(root_dir: Path):
    with open(root_dir / "config.json", "r", encoding="utf8") as f:
        return json.load(f)


def resolve_checkpoint_epoch(model_dir: Path, checkpoint_epoch: int | None = None):
    epochs = []
    for model_path in model_dir.glob("model_*"):
        try:
            epochs.append(int(model_path.name.split("_")[-1]))
        except ValueError:
            continue
    if not epochs:
        raise FileNotFoundError("No checkpoint found under {}".format(model_dir))
    if checkpoint_epoch is None:
        return max(epochs)
    if checkpoint_epoch not in epochs:
        raise FileNotFoundError(
            "Checkpoint epoch {} not found under {}".format(checkpoint_epoch, model_dir)
        )
    return checkpoint_epoch


def load_model(root_dir: Path, checkpoint_epoch: int | None = None):
    config = load_config(root_dir)
    model_dir = root_dir / "model_state_dict"
    epoch = resolve_checkpoint_epoch(model_dir, checkpoint_epoch)
    model = MODEL_DICT[config["model"]](ConfigDict(config["NetConfig"])).to(DEVICE)
    model_path = model_dir / "model_{:02d}".format(epoch)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model, config, epoch


def select_time_slice(x_test, q_test, target_t: float):
    unique_t = np.unique(x_test[:, 0])
    nearest_t = unique_t[np.argmin(np.abs(unique_t - target_t))]
    mask = np.isclose(x_test[:, 0], nearest_t)
    if not np.any(mask):
        raise ValueError("No test-data slice found near t={}".format(target_t))
    return nearest_t, x_test[mask], q_test[mask]


def plot_burgers_slice(
    root_dir: Path,
    target_t: float = 0.6,
    checkpoint_epoch: int | None = None,
    save_path: Path | None = None,
):
    model, config, epoch = load_model(root_dir, checkpoint_epoch)
    testdata = np.load(config["DataConfig"]["testdata_path"])
    x_test, q_test = testdata[:, 0:2], testdata[:, 2:3]
    actual_t, x_part, q_part = select_time_slice(x_test, q_test, target_t)

    x_tensor = torch.tensor(x_part, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        q_pred = to_numpy(model(x_tensor))

    order = np.argsort(x_part[:, 1])
    x_values = x_part[order, 1]
    q_true = q_part[order, 0]
    q_pred = q_pred[order, 0]

    if save_path is None:
        save_path = root_dir / "comparison_t_{:.1f}.pdf".format(actual_t)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(x_values, q_true, linewidth=2.0, label="Reference")
    ax.plot(x_values, q_pred, "--", linewidth=2.0, label=config.get("plot_label", "PINN"))
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("Burgers at t={:.1f} (epoch {})".format(actual_t, epoch))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def load_history(history_path: Path):
    with open(history_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("History file is empty: {}".format(history_path))
    return rows


def _as_float_array(rows, key):
    values = []
    for row in rows:
        value = row.get(key, "")
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=np.float64)


def _safe_for_log(values):
    values = values.copy()
    values[values <= 0] = np.nan
    return values


def plot_history(root_dir: Path, save_path: Path | None = None):
    rows = load_history(root_dir / "history.csv")
    epochs = _as_float_array(rows, "epoch")
    total = _as_float_array(rows, "total")

    if save_path is None:
        save_path = root_dir / "loss_history.pdf"

    fig, ax_loss = plt.subplots(figsize=(6.6, 4.6))

    ax_loss.semilogy(epochs, _safe_for_log(total), linewidth=2.0, label="total")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training History")
    ax_loss.set_xlabel("Epoch")
    ax_loss.grid(alpha=0.25)
    ax_loss.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def evaluate_burgers_run(
    path: Path,
    target_t: float = 0.6,
    checkpoint_epoch: int | None = None,
):
    comparison_path = plot_burgers_slice(
        root_dir=path,
        target_t=target_t,
        checkpoint_epoch=checkpoint_epoch,
    )
    history_path = plot_history(path)
    return {
        "comparison_path": comparison_path,
        "history_path": history_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Plot PINN evaluation figures.")
    parser.add_argument("--path", required=True, help="Run directory containing config/history/checkpoints.")
    parser.add_argument("--mode", default="burgers", choices=["burgers"])
    parser.add_argument("--time", type=float, default=0.6, help="Target time slice to plot.")
    parser.add_argument("--checkpoint_epoch", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.path)
    if args.mode != "burgers":
        raise ValueError("Unsupported mode {}".format(args.mode))
    outputs = evaluate_burgers_run(
        path=path,
        target_t=args.time,
        checkpoint_epoch=args.checkpoint_epoch,
    )
    for name, output_path in outputs.items():
        print("{}={}".format(name, output_path))


if __name__ == "__main__":
    main()
