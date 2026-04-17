import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
from ml_collections import ConfigDict
from pinns.model import burgers
from pinns_utils_torch.runtime import DEVICE

MODEL_DICT = {
    "burgers": burgers.BurgersNet,
}


def to_numpy(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.detach().cpu().numpy()
    if isinstance(inputs, np.ndarray):
        return inputs
    raise TypeError("Expected torch.Tensor or np.ndarray, got {}".format(type(inputs)))


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


def _format_time_token(value: float):
    return "{:.3f}".format(float(value)).rstrip("0").rstrip(".")


def _plot_components(
    x_values,
    q_true,
    q_pred,
    component_labels,
    title,
    save_path: Path,
    plot_label: str,
):
    num_components = q_true.shape[1]
    fig, axes = plt.subplots(num_components, 1, figsize=(6.6, 2.6 * num_components))
    if num_components == 1:
        axes = [axes]

    for index, ax in enumerate(axes):
        ax.plot(x_values, q_true[:, index], linewidth=2.0, label="Reference")
        ax.plot(x_values, q_pred[:, index], "--", linewidth=2.0, label=plot_label)
        ax.set_ylabel(component_labels[index])
        ax.grid(alpha=0.25)
        if index == 0:
            ax.legend()

    axes[-1].set_xlabel("x")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _prepare_slice_data(root_dir: Path, target_t: float, checkpoint_epoch: int | None):
    model, config, epoch = load_model(root_dir, checkpoint_epoch)
    testdata = np.load(config["DataConfig"]["testdata_path"])
    x_test, q_test = testdata[:, 0:2], testdata[:, 2 : testdata.shape[1]]
    actual_t, x_part, q_part = select_time_slice(x_test, q_test, target_t)

    order = np.argsort(x_part[:, 1])
    x_sorted = x_part[order]
    q_true = q_part[order]
    x_tensor = torch.tensor(x_sorted, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        q_pred = to_numpy(model(x_tensor))
        flux_pred = to_numpy(model.flux(x_tensor))

    return dict(
        config=config,
        epoch=epoch,
        actual_t=actual_t,
        x_values=x_sorted[:, 1],
        q_true=q_true,
        q_pred=q_pred,
        flux_pred=flux_pred,
    )


def plot_state_slice(root_dir: Path, target_t: float, checkpoint_epoch: int | None):
    slice_data = _prepare_slice_data(root_dir, target_t, checkpoint_epoch)
    save_path = root_dir / "comparison_t_{}.pdf".format(
        _format_time_token(slice_data["actual_t"])
    )
    return _plot_components(
        x_values=slice_data["x_values"],
        q_true=slice_data["q_true"],
        q_pred=slice_data["q_pred"],
        component_labels=["u"],
        title="WPINN Burgers at t={} (epoch {})".format(
            _format_time_token(slice_data["actual_t"]),
            slice_data["epoch"],
        ),
        save_path=save_path,
        plot_label=slice_data["config"].get("plot_label", "WPINN"),
    )


def plot_flux_slice(root_dir: Path, target_t: float, checkpoint_epoch: int | None):
    slice_data = _prepare_slice_data(root_dir, target_t, checkpoint_epoch)
    save_path = root_dir / "flux_t_{}.pdf".format(
        _format_time_token(slice_data["actual_t"])
    )
    return _plot_components(
        x_values=slice_data["x_values"],
        q_true=0.5 * slice_data["q_true"] ** 2,
        q_pred=slice_data["flux_pred"],
        component_labels=["f"],
        title="WPINN Burgers flux at t={} (epoch {})".format(
            _format_time_token(slice_data["actual_t"]),
            slice_data["epoch"],
        ),
        save_path=save_path,
        plot_label=slice_data["config"].get("plot_label", "WPINN"),
    )


def load_history(history_path: Path):
    with open(history_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("History file {} is empty".format(history_path))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("--target_t", type=float, default=0.5)
    parser.add_argument("--checkpoint_epoch", type=int, default=None)
    args = parser.parse_args()

    state_path = plot_state_slice(args.root_dir, args.target_t, args.checkpoint_epoch)
    flux_path = plot_flux_slice(args.root_dir, args.target_t, args.checkpoint_epoch)
    print(state_path)
    print(flux_path)


if __name__ == "__main__":
    main()
