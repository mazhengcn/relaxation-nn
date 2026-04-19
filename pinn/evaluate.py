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

from pinn.model import burgers, burgers_viscid, euler, swe
from shared.path_utils import resolve_repo_path
from shared.runtime import DEVICE

MODEL_DICT = {
    "burgers": burgers.BurgersNet,
    "burgers_viscid": burgers_viscid.BurgersNet,
    "swe": swe.SweNet,
    "euler": euler.EulerNet,
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


def _mode_group(mode: str):
    if mode in {"burgers", "burgers_viscid"}:
        return "burgers"
    return mode


def _resolve_mode(requested_mode: str, config_model: str):
    if requested_mode == "auto":
        return config_model
    if _mode_group(requested_mode) != _mode_group(config_model):
        raise ValueError(
            "Requested mode {} does not match run config model {}".format(
                requested_mode, config_model
            )
        )
    return requested_mode


def _prepare_slice_data(
    root_dir: Path,
    target_t: float,
    checkpoint_epoch: int | None,
):
    model, config, epoch = load_model(root_dir, checkpoint_epoch)
    testdata = np.load(resolve_repo_path(config["DataConfig"]["testdata_path"]))
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
        ax.plot(
            x_values,
            q_pred[:, index],
            "--",
            linewidth=2.0,
            label=plot_label,
        )
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


def _compute_true_flux(mode: str, q_true: np.ndarray, config: dict):
    if mode in {"burgers", "burgers_viscid"}:
        return 0.5 * q_true**2, ["f"]

    if mode == "swe":
        gravity = float(config["NetConfig"].get("gravity", 1.0))
        height = q_true[:, 0:1]
        velocity = q_true[:, 1:2]
        flux = np.concatenate(
            [
                height * velocity,
                height * velocity**2 + 0.5 * gravity * height**2,
            ],
            axis=1,
        )
        return flux, ["hu", "hu^2 + gh^2/2"]

    if mode == "euler":
        gamma = float(config["NetConfig"].get("gamma", 1.4))
        rho = q_true[:, 0:1]
        velocity = q_true[:, 1:2]
        pressure = q_true[:, 2:3]
        energy = pressure / (gamma - 1.0) + 0.5 * rho * velocity**2
        flux = np.concatenate(
            [
                rho * velocity,
                rho * velocity**2 + pressure,
                velocity * (energy + pressure),
            ],
            axis=1,
        )
        return flux, ["rho u", "rho u^2 + p", "u(E+p)"]

    raise ValueError("Unsupported mode {}".format(mode))


def _plot_state_slice(mode: str, root_dir: Path, slice_data: dict, save_path: Path | None):
    actual_t = slice_data["actual_t"]
    epoch = slice_data["epoch"]
    if save_path is None:
        save_path = root_dir / "comparison_t_{}.pdf".format(_format_time_token(actual_t))

    state_labels = {
        "burgers": ["u"],
        "burgers_viscid": ["u"],
        "swe": ["h", "u"],
        "euler": ["rho", "u", "p"],
    }
    title_labels = {
        "burgers": "Burgers",
        "burgers_viscid": "Viscous Burgers",
        "swe": "SWE",
        "euler": "Euler",
    }
    return _plot_components(
        x_values=slice_data["x_values"],
        q_true=slice_data["q_true"],
        q_pred=slice_data["q_pred"],
        component_labels=state_labels[mode],
        title="{} at t={} (epoch {})".format(
            title_labels[mode], _format_time_token(actual_t), epoch
        ),
        save_path=save_path,
        plot_label=slice_data["config"].get("plot_label", "PINN"),
    )


def _plot_flux_slice(mode: str, root_dir: Path, slice_data: dict, save_path: Path | None):
    actual_t = slice_data["actual_t"]
    epoch = slice_data["epoch"]
    if save_path is None:
        save_path = root_dir / "flux_t_{}.pdf".format(_format_time_token(actual_t))

    flux_true, flux_labels = _compute_true_flux(
        mode, slice_data["q_true"], slice_data["config"]
    )
    return _plot_components(
        x_values=slice_data["x_values"],
        q_true=flux_true,
        q_pred=slice_data["flux_pred"],
        component_labels=flux_labels,
        title="{} flux at t={} (epoch {})".format(
            mode, _format_time_token(actual_t), epoch
        ),
        save_path=save_path,
        plot_label=slice_data["config"].get("plot_label", "PINN"),
    )


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


def evaluate_run(
    path: Path,
    mode: str = "auto",
    target_t: float = 0.6,
    checkpoint_epoch: int | None = None,
):
    config = load_config(path)
    resolved_mode = _resolve_mode(mode, config["model"])
    slice_data = _prepare_slice_data(path, target_t, checkpoint_epoch)

    outputs = {
        "comparison_path": _plot_state_slice(resolved_mode, path, slice_data, None),
        "history_path": plot_history(path),
    }
    if resolved_mode in {"swe", "euler"}:
        outputs["flux_path"] = _plot_flux_slice(resolved_mode, path, slice_data, None)
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Plot PINN evaluation figures.")
    parser.add_argument("--path", required=True, help="Run directory containing config/history/checkpoints.")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "burgers", "burgers_viscid", "swe", "euler"],
    )
    parser.add_argument("--time", type=float, default=0.6, help="Target time slice to plot.")
    parser.add_argument("--checkpoint_epoch", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.path)
    outputs = evaluate_run(
        path=path,
        mode=args.mode,
        target_t=args.time,
        checkpoint_epoch=args.checkpoint_epoch,
    )
    for name, output_path in outputs.items():
        print("{}={}".format(name, output_path))


if __name__ == "__main__":
    main()
