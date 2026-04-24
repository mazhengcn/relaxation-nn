import argparse
import csv
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn.evaluate import _prepare_slice_data as prepare_pinn_slice
from pinn.evaluate import load_model
from shared.path_utils import resolve_repo_path
from shared.plot_family_comparison import PINN_STYLE
from shared.plot_family_loss import _load_history, _variation_band
from shared.runtime import DEVICE

LOSS_TERM_STYLES = {
    "res_loss": {
        "label": r"$\mathcal{L}_{\mathrm{PDE}}$",
        "color": PINN_STYLE["color"],
    },
    "u_ic": {
        "label": r"$\mathcal{L}_{\mathrm{IC}}$",
        "color": "#F59E0B",
    },
    "u_bc": {
        "label": r"$\mathcal{L}_{\mathrm{BC}}$",
        "color": "#8B5CF6",
    },
}


def _line_style(color: str) -> dict:
    return {
        "color": color,
        "linewidth": 1.0,
        "alpha": 0.88,
    }


def _format_time_token(value: float) -> str:
    return "{:.3f}".format(float(value)).rstrip("0").rstrip(".")


def _marker_stride(num_points: int, requested_stride: int | None = None) -> int:
    if requested_stride is not None:
        return max(1, int(requested_stride))
    return max(1, int(np.ceil(num_points / 32.0)))


def plot_state_slice(
    root_dir: Path,
    target_t: float,
    save_path: Path,
    checkpoint_epoch: int | None,
    figure_size: tuple[float, float],
    marker_stride: int | None = 48,
):
    slice_data = prepare_pinn_slice(root_dir, target_t, checkpoint_epoch)
    x_values = slice_data["x_values"]
    q_true = slice_data["q_true"][:, 0]
    q_pred = slice_data["q_pred"][:, 0]

    stride = _marker_stride(len(x_values), marker_stride)
    marker_indices = np.arange(0, len(x_values), stride, dtype=np.int64)
    if marker_indices[-1] != len(x_values) - 1:
        marker_indices = np.unique(
            np.concatenate([marker_indices, np.asarray([len(x_values) - 1])])
        )

    fig, ax = plt.subplots(figsize=figure_size)
    ax.plot(
        x_values,
        q_true,
        color="black",
        linewidth=0.9,
        linestyle="--",
        alpha=0.95,
        label="Reference",
    )
    ax.plot(
        x_values[marker_indices],
        q_pred[marker_indices],
        linestyle="None",
        marker=PINN_STYLE["marker"],
        color=PINN_STYLE["color"],
        markersize=5.0,
        markeredgewidth=0.9,
        label="PINN",
    )
    ax.set_xlim(float(np.min(x_values)), float(np.max(x_values)))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u$", rotation=0, labelpad=18)
    ax.yaxis.set_label_coords(-0.14, 0.5)
    ax.legend(fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _format_prediction_grid(root_dir: Path, checkpoint_epoch: int | None):
    model, config, epoch = load_model(root_dir, checkpoint_epoch)
    testdata = np.load(resolve_repo_path(config["DataConfig"]["testdata_path"]))
    x_test = testdata[:, 0:2]
    q_true = testdata[:, 2:3]

    x_tensor = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        q_pred = model(x_tensor).detach().cpu().numpy()

    order = np.lexsort((x_test[:, 1], x_test[:, 0]))
    ordered_x = x_test[order]
    ordered_true = q_true[order, 0]
    ordered_pred = q_pred[order, 0]

    unique_t = np.unique(ordered_x[:, 0])
    unique_x = np.unique(ordered_x[:, 1])
    num_t = len(unique_t)
    num_x = len(unique_x)

    true_grid = ordered_true.reshape(num_t, num_x)
    pred_grid = ordered_pred.reshape(num_t, num_x)
    error_grid = np.abs(pred_grid - true_grid)

    return {
        "config": config,
        "epoch": epoch,
        "time_values": unique_t,
        "x_values": unique_x,
        "error_grid": error_grid,
    }


def plot_loss_terms(
    root_dir: Path,
    save_path: Path,
    smooth_window: int,
    band_window: int,
    band_scale: float,
    band_alpha: float,
    figure_size: tuple[float, float],
):
    history = _load_history(root_dir)
    history_path = root_dir / "history.csv"
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    fig, ax = plt.subplots(figsize=figure_size)

    for key, style_info in LOSS_TERM_STYLES.items():
        values = np.asarray(
            [float(row[key]) for row in rows],
            dtype=np.float64,
        )
        center, lower, upper = _variation_band(
            values,
            smooth_window,
            band_window,
            band_scale,
        )
        line_style = _line_style(style_info["color"])
        ax.fill_between(
            history["epochs"],
            lower,
            upper,
            color=style_info["color"],
            alpha=band_alpha,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            history["epochs"],
            center,
            label=style_info["label"],
            zorder=2,
            **line_style,
        )

    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\mathcal{L}$", rotation=0, labelpad=18)
    ax.yaxis.set_label_coords(-0.11, 0.5)
    ax.set_yscale("log")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(5, 5))
    ax.legend(fontsize=10.0, framealpha=0.95, loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_abs_error_map(
    root_dir: Path,
    save_path: Path,
    checkpoint_epoch: int | None,
    figure_size: tuple[float, float],
    cmap: str,
    gamma: float,
):
    grid = _format_prediction_grid(root_dir, checkpoint_epoch)
    error_grid = grid["error_grid"]
    x_values = grid["x_values"]
    time_values = grid["time_values"]

    norm = mcolors.PowerNorm(
        gamma=max(float(gamma), 1e-6),
        vmin=0.0,
        vmax=float(np.max(error_grid)),
    )

    fig, ax = plt.subplots(figsize=figure_size)
    image = ax.imshow(
        error_grid,
        origin="lower",
        aspect="auto",
        extent=(
            float(np.min(x_values)),
            float(np.max(x_values)),
            float(np.min(time_values)),
            float(np.max(time_values)),
        ),
        cmap=cmap,
        norm=norm,
        interpolation="bicubic",
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$", rotation=0, labelpad=18)
    ax.yaxis.set_label_coords(-0.14, 0.5)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.ax.tick_params(labelsize=10.0)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_burgers_sine_pinn_paper(
    run_path: str | Path,
    checkpoint_epoch: int | None = None,
    smooth_window: int = 21,
    band_window: int = 31,
    band_scale: float = 0.85,
    band_alpha: float = 0.12,
    figure_size: tuple[float, float] = (4.45, 3.2),
    cmap: str = "magma",
    gamma: float = 0.75,
):
    root_dir = resolve_repo_path(run_path)
    figure_dir = resolve_repo_path(Path("_figures") / "burgers" / "sine")
    figure_dir.mkdir(parents=True, exist_ok=True)

    loss_path = figure_dir / "burgers_sine_pinn_loss_terms.pdf"
    error_path = figure_dir / "burgers_sine_pinn_abs_error.pdf"

    plot_loss_terms(
        root_dir=root_dir,
        save_path=loss_path,
        smooth_window=smooth_window,
        band_window=band_window,
        band_scale=band_scale,
        band_alpha=band_alpha,
        figure_size=figure_size,
    )
    plot_abs_error_map(
        root_dir=root_dir,
        save_path=error_path,
        checkpoint_epoch=checkpoint_epoch,
        figure_size=figure_size,
        cmap=cmap,
        gamma=gamma,
    )

    return {
        "loss_pdf": loss_path,
        "error_pdf": error_path,
        "figure_size": figure_size,
        "cmap": cmap,
        "gamma": gamma,
    }


def plot_burgers_sine_pinn_slices(
    run_path: str | Path,
    times: list[float],
    checkpoint_epoch: int | None = None,
    figure_size: tuple[float, float] = (3.8, 3.1),
    marker_stride: int | None = 48,
):
    root_dir = resolve_repo_path(run_path)
    figure_dir = resolve_repo_path(Path("_figures") / "burgers" / "sine")
    figure_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for target_t in times:
        file_path = figure_dir / "burgers_sine_pinn_t_{}_u.pdf".format(
            _format_time_token(target_t)
        )
        outputs.append(
            plot_state_slice(
                root_dir=root_dir,
                target_t=target_t,
                save_path=file_path,
                checkpoint_epoch=checkpoint_epoch,
                figure_size=figure_size,
                marker_stride=marker_stride,
            )
        )
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot publication-style loss and absolute-error figures for Burgers sine PINN."
    )
    parser.add_argument(
        "--run_path",
        default="_output/pinn/burgers/sine/2026-04-14T12-53-45",
        help="PINN run directory for the Burgers sine case.",
    )
    parser.add_argument("--checkpoint_epoch", type=int, default=None)
    parser.add_argument("--smooth_window", type=int, default=21)
    parser.add_argument("--band_window", type=int, default=31)
    parser.add_argument("--band_scale", type=float, default=0.85)
    parser.add_argument("--band_alpha", type=float, default=0.12)
    parser.add_argument("--fig_width", type=float, default=4.45)
    parser.add_argument("--fig_height", type=float, default=3.2)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--gamma", type=float, default=0.75)
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = plot_burgers_sine_pinn_paper(
        run_path=args.run_path,
        checkpoint_epoch=args.checkpoint_epoch,
        smooth_window=args.smooth_window,
        band_window=args.band_window,
        band_scale=args.band_scale,
        band_alpha=args.band_alpha,
        figure_size=(args.fig_width, args.fig_height),
        cmap=args.cmap,
        gamma=args.gamma,
    )
    print("loss_pdf={}".format(outputs["loss_pdf"]))
    print("error_pdf={}".format(outputs["error_pdf"]))
    print("figure_size={}".format(outputs["figure_size"]))
    print("cmap={}".format(outputs["cmap"]))
    print("gamma={}".format(outputs["gamma"]))


if __name__ == "__main__":
    main()
