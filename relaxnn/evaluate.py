import json
from pathlib import Path

import matplotlib.pyplot as plt
import model.burgers as burgers
import model.euler_v1 as euler_v1
import model.euler_v2 as euler_v2
import model.euler_v3 as euler_v3
import model.swe_v1 as swe_v1
import model.swe_v2 as swe_v2
import numpy as np
import torch
from ml_collections import ConfigDict

DEVICE = torch.device("cuda:0")

def to_numpy(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.detach().cpu().numpy()
    elif isinstance(inputs, np.ndarray):
        return inputs
    else:
        raise TypeError(
            "Unknown type of input, expected torch.Tensor or "
            "np.ndarray, but got {}".format(type(input))
        )


def evaluate(mode, path: Path):
    root_dir = path
    json_path = root_dir / "config.json"
    slice_path = root_dir / "slice_pool"
    model_dir = root_dir / "model_state_dict"
    if not slice_path.exists():
        slice_path.mkdir()
    with open(json_path, "r", encoding="utf8") as jp:
        config = json.load(jp)
    if mode == "burgers":
        mx = 1000
        testdata = np.load(config["DataConfig"]["testdata_path"])
        x_test, q_test = testdata[:, 0:2], testdata[:, 2:3]
        x = x_test[0:mx, 1:2]
        model = burgers.BurgersNet(ConfigDict(config["NetConfig"])).to(DEVICE)
        for j in range(300000, 300001):
            model_path = model_dir / "model_{:02d}".format(j)
            model.load_state_dict(torch.load(model_path))
            for i in range(11):
                t = round(x_test[i * mx, 0], 1)
                x_part = x_test[i * mx : i * mx + mx, :]
                q_part = q_test[i * mx : i * mx + mx, :]
                flux = 0.5 * q_part**2
                x_part = torch.tensor(x_part, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    q_pred = model(x_part)
                    flux_pred = model.flux(x_part)
                q_pred = to_numpy(q_pred)
                flux_pred = to_numpy(flux_pred)
                plt.figure()
                plt.suptitle("burgers equation at t={}".format(t))
                plt.xlabel("x")
                plt.ylabel("u")
                plt.plot(x, q_part, label="clawpack")
                plt.plot(
                    x, q_pred, "--o", label="Relaxation", markevery=10, markersize=3
                )
                plt.legend()
                plt.savefig(
                    slice_path / "epoch_{}_t_{}.png".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
                plt.figure()
                plt.xlabel("x")
                plt.ylabel("F")
                plt.plot(x, flux, label="clawpack")
                plt.plot(
                    x, flux_pred, "--o", label="Relaxation", markevery=10, markersize=3
                )
                plt.legend()
                plt.savefig(
                    slice_path / "F_epoch_{}_t_{}.png".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
    elif mode == "swe_v1":
        testdata = np.load(config["DataConfig"]["testdata_path"])
        x_test, q_test = testdata[:, 0:2], testdata[:, 2:4]
        mx = 1000
        x = x_test[0:mx, 1:2]
        model = swe_v1.SweNet(ConfigDict(config["NetConfig"])).to(DEVICE)
        for j in range(600000, 600001):
            model_path = model_dir / "model_{:02d}".format(j)
            model.load_state_dict(torch.load(model_path))
            for i in range(11):
                t = round(x_test[i * mx, 0], 1)
                x_part = x_test[i * mx : i * mx + mx, :]
                q_part = q_test[i * mx : i * mx + mx, :]
                flux1 = q_part[:, 0:1] * q_part[:, 1:2]
                flux2 = q_part[:, 0:1] * q_part[:, 1:2] ** 2 + 0.5 * q_part[:, 0:1] ** 2
                x_part = torch.tensor(x_part, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    q_pred = model(x_part)
                    flux_pred = model.flux(x_part)
                q_pred = to_numpy(q_pred)
                flux_pred = to_numpy(flux_pred)
                fig, (ax1, ax2) = plt.subplots(2, 1)
                fig.suptitle("h and M of Clawpack")
                ax1.set_ylabel("h")
                ax1.plot(x, q_part[:, 0:1], label="true")
                ax1.plot(
                    x, q_pred[:, 0:1], "--o", label="pred", markevery=10, markersize=3
                )
                ax2.set_xlabel("x")
                ax2.set_ylabel("M")
                ax2.plot(x, q_part[:, 1:2], label="true")
                ax2.plot(
                    x, q_pred[:, 1:2], "--o", label="pred", markevery=10, markersize=3
                )
                plt.legend()
                plt.savefig(
                    slice_path / "epoch_{}_t_{}.png".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()

                fig, (ax1, ax2) = plt.subplots(2, 1)
                fig.suptitle("flux1 and flux2 ")
                ax1.set_ylabel("F1")
                ax1.plot(x, flux1, label="true")
                ax1.plot(
                    x,
                    flux_pred[:, 0:1],
                    "--o",
                    label="pred",
                    markevery=10,
                    markersize=3,
                )
                ax2.set_xlabel("x")
                ax2.set_ylabel("F2")
                ax2.plot(x, flux2, label="true")
                ax2.plot(
                    x,
                    flux_pred[:, 1:2],
                    "--o",
                    label="pred",
                    markevery=10,
                    markersize=3,
                )
                plt.legend()
                plt.savefig(
                    slice_path / "F_epoch_{}_t_{}.png".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
    elif mode == "swe_v2":
        testdata = np.load(config["DataConfig"]["testdata_path"])
        x_test, q_test = testdata[:, 0:2], testdata[:, 2:4]
        mx = 1000
        x = x_test[0:mx, 1:2]
        model = swe_v2.SweNet(ConfigDict(config["NetConfig"])).to(DEVICE)
        for j in range(600000, 600001):
            model_path = model_dir / "model_{:02d}".format(j)
            model.load_state_dict(torch.load(model_path))
            for i in range(11):
                t = round(x_test[i * mx, 0], 1)
                x_part = x_test[i * mx : i * mx + mx, :]
                q_part = q_test[i * mx : i * mx + mx, :]
                flux = q_part[:, 1:2] ** 2 * q_part[:, 0:1] + 0.5 * q_part[:, 0:1] ** 2
                x_part = torch.tensor(x_part, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    q_pred = model(x_part)
                    flux_pred = model.flux(x_part)
                q_pred = to_numpy(q_pred)
                flux_pred = to_numpy(flux_pred)
                fig, (ax1, ax2) = plt.subplots(2, 1)
                fig.suptitle("h and u of Clawpack")
                ax1.set_ylabel("h")
                ax1.plot(x, q_part[:, 0:1], label="true")
                ax1.plot(
                    x, q_pred[:, 0:1], "--o", label="pred", markevery=10, markersize=3
                )
                ax2.set_xlabel("x")
                ax2.set_ylabel("u")
                ax2.plot(x, q_part[:, 1:2], label="true")
                ax2.plot(
                    x, q_pred[:, 1:2], "--o", label="pred", markevery=10, markersize=3
                )
                plt.legend()
                plt.savefig(
                    slice_path / "epoch_{}_t_{}.png".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()

                plt.figure()
                plt.xlabel("x")
                plt.ylabel("F")
                plt.plot(x, flux, label="clawpack")
                plt.plot(
                    x, flux_pred, "--o", label="Relaxation", markevery=10, markersize=2
                )
                plt.legend()
                plt.savefig(
                    slice_path / "F_epoch_{}_t_{}.png".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
    elif mode == "euler_v1":
        testdata = np.load(config["DataConfig"]["testdata_path"])
        x_test, q_test = testdata[:, 0:2], testdata[:, 2:5]
        mx = 1000
        x = x_test[0:mx, 1:2]
        model = euler_v1.EulerNet(ConfigDict(config["NetConfig"])).to(DEVICE)
        for j in range(600000, 600001):
            model_path = model_dir / "model_{:02d}".format(j)
            model.load_state_dict(torch.load(model_path))
            for i in range(11):
                t = round(x_test[i * mx, 0], 3)
                x_part = x_test[i * mx : i * mx + mx, :]
                q_part = q_test[i * mx : i * mx + mx, :]
                x_part = torch.tensor(x_part, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    q_pred = model(x_part)
                    flux_pred = model.flux(x_part)
                q_pred = to_numpy(q_pred)
                flux_pred = to_numpy(flux_pred)
                rho = q_part[:, 0:1]
                velocity = q_part[:, 1:2]
                pressure = q_part[:, 2:3]
                energy = pressure / 0.4 + 0.5 * rho * velocity**2
                flux1_true = rho * velocity
                flux2_true = rho * velocity**2 + pressure
                flux3_true = velocity * (energy + pressure)
                flux_pred = to_numpy(flux_pred)
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
                fig.suptitle("euler equation at t={}".format(t))
                ax1.set_ylabel("rho")
                ax1.plot(x, q_part[:, 0:1], label="true")
                ax1.plot(
                    x, q_pred[:, 0:1], "--o", label="pred", markevery=10, markersize=2
                )
                ax2.set_ylabel("velocity")
                ax2.plot(x, q_part[:, 1:2], label="true")
                ax2.plot(
                    x, q_pred[:, 1:2], "--o", label="pred", markevery=10, markersize=2
                )
                ax3.set_xlabel("x")
                ax3.set_ylabel("pressure")
                ax3.plot(x, q_part[:, 2:3], label="true")
                ax3.plot(
                    x, q_pred[:, 2:3], "--o", label="pred", markevery=10, markersize=2
                )
                plt.legend()
                plt.savefig(
                    slice_path / "epoch_{}_t_{}.png".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()

                fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
                ax1.set_ylabel("flux1")
                ax1.plot(x, flux1_true, label="true")
                ax1.plot(
                    x,
                    flux_pred[:, 0:1],
                    "--o",
                    label="pred",
                    markevery=10,
                    markersize=2,
                )
                ax2.set_ylabel("flux2")
                ax2.plot(x, flux2_true, label="true")
                ax2.plot(
                    x,
                    flux_pred[:, 1:2],
                    "--o",
                    label="pred",
                    markevery=10,
                    markersize=2,
                )
                ax3.set_xlabel("x")
                ax3.set_ylabel("flux3")
                ax3.plot(x, flux3_true, label="true")
                ax3.plot(
                    x,
                    flux_pred[:, 2:3],
                    "--o",
                    label="pred",
                    markevery=10,
                    markersize=2,
                )
                plt.legend()
                plt.savefig(
                    slice_path / "F_epoch_{}_t_{}.png".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
    elif mode == "euler_v2":
        testdata = np.load(config["DataConfig"]["testdata_path"])
        x_test, q_test = testdata[:, 0:2], testdata[:, 2:5]
        mx = 1000
        x = x_test[0:mx, 1:2]
        model = euler_v2.EulerNet(ConfigDict(config["NetConfig"])).to(DEVICE)
        for j in range(600000, 600001):
            model_path = model_dir / "model_{:02d}".format(j)
            model.load_state_dict(torch.load(model_path))
            for i in range(11):
                t = round(x_test[i * mx, 0], 3)
                x_part = x_test[i * mx : i * mx + mx, :]
                q_part = q_test[i * mx : i * mx + mx, :]
                x_part = torch.tensor(x_part, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    q_pred = model(x_part)
                    flux_pred = model.flux(x_part)
                q_pred = to_numpy(q_pred)
                rho = q_part[:, 0:1]
                velocity = q_part[:, 1:2]
                pressure = q_part[:, 2:3]
                energy = pressure / 0.4 + 0.5 * rho * velocity**2
                F1 = rho * velocity**2 + pressure
                F2 = velocity * (energy + pressure)
                flux_pred = to_numpy(flux_pred)
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
                fig.suptitle("euler equation at t={}".format(t))
                ax1.set_ylabel("rho")
                ax1.plot(x, q_part[:, 0:1], label="true")
                ax1.plot(
                    x, q_pred[:, 0:1], "--o", label="pred", markevery=10, markersize=3
                )
                ax2.set_ylabel("velocity")
                ax2.plot(x, q_part[:, 1:2], label="true")
                ax2.plot(
                    x, q_pred[:, 1:2], "--o", label="pred", markevery=10, markersize=3
                )
                ax3.set_xlabel("x")
                ax3.set_ylabel("pressure")
                ax3.plot(x, q_part[:, 2:3], label="true")
                ax3.plot(
                    x, q_pred[:, 2:3], "--o", label="pred", markevery=10, markersize=3
                )
                plt.legend()
                plt.savefig(
                    slice_path / "epoch_{}_t_{}.png".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()

                fig, (ax1, ax2) = plt.subplots(2, 1)
                ax1.set_ylabel("flux1")
                ax1.plot(x, F1, label="true")
                ax1.plot(
                    x,
                    flux_pred[:, 0:1],
                    "--o",
                    label="pred",
                    markevery=10,
                    markersize=3,
                )
                ax2.set_xlabel("x")
                ax2.set_ylabel("flux2")
                ax2.plot(x, F2, label="true")
                ax2.plot(
                    x,
                    flux_pred[:, 1:2],
                    "--o",
                    label="pred",
                    markevery=10,
                    markersize=3,
                )
                plt.legend()
                plt.savefig(
                    slice_path / "F_epoch_{}_t_{}.png".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
    elif mode == "euler_v3":
        sizes = 1000
        testdata = np.load(config["DataConfig"]["testdata_path"])
        x_test, q_test = testdata[:, 0:2], testdata[:, 2:5]
        x = x_test[0:sizes, 1:2]
        model = euler_v3.EulerNet(ConfigDict(config["NetConfig"])).to(DEVICE)
        for j in range(600000, 600001):
            model_path = model_dir / "model_{:02d}".format(j)
            model.load_state_dict(torch.load(model_path))
            for i in range(11):
                t = round(x_test[i * sizes, 0], 3)
                x_part = x_test[i * sizes : i * sizes + sizes, :]
                q_part = q_test[i * sizes : i * sizes + sizes, :]
                x_part = torch.tensor(x_part, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    q_pred = model(x_part)
                    flux_pred = model.flux(x_part)
                q_pred = to_numpy(q_pred)
                rho = q_part[:, 0:1]
                velocity = q_part[:, 1:2]
                pressure = q_part[:, 2:3]
                energy = pressure / 0.4 + 0.5 * rho * velocity**2
                F = velocity * (energy + pressure)
                flux_pred = to_numpy(flux_pred)
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
                fig.suptitle("rho, velocity, pressure  of Clawpack")
                ax1.set_ylabel("rho")
                ax1.plot(x, q_part[:, 0:1], label="true")
                ax1.plot(
                    x, q_pred[:, 0:1], "--o", label="pred", markevery=10, markersize=2
                )
                ax2.set_ylabel("velocity")
                ax2.plot(x, q_part[:, 1:2], label="true")
                ax2.plot(
                    x, q_pred[:, 1:2], "--o", label="pred", markevery=10, markersize=2
                )
                ax3.set_xlabel("x")
                ax3.set_ylabel("pressure")
                ax3.plot(x, q_part[:, 2:3], label="true")
                ax3.plot(
                    x, q_pred[:, 2:3], "--o", label="pred", markevery=10, markersize=2
                )
                plt.legend()
                plt.savefig(
                    slice_path / "epoch_{}_t_{}.png".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()

                plt.figure()
                plt.xlabel("x")
                plt.ylabel("F")
                plt.plot(x, F, label="clawpack")
                plt.plot(
                    x, flux_pred, "--o", label="Relaxation", markevery=10, markersize=2
                )
                plt.legend()
                plt.savefig(
                    slice_path / "F_epoch_{}_t_{}.png".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
    else:
        raise ValueError("other mode have not been implemented")


if __name__ == "__main__":
    evaluate(
        mode="burgers",
        path=Path("./relaxnn/_output/burgers/riemann/2023-11-29T12-01-10"),
    )
