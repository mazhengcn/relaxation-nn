import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ml_collections import ConfigDict

import advection
import burgers
import euler
import swe
import utils

DEVICE = torch.device("cuda:0")


# model = advection.AdvectionNet(ConfigDict(config["NetConfig"])).to(DEVICE)
def evaluate(model, root_path: str):
    root_dir = Path(root_path)
    json_path = root_dir / "config.json"
    slice_path = root_dir / "slice_pool"
    if not slice_path.exists():
        slice_path.mkdir()
    model_dir = root_dir / "model_state_dict"

    if not slice_path.exists():
        slice_path.mkdir()
    with open(json_path, "r", encoding="utf8") as jp:
        config = json.load(jp)
    if model == "advection":
        testdata = np.load(config["DataConfig"]["testdata_path"])
        x_test, q_test = testdata[:, 0:2], testdata[:, 2:3]
        x = x_test[0:500, 1:2]
        model = advection.AdvectionNet(ConfigDict(config["NetConfig"])).to(DEVICE)
        for j in range(30000, 30001):
            model_path = model_dir / "model_{:02d}".format(j)
            model.load_state_dict(torch.load(model_path))
            for i in range(11):
                t = round(x_test[i * 500, 0], 1)
                x_part = x_test[i * 500 : i * 500 + 500, :]
                q_part = q_test[i * 500 : i * 500 + 500, :]
                x_part = torch.tensor(x_part, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    q_pred = model(x_part)
                q_pred = utils.to_numpy(q_pred)
                plt.figure()
                plt.xlabel("x")
                plt.ylabel("u")
                plt.plot(x, q_part, label="clawpack")
                plt.plot(x, q_pred, label="resnet10")
                plt.legend()
                plt.savefig(
                    slice_path / "epoch_{}_t_{}.png".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
    elif model == "burgers":
        testdata = np.load(config["DataConfig"]["testdata_path"])
        x_test, q_test = testdata[:, 0:2], testdata[:, 2:3]
        mx = 1000
        x = x_test[0:mx, 1:2]
        model = burgers.BurgersNet(ConfigDict(config["NetConfig"])).to(DEVICE)
        for j in range(300000, 300001):
            model_path = model_dir / "model_{:02d}".format(j)
            model.load_state_dict(torch.load(model_path))
            for i in range(11):
                t = round(x_test[i * mx, 0], 2)
                x_part = x_test[i * mx : i * mx + mx, :]
                q_part = q_test[i * mx : i * mx + mx, :]
                x_part = torch.tensor(x_part, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    q_pred = model(x_part)
                q_pred = utils.to_numpy(q_pred)
                plt.figure()
                plt.xlabel("x")
                plt.ylabel("u")
                plt.plot(x, q_part, label="clawpack")
                plt.plot(x, q_pred, label="DNN")
                plt.legend()
                plt.savefig(
                    slice_path / "epoch_{}_t_{}.png".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
    elif model == "swe":
        testdata = np.load(config["DataConfig"]["testdata_path"])
        x_test, q_test = testdata[:, 0:2], testdata[:, 2:4]
        mx = 1000
        x = x_test[0:mx, 1:2]
        model = swe.SweNet(ConfigDict(config["NetConfig"])).to(DEVICE)
        for j in range(300000, 300001):
            model_path = model_dir / "model_{:02d}".format(j)
            model.load_state_dict(torch.load(model_path))
            for i in range(11):
                t = round(x_test[i * mx, 0], 1)
                x_part = x_test[i * mx : i * mx + mx, :]
                q_part = q_test[i * mx : i * mx + mx, :]
                x_part = torch.tensor(x_part, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    q_pred = model(x_part)
                q_pred = utils.to_numpy(q_pred)
                fig, (ax1, ax2) = plt.subplots(2, 1)
                fig.suptitle("h and u of Clawpack")
                ax1.set_ylabel("h")
                ax1.plot(x, q_part[:, 0:1], label="true")
                ax1.plot(
                    x, q_pred[:, 0:1], "--o", label="pred", markevery=10, markersize=2
                )
                ax2.set_xlabel("x")
                ax2.set_ylabel("u")
                ax2.plot(x, q_part[:, 1:2], label="true")
                ax2.plot(
                    x, q_pred[:, 1:2], "--o", label="pred", markevery=10, markersize=2
                )
                plt.legend()
                plt.savefig(
                    slice_path / "epoch_{}_t_{}.png".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
    elif model == "euler":
        testdata = np.load(config["DataConfig"]["testdata_path"])
        x_test, q_test = testdata[:, 0:2], testdata[:, 2:5]
        mx = 1000
        x = x_test[0:mx, 1:2]
        model = euler.EulerNet(ConfigDict(config["NetConfig"])).to(DEVICE)
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
                q_pred = utils.to_numpy(q_pred)
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
    else:
        raise ValueError("model not implemented")


if __name__ == "__main__":
    evaluate(
        model="euler",
        root_path="/nfs/my/Origin/OriginPINN/_output/euler/lax/2023-11-07T04-54-33",
    )
