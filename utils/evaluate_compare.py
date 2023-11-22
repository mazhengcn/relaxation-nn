import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ml_collections import ConfigDict

from OriginPINN import burgers as PINN_bg
from OriginPINN import euler as PINN_ev
from OriginPINN import swe as PINN_swe
from OriginRela import burgers as Re_bg
from OriginRela import euler_v1 as Re_ev1
from OriginRela import euler_v2 as Re_ev2
from OriginRela import euler_v3 as Re_ev3
from OriginRela import swe_v1 as Re_swev1
from OriginRela import swe_v2 as Re_swev2
from OriginRela import utils

DEVICE = torch.device("cuda:0")


def evaluate(mode, archive: Path, PINNpath: Path, ReNNpath: Path):
    slice_path = archive / "slice_pool"
    PINN_json_path = PINNpath / "config.json"
    ReNN_json_path = ReNNpath / "config.json"
    PINN_model_dir = PINNpath / "model_state_dict"
    ReNN_model_dir = ReNNpath / "model_state_dict"
    if not slice_path.exists():
        slice_path.mkdir()
    with open(PINN_json_path, "r", encoding="utf8") as jp1:
        PINN_config = json.load(jp1)
    with open(ReNN_json_path, "r", encoding="utf8") as jp2:
        ReNN_config = json.load(jp2)
    if mode == "burgers":
        testdata = np.load(ReNN_config["DataConfig"]["testdata_path"])
        x_test, q_test = testdata[:, 0:2], testdata[:, 2:3]
        mx = 1000
        x = x_test[0:mx, 1:2]
        ReNN_model = Re_bg.BurgersNet(ConfigDict(ReNN_config["NetConfig"])).to(DEVICE)
        PINN_model = PINN_bg.BurgersNet(ConfigDict(PINN_config["NetConfig"])).to(DEVICE)
        for j in range(300000, 300001):
            ReNN_model.load_state_dict(
                torch.load(ReNN_model_dir / "model_{:02d}".format(j))
            )
            PINN_model.load_state_dict(
                torch.load(PINN_model_dir / "model_{:02d}".format(j))
            )
            for i in range(11):
                t = round(x_test[i * mx, 0], 1)
                x_part = x_test[i * mx : (i + 1) * mx, :]
                q_part = q_test[i * mx : (i + 1) * mx, :]
                x_part = torch.tensor(x_part, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    q_pred_ReNN = ReNN_model(x_part)
                    q_pred_PINN = PINN_model(x_part)
                q_pred_ReNN = utils.to_numpy(q_pred_ReNN)
                q_pred_PINN = utils.to_numpy(q_pred_PINN)
                plt.figure()
                plt.suptitle("burgers equation at t={}".format(t))
                plt.xlabel(r"$x$")
                plt.ylabel(r"$u$", rotation=0)
                plt.ticklabel_format(style="sci", axis="x", scilimits=(-1, 2))
                plt.plot(x, q_part, "k", label="reference")
                plt.plot(
                    x,
                    q_pred_ReNN,
                    "ro",
                    label="ReNN",
                    markevery=10,
                    markersize=3,
                    markerfacecolor="none",
                )
                plt.plot(
                    x,
                    q_pred_PINN,
                    "b--x",
                    label="PINN",
                    markevery=10,
                    markersize=3,
                )
                plt.legend()
                plt.savefig(
                    slice_path / "epoch_{}_t_{}.pdf".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
    elif mode == "swe_v1":
        testdata = np.load(ReNN_config["DataConfig"]["testdata_path"])
        x_test, q_test = testdata[:, 0:2], testdata[:, 2:4]
        mx = 1000
        x = x_test[0:mx, 1:2]
        ReNN_model = Re_swev1.SweNet(ConfigDict(ReNN_config["NetConfig"])).to(DEVICE)
        PINN_model = PINN_swe.SweNet(ConfigDict(PINN_config["NetConfig"])).to(DEVICE)
        for j in range(600000, 600001):
            ReNN_model.load_state_dict(
                torch.load(ReNN_model_dir / "model_{:02d}".format(j))
            )
            PINN_model.load_state_dict(
                torch.load(PINN_model_dir / "model_{:02d}".format(j))
            )
            for i in range(11):
                t = round(x_test[i * mx, 0], 1)
                x_part = x_test[i * mx : i * mx + mx, :]
                q_part = q_test[i * mx : i * mx + mx, :]
                x_part = torch.tensor(x_part, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    q_pred_ReNN = ReNN_model(x_part)
                    q_pred_PINN = PINN_model(x_part)
                q_pred_ReNN = utils.to_numpy(q_pred_ReNN)
                q_pred_PINN = utils.to_numpy(q_pred_PINN)
                fig, (ax1, ax2) = plt.subplots(2, 1)
                fig.suptitle("shallow water equation at t={}".format(t))
                ax1.set_ylabel(r"$h$", rotation=0)
                ax1.plot(x, q_part[:, 0:1], "k", label="reference")

                ax1.plot(
                    x,
                    q_pred_ReNN[:, 0:1],
                    "ro",
                    label="ReNN_op1",
                    markevery=10,
                    markersize=3,
                    markerfacecolor="none",
                )
                ax1.plot(
                    x,
                    q_pred_PINN[:, 0:1],
                    "b--x",
                    label="PINN",
                    markevery=10,
                    markersize=3,
                )
                ax1.ticklabel_format(style="sci", axis="x", scilimits=(-1, 2))
                ax2.set_xlabel(r"$x$")
                ax2.set_ylabel(r"$u$", rotation=0)
                ax2.plot(x, q_part[:, 1:2], "k", label="reference")
                ax2.plot(
                    x,
                    q_pred_ReNN[:, 1:2],
                    "ro",
                    label="ReNN_op1",
                    markevery=10,
                    markersize=3,
                    markerfacecolor="none",
                )
                ax2.plot(
                    x,
                    q_pred_PINN[:, 1:2],
                    "b--x",
                    label="PINN",
                    markevery=10,
                    markersize=3,
                )
                ax2.ticklabel_format(style="sci", axis="x", scilimits=(-1, 2))
                plt.legend()
                plt.savefig(
                    slice_path / "op1_epoch_{}_t_{}.pdf".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
    elif mode == "swe_v2":
        testdata = np.load(ReNN_config["DataConfig"]["testdata_path"])
        x_test, q_test = testdata[:, 0:2], testdata[:, 2:4]
        mx = 1000
        x = x_test[0:mx, 1:2]
        ReNN_model = Re_swev2.SweNet(ConfigDict(ReNN_config["NetConfig"])).to(DEVICE)
        PINN_model = PINN_swe.SweNet(ConfigDict(PINN_config["NetConfig"])).to(DEVICE)
        for j in range(600000, 600001):
            ReNN_model.load_state_dict(
                torch.load(ReNN_model_dir / "model_{:02d}".format(j))
            )
            PINN_model.load_state_dict(
                torch.load(PINN_model_dir / "model_{:02d}".format(j))
            )
            for i in range(11):
                t = round(x_test[i * mx, 0], 1)
                x_part = x_test[i * mx : i * mx + mx, :]
                q_part = q_test[i * mx : i * mx + mx, :]
                x_part = torch.tensor(x_part, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    q_pred_ReNN = ReNN_model(x_part)
                    q_pred_PINN = PINN_model(x_part)
                q_pred_ReNN = utils.to_numpy(q_pred_ReNN)
                q_pred_PINN = utils.to_numpy(q_pred_PINN)
                fig, (ax1, ax2) = plt.subplots(2, 1)
                fig.suptitle("shallow water equation at t={}".format(t))
                ax1.set_ylabel(r"$h$", rotation=0)
                ax1.plot(x, q_part[:, 0:1], "k", label="reference")

                ax1.plot(
                    x,
                    q_pred_ReNN[:, 0:1],
                    "ro",
                    label="ReNN_op2",
                    markevery=10,
                    markersize=3,
                    markerfacecolor="none",
                )
                ax1.plot(
                    x,
                    q_pred_PINN[:, 0:1],
                    "b--x",
                    label="PINN",
                    markevery=10,
                    markersize=3,
                )
                ax1.ticklabel_format(style="sci", axis="x", scilimits=(-1, 2))
                ax2.set_xlabel(r"$x$")
                ax2.set_ylabel(r"$u$", rotation=0)
                ax2.plot(x, q_part[:, 1:2], "k", label="reference")
                ax2.plot(
                    x,
                    q_pred_ReNN[:, 1:2],
                    "ro",
                    label="ReNN_op2",
                    markevery=10,
                    markersize=3,
                    markerfacecolor="none",
                )
                ax2.plot(
                    x,
                    q_pred_PINN[:, 1:2],
                    "b--x",
                    label="PINN",
                    markevery=10,
                    markersize=3,
                )
                ax2.ticklabel_format(style="sci", axis="x", scilimits=(-1, 2))
                plt.legend()
                plt.savefig(
                    slice_path / "op2_epoch_{}_t_{}.pdf".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
    elif mode == "euler_v1":
        testdata = np.load(ReNN_config["DataConfig"]["testdata_path"])
        x_test, q_test = testdata[:, 0:2], testdata[:, 2:5]
        mx = 1000
        x = x_test[0:mx, 1:2]
        PINN_model = PINN_ev.EulerNet(ConfigDict(PINN_config["NetConfig"])).to(DEVICE)
        ReNN_model = Re_ev1.EulerNet(ConfigDict(ReNN_config["NetConfig"])).to(DEVICE)
        for j in range(600000, 600001):
            ReNN_model.load_state_dict(
                torch.load(ReNN_model_dir / "model_{:02d}".format(j))
            )
            PINN_model.load_state_dict(
                torch.load(PINN_model_dir / "model_{:02d}".format(j))
            )
            for i in range(11):
                t = round(x_test[i * mx, 0], 3)
                x_part = x_test[i * mx : i * mx + mx, :]
                q_part = q_test[i * mx : i * mx + mx, :]
                x_part = torch.tensor(x_part, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    q_pred_PINN = PINN_model(x_part)
                    q_pred_ReNN = ReNN_model(x_part)
                q_pred_PINN = utils.to_numpy(q_pred_PINN)
                q_pred_ReNN = utils.to_numpy(q_pred_ReNN)
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
                fig.suptitle("euler equation at t={}".format(t))
                ax1.set_ylabel(r"$\rho$", rotation=0)
                ax1.plot(x, q_part[:, 0:1], "k", label="reference")
                ax1.plot(
                    x,
                    q_pred_PINN[:, 0:1],
                    "b--x",
                    label="PINN",
                    markevery=10,
                    markersize=3,
                )
                ax1.plot(
                    x,
                    q_pred_ReNN[:, 0:1],
                    "ro",
                    label="ReNN_op1",
                    markevery=10,
                    markersize=3,
                    markerfacecolor="none",
                )
                ax2.set_ylabel(r"$u$", rotation=0)
                ax2.plot(x, q_part[:, 1:2], "k", label="reference")
                ax2.plot(
                    x,
                    q_pred_PINN[:, 1:2],
                    "b--x",
                    label="PINN",
                    markevery=10,
                    markersize=3,
                )
                ax2.plot(
                    x,
                    q_pred_ReNN[:, 1:2],
                    "ro",
                    label="ReNN_op1",
                    markevery=10,
                    markersize=3,
                    markerfacecolor="none",
                )
                ax3.set_xlabel(r"$x$")
                ax3.set_ylabel(r"$p$", rotation=0)
                ax3.plot(x, q_part[:, 2:3], "k", label="reference")
                ax3.plot(
                    x,
                    q_pred_PINN[:, 2:3],
                    "b--x",
                    label="PINN",
                    markevery=10,
                    markersize=3,
                )
                ax3.plot(
                    x,
                    q_pred_ReNN[:, 2:3],
                    "ro",
                    label="ReNN_op1",
                    markevery=10,
                    markersize=3,
                    markerfacecolor="none",
                )
                plt.legend()
                plt.savefig(
                    slice_path / "op1_epoch_{}_t_{}.pdf".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
    elif mode == "euler_v2":
        testdata = np.load(ReNN_config["DataConfig"]["testdata_path"])
        x_test, q_test = testdata[:, 0:2], testdata[:, 2:5]
        mx = 1000
        x = x_test[0:mx, 1:2]
        PINN_model = PINN_ev.EulerNet(ConfigDict(PINN_config["NetConfig"])).to(DEVICE)
        ReNN_model = Re_ev2.EulerNet(ConfigDict(ReNN_config["NetConfig"])).to(DEVICE)
        for j in range(600000, 600001):
            ReNN_model.load_state_dict(
                torch.load(ReNN_model_dir / "model_{:02d}".format(j))
            )
            PINN_model.load_state_dict(
                torch.load(PINN_model_dir / "model_{:02d}".format(j))
            )
            for i in range(11):
                t = round(x_test[i * mx, 0], 3)
                x_part = x_test[i * mx : i * mx + mx, :]
                q_part = q_test[i * mx : i * mx + mx, :]
                x_part = torch.tensor(x_part, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    q_pred_PINN = PINN_model(x_part)
                    q_pred_ReNN = ReNN_model(x_part)
                q_pred_PINN = utils.to_numpy(q_pred_PINN)
                q_pred_ReNN = utils.to_numpy(q_pred_ReNN)
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
                fig.suptitle("euler equation at t={}".format(t))
                ax1.set_ylabel(r"$\rho$", rotation=0)
                ax1.plot(x, q_part[:, 0:1], "k", label="reference")
                ax1.plot(
                    x,
                    q_pred_PINN[:, 0:1],
                    "b--x",
                    label="PINN",
                    markevery=10,
                    markersize=3,
                )
                ax1.plot(
                    x,
                    q_pred_ReNN[:, 0:1],
                    "ro",
                    label="ReNN_op2",
                    markevery=10,
                    markersize=3,
                    markerfacecolor="none",
                )
                ax2.set_ylabel(r"$u$", rotation=0)
                ax2.plot(x, q_part[:, 1:2], "k", label="reference")
                ax2.plot(
                    x,
                    q_pred_PINN[:, 1:2],
                    "b--x",
                    label="PINN",
                    markevery=10,
                    markersize=3,
                )
                ax2.plot(
                    x,
                    q_pred_ReNN[:, 1:2],
                    "ro",
                    label="ReNN_op2",
                    markevery=10,
                    markersize=3,
                    markerfacecolor="none",
                )
                ax3.set_xlabel(r"$x$")
                ax3.set_ylabel(r"$p$", rotation=0)
                ax3.plot(x, q_part[:, 2:3], "k", label="reference")
                ax3.plot(
                    x,
                    q_pred_PINN[:, 2:3],
                    "b--x",
                    label="PINN",
                    markevery=10,
                    markersize=3,
                )
                ax3.plot(
                    x,
                    q_pred_ReNN[:, 2:3],
                    "ro",
                    label="ReNN_op2",
                    markevery=10,
                    markersize=3,
                    markerfacecolor="none",
                )
                plt.legend()
                plt.savefig(
                    slice_path / "op2_epoch_{}_t_{}.pdf".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
    elif mode == "euler_v3":
        testdata = np.load(ReNN_config["DataConfig"]["testdata_path"])
        x_test, q_test = testdata[:, 0:2], testdata[:, 2:5]
        mx = 1000
        x = x_test[0:mx, 1:2]
        PINN_model = PINN_ev.EulerNet(ConfigDict(PINN_config["NetConfig"])).to(DEVICE)
        ReNN_model = Re_ev3.EulerNet(ConfigDict(ReNN_config["NetConfig"])).to(DEVICE)
        for j in range(600000, 600001):
            ReNN_model.load_state_dict(
                torch.load(ReNN_model_dir / "model_{:02d}".format(j))
            )
            PINN_model.load_state_dict(
                torch.load(PINN_model_dir / "model_{:02d}".format(j))
            )
            for i in range(11):
                t = round(x_test[i * mx, 0], 3)
                x_part = x_test[i * mx : i * mx + mx, :]
                q_part = q_test[i * mx : i * mx + mx, :]
                x_part = torch.tensor(x_part, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    q_pred_PINN = PINN_model(x_part)
                    q_pred_ReNN = ReNN_model(x_part)
                q_pred_PINN = utils.to_numpy(q_pred_PINN)
                q_pred_ReNN = utils.to_numpy(q_pred_ReNN)
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
                fig.suptitle("euler equation at t={}".format(t))
                ax1.set_ylabel(r"$\rho$", rotation=0)
                ax1.plot(x, q_part[:, 0:1], "k", label="reference")
                ax1.plot(
                    x,
                    q_pred_PINN[:, 0:1],
                    "b--x",
                    label="PINN",
                    markevery=10,
                    markersize=3,
                )
                ax1.plot(
                    x,
                    q_pred_ReNN[:, 0:1],
                    "ro",
                    label="ReNN_op3",
                    markevery=10,
                    markersize=3,
                    markerfacecolor="none",
                )
                ax2.set_ylabel(r"$u$", rotation=0)
                ax2.plot(x, q_part[:, 1:2], "k", label="reference")
                ax2.plot(
                    x,
                    q_pred_PINN[:, 1:2],
                    "b--x",
                    label="PINN",
                    markevery=10,
                    markersize=3,
                )
                ax2.plot(
                    x,
                    q_pred_ReNN[:, 1:2],
                    "ro",
                    label="ReNN_op3",
                    markevery=10,
                    markersize=3,
                    markerfacecolor="none",
                )
                ax3.set_xlabel(r"$x$")
                ax3.set_ylabel(r"$p$", rotation=0)
                ax3.plot(x, q_part[:, 2:3], "k", label="reference")
                ax3.plot(
                    x,
                    q_pred_PINN[:, 2:3],
                    "b--x",
                    label="PINN",
                    markevery=10,
                    markersize=3,
                )
                ax3.plot(
                    x,
                    q_pred_ReNN[:, 2:3],
                    "ro",
                    label="ReNN_op3",
                    markevery=10,
                    markersize=3,
                    markerfacecolor="none",
                )
                plt.legend()
                plt.savefig(
                    slice_path / "op3_epoch_{}_t_{}.pdf".format(j, t),
                    dpi=500,
                    bbox_inches="tight",
                )
                plt.close()
    else:
        raise ValueError("other mode have not been implemented")


if __name__ == "__main__":
    evaluate(
        mode="swe_v1",
        archive=Path("/nfs/my/Origin/_experiment_figures/swe/dam-break"),
        PINNpath=Path(
            "/nfs/my/Origin/OriginPINN/_output/swe/dam-break/2023-11-03T10-50-32"
        ),
        ReNNpath=Path(
            "/nfs/my/Origin/OriginRela/_output/swe_v1/dam-break/2023-11-03T07-15-08star"
        ),
    )
