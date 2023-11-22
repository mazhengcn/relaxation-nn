from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compare(path1: Path, path2: Path):
    pinn_data = np.loadtxt(path1 / "history.csv", delimiter=",", skiprows=1)
    rela_data = np.loadtxt(path2 / "history.csv", delimiter=",", skiprows=1)
    plt.semilogy(pinn_data[:, 0], pinn_data[:, 1], label=r"$\mathcal{L}_{\text{PINN}}$")
    plt.semilogy(rela_data[:, 0], rela_data[:, 1], label=r"$\mathcal{L}_{\text{ReNN}}$")
    plt.ticklabel_format(style="sci", axis="x", scilimits=(-1, 2))
    plt.xlabel("epoch")
    plt.ylabel("total_loss")
    plt.legend()
    plt.savefig(archive / "burgers_total_loss.pdf", dpi=500, bbox_inches="tight")


def swecompare(path1: Path, path2: Path, path3: Path):
    pinn_data = np.loadtxt(path1 / "history.csv", delimiter=",", skiprows=1)
    rela_op1_data = np.loadtxt(path2 / "history.csv", delimiter=",", skiprows=1)
    rela_op2_data = np.loadtxt(path3 / "history.csv", delimiter=",", skiprows=1)
    plt.semilogy(pinn_data[:, 0], pinn_data[:, 1], label=r"$\mathcal{L}_{\text{PINN}}$")
    plt.semilogy(
        rela_op1_data[:, 0],
        rela_op1_data[:, 1],
        label=r"$\mathcal{L}_{\text{ReNN}}^{\text{op1}}$",
    )
    plt.semilogy(
        rela_op2_data[:, 0],
        rela_op2_data[:, 1],
        label=r"$\mathcal{L}_{\text{ReNN}}^{\text{op2}}$",
    )
    plt.ticklabel_format(style="sci", axis="x", scilimits=(-1, 2))
    plt.xlabel("epoch")
    plt.ylabel("total_loss")
    plt.legend()
    plt.savefig(archive / "swe_total_loss.pdf", dpi=500, bbox_inches="tight")


def eulercompare(path1: Path, path2: Path, path3: Path, path4: Path):
    pinn_data = np.loadtxt(path1 / "history.csv", delimiter=",", skiprows=1)
    rela_op1_data = np.loadtxt(path2 / "history.csv", delimiter=",", skiprows=1)
    rela_op2_data = np.loadtxt(path3 / "history.csv", delimiter=",", skiprows=1)
    rela_op3_data = np.loadtxt(path4 / "history.csv", delimiter=",", skiprows=1)
    plt.semilogy(pinn_data[:, 0], pinn_data[:, 1], label=r"$\mathcal{L}_{\text{PINN}}$")
    plt.semilogy(
        rela_op1_data[:, 0],
        rela_op1_data[:, 1],
        label=r"$\mathcal{L}_{\text{ReNN}}^{\text{op1}}$",
    )
    plt.semilogy(
        rela_op2_data[:, 0],
        rela_op2_data[:, 1],
        label=r"$\mathcal{L}_{\text{ReNN}}^{\text{op2}}$",
    )
    plt.semilogy(
        rela_op3_data[:, 0],
        rela_op3_data[:, 1],
        label=r"$\mathcal{L}_{\text{ReNN}}^{\text{op3}}$",
    )
    plt.ticklabel_format(style="sci", axis="x", scilimits=(-1, 2))
    plt.xlabel("epoch")
    plt.ylabel("total_loss")
    plt.legend()
    plt.savefig(archive / "euler_total_loss.pdf", dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    # archive = Path("/nfs/my/Origin/_experiment_figures/burgers/riemann")
    # # for burgers equation
    # pinn_path = Path(
    #     "/nfs/my/Origin/OriginPINN/_output/burgers/riemann/2023-10-30T12-26-30"
    # )
    # rela_path = Path(
    #     "/nfs/my/Origin/OriginRela/_output/burgers/riemann/2023-10-30T10-10-43star"
    # )
    # compare(pinn_path, rela_path)
    # # for swe equation
    # archive = Path("/nfs/my/Origin/_experiment_figures/swe/dam-break")
    # pinn_path = Path(
    #     "/nfs/my/Origin/OriginPINN/_output/swe/dam-break/2023-11-03T10-50-32"
    # )
    # rela_op1_path = Path(
    #     "/nfs/my/Origin/OriginRela/_output/swe_v1/dam-break/2023-11-03T07-15-08star"
    # )
    # rela_op2_path = Path(
    #     "/nfs/my/Origin/OriginRela/_output/swe_v2/dam-break/2023-11-03T17-41-51star"
    # )
    # swecompare(pinn_path, rela_op1_path, rela_op2_path)
    # # for euler equation
    archive = Path("/nfs/my/Origin/_experiment_figures/euler/sod")
    pinn_path = Path("/nfs/my/Origin/OriginPINN/_output/euler/sod/2023-11-08T12-12-12")
    rela_op1_path = Path(
        "/nfs/my/Origin/OriginRela/_output/euler_v1/sod/2023-11-08T12-06-39"
    )
    rela_op2_path = Path(
        "/nfs/my/Origin/R3sample/_output/euler_v2/sod/2023-11-08T04-11-06"
    )
    rela_op3_path = Path(
        "/nfs/my/Origin/OriginRela/_output/euler_v3/sod/2023-11-08T03-49-55"
    )
    eulercompare(pinn_path, rela_op1_path, rela_op2_path, rela_op3_path)
