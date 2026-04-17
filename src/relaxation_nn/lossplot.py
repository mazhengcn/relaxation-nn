from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root_dir = Path("/root/Origin/OriginRela/_output/euler_v3/lax/2023-11-07T04-47-52")
csv_path = root_dir / "history.csv"
loss_path = root_dir / "loss.png"
mae_path = root_dir / "metric.png"
data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
epochs = data[:, 0:1]
res_loss = data[:, 2:3]
flux_loss = data[:, 3:4]
u_ic = data[:, 4:5]
F_ic = data[:, 5:6]
u_bc = data[:, 6:7]
mae = data[:, 8:9]

plt.figure()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.semilogy(epochs, res_loss, label=r"$\mathcal{L}_{\text{RELAX}}$")
plt.semilogy(epochs, flux_loss, label=r"$\mathcal{L}_{\text{FLUX}}$")
plt.semilogy(epochs, u_ic, label=r"$\mathcal{L}_{\text{IC}}$")
plt.semilogy(epochs, u_bc, label=r"$\mathcal{L}_{\text{BC}}$")
plt.legend()
plt.savefig(loss_path, dpi=500, bbox_inches="tight")
plt.close()

plt.figure()
plt.xlabel("epochs")
plt.ylabel("MAE")
plt.semilogy(epochs, mae, label="MAE")
plt.legend()
plt.savefig(mae_path, dpi=500, bbox_inches="tight")
plt.close()
