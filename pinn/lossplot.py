from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root_dir = Path("/nfs/my/Origin/OriginPINN/_output/burgers/sine/2023-10-30T13-18-56")
csv_path = root_dir / "history.csv"
loss_path = root_dir / "loss.png"
mae_path = root_dir / "mae.png"
data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
epochs = data[:, 0:1]
res_loss = data[:, 2:3]
u_ic = data[:, 3:4]
u_bc = data[:, 4:5]
mae = data[:, 5:6]

plt.figure()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.semilogy(epochs, res_loss, label=r"$\mathcal{L}_{\text{PDE}}$")
plt.semilogy(epochs, u_ic, label=r"$\mathcal{L}_{\text{IC}}$")
plt.semilogy(epochs, u_bc, label=r"$\mathcal{L}_{\text{BC}}$")
plt.legend()
plt.savefig(loss_path, dpi=500, bbox_inches="tight")
plt.close()

plt.figure()
plt.xlabel("epochs")
plt.ylabel("mae")
plt.semilogy(epochs, mae, label="mae")
plt.legend()
plt.savefig(mae_path, dpi=500, bbox_inches="tight")
plt.close()
