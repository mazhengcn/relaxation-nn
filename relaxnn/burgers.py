import math

import torch
from ml_collections import ConfigDict

import OriginRela.basic


def gradients(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )


class BurgersNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.u = OriginRela.basic.Net(
            config.layer_sizes[0], config.activation[0], config.configuration[0]
        )
        self.flux = OriginRela.basic.Net(
            config.layer_sizes[1], config.activation[1], config.configuration[1]
        )
        if config.loss == "MSE":
            self.loss_fn = torch.nn.MSELoss()
        self.ibc_type = config.ibc_type

    def forward(self, x):
        return self.u(x)

    def interior_loss(self, x, weights=[1.0, 1.0]):
        u = self.u(x)
        flux = self.flux(x)
        dflux_g = gradients(flux, x)[0]
        flux_x = dflux_g[:, 1:2]
        du_g = gradients(u, x)[0]
        u_t = du_g[:, 0:1]
        f1 = self.loss_fn(u_t + flux_x, torch.zeros_like(u_t))
        f2 = self.loss_fn(flux - 0.5 * u**2, torch.zeros_like(flux))
        return f1, f2

    def init_loss(self, x_ic):
        u_ic_nn = self.u(x_ic)
        F_ic_nn = self.flux(x_ic)
        f1 = self.loss_fn(u_ic_nn, self.u_ic(x_ic))
        f2 = self.loss_fn(F_ic_nn, self.F_ic(x_ic))
        return f1, f2

    def bc_loss(self, x_bc):
        u_bc_nn = self.u(x_bc)
        F_bc_nn = self.flux(x_bc)
        f1 = self.loss_fn(u_bc_nn, self.u_bc(x_bc))
        f2 = self.loss_fn(F_bc_nn, self.F_bc(x_bc))
        return f1, f2

    def u_ic(self, x):
        if self.ibc_type[0] == "riemann":
            xc = torch.tensor(0.0)
            ul = torch.tensor(1.0)
            ur = torch.tensor(0.0)
            return ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
        elif self.ibc_type[0] == "sine":
            xc = torch.tensor(0.0)
            return -torch.sin(math.pi * (x[:, 1:2] - xc))

    def u_bc(self, x):
        if self.ibc_type[1] == "riemann":
            xc = torch.tensor(0.0)
            ul = torch.tensor(1.0)
            ur = torch.tensor(0.0)
            return ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
        elif self.ibc_type[1] == "sine":
            return torch.zeros_like(x[:, 1:2])

    def F_ic(self, x):
        return 0.5 * self.u_ic(x) ** 2

    def F_bc(self, x):
        return 0.5 * self.u_bc(x) ** 2
