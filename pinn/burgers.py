import math

import torch
from ml_collections import ConfigDict

import OriginPINN.basic


def gradients(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )


class BurgersNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.u = OriginPINN.basic.Net(
            config.layer_sizes, config.activation, config.configuration
        )
        if config.loss == "MSE":
            self.loss_fn = torch.nn.MSELoss()
        self.ibc_type = config.ibc_type

    def forward(self, x):
        return self.u(x)

    def interior_loss(self, x):
        u = self.u(x)
        F = 0.5 * u**2
        dF_g = gradients(F, x)[0]
        F_x = dF_g[:, 1:2]
        du_g = gradients(u, x)[0]
        u_t = du_g[:, 0:1]
        f1 = self.loss_fn(u_t + F_x, torch.zeros_like(u_t))
        return f1

    def supervise(self, x, y):
        u = self.u(x)
        f = self.loss_fn(u, y)
        return f

    def init_loss(self, x_ic):
        u_ic_nn = self.u(x_ic)
        f1 = self.loss_fn(u_ic_nn, self.u_ic(x_ic))
        return f1

    def bc_loss(self, x_bc):
        u_bc_nn = self.u(x_bc)
        f1 = self.loss_fn(u_bc_nn, self.u_bc(x_bc))
        return f1

    def u_ic(self, x):
        if self.ibc_type[0] == "riemann":
            xc = torch.tensor(0.0)
            ul = torch.tensor(1.0)
            ur = torch.tensor(0.0)
            return ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
        elif self.ibc_type[0] == "sine":
            xc = torch.tensor(0.0)
            return -torch.sin(math.pi * (x[:, 1:2] - xc))
        elif self.ibc_type[0] == "rare":
            xc = torch.tensor(0.0)
            ul = torch.tensor(0.0)
            ur = torch.tensor(1.0)
            return ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)

    def u_bc(self, x):
        if self.ibc_type[1] == "riemann":
            xc = torch.tensor(0.0)
            ul = torch.tensor(1.0)
            ur = torch.tensor(0.0)
            return ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
        elif self.ibc_type[1] == "sine":
            return torch.zeros_like(x[:, 1:2])
        elif self.ibc_type[1] == "rare":
            xc = torch.tensor(0.0)
            ul = torch.tensor(0.0)
            ur = torch.tensor(1.0)
            return ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
