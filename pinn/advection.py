import math

import torch
from ml_collections import ConfigDict

import basic


def gradients(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )


class AdvectionNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.u = basic.Net(
            config.layer_sizes[0], config.activation, config.configuration
        )
        if config.loss == "MSE":
            self.loss_fn = torch.nn.MSELoss()
        self.ibc_type = config.ibc_type

    def forward(self, x):
        return self.u(x)

    def interior_loss(self, x):
        x = x.to(torch.float32)
        u = self.u(x)
        du_g = gradients(u, x)[0]
        u_t = du_g[:, 0:1]
        u_x = du_g[:, 1:2]
        f1 = self.loss_fn(u_t + 0.5 * u_x, torch.zeros_like(u_t))
        return f1

    def init_loss(self, x_ic):
        x_ic = x_ic.to(torch.float32)
        u_ic_nn = self.u(x_ic)
        f1 = self.loss_fn(u_ic_nn, self.u_ic(x_ic))
        return f1

    def bc_loss(self, x_bc):
        # zero boundary condition
        # x_bc = x_bc.to(torch.float32)
        # u_bc_nn = self.u(x_bc)
        # f1 = self.loss_fn(u_bc_nn, self.u_bc(x_bc))
        # periodic boundary condition
        x_bc = x_bc.to(torch.float32)
        x_lbc, x_temp = torch.split(x_bc, x_bc.shape[0] // 2, dim=0)
        x_rbc = torch.hstack((x_lbc[:, 0:1], x_temp[:, 1:2]))
        f1 = self.loss_fn(self.u(x_lbc), self.u(x_rbc))
        return f1

    def u_ic(self, x):
        if self.ibc_type == "riemann":
            xc = torch.tensor(0.0)
            ul = torch.tensor(1.0)
            ur = torch.tensor(0.0)
            return ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
        elif self.ibc_type == "sine":
            xc = torch.tensor(0.0)
            return -torch.sin(math.pi * (x[:, 1:2] - xc))

    def u_bc(self, x):
        if self.ibc_type == "riemann":
            xc = torch.tensor(0.0)
            ul = torch.tensor(1.0)
            ur = torch.tensor(0.0)
            return ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
        elif self.ibc_type == "sine":
            return torch.zeros_like(x[:, 1:2])

    def supervise(self, x, y):
        u = self.u(x)
        f = self.loss_fn(u, y)
        return f
