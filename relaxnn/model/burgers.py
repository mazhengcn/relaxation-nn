import math

import torch
from ml_collections import ConfigDict
from model import basic
from torch.func import jacrev, vmap


class BurgersNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self._u = basic.Net(
            config.layer_sizes[0], config.activation[0], config.configuration[0]
        )
        self._flux_u2 = basic.Net(
            config.layer_sizes[1], config.activation[1], config.configuration[1]
        )
        if config.loss == "MSE":
            self.loss_fn = basic.PDElossfn()
        self.ibc_type = config.ibc_type

    def forward(self, x):
        return self._u(x)

    def q(self, t, x):
        inputs = torch.cat([t, x], dim=-1)
        u = self._u(inputs)
        return u

    def f(self, t, x):
        inputs = torch.cat([t, x], dim=-1)
        flux_u2 = self._flux_u2(inputs)
        return flux_u2

    def flux(self, x):
        return self._flux_u2(x)

    def flux_true(self, x):
        u = self._u(x)
        return 0.5 * u**2

    def interior_loss(self, x, weights=[1.0, 1.0]):
        x = x.to(torch.float32)
        tt, xx = x.hsplit(2)
        q_t = vmap(jacrev(self.q, argnums=0), in_dims=(0, 0))(tt, xx)
        f_x = vmap(jacrev(self.f, argnums=1), in_dims=(0, 0))(tt, xx)
        L_eq = self.loss_fn(q_t, -f_x)
        L_flux = self.loss_fn(self.flux(x), self.flux_true(x))
        return L_eq, L_flux

    def init_loss(self, x_ic):
        x_ic = x_ic.to(torch.float32)
        L_eq = self.loss_fn(self.forward(x_ic), self.q_ic(x_ic))
        L_flux = self.loss_fn(self.flux(x_ic), self.F_ic(x_ic))
        return L_eq, L_flux

    def bc_loss(self, x_bc):
        x_bc = x_bc.to(torch.float32)
        L_eq = self.loss_fn(self.forward(x_bc), self.q_bc(x_bc))
        L_flux = self.loss_fn(self.flux(x_bc), self.F_bc(x_bc))
        return L_eq, L_flux

    def q_ic(self, x):
        if self.ibc_type[0] == "riemann":
            xc = torch.tensor(0.0)
            ul = torch.tensor(1.0)
            ur = torch.tensor(0.0)
            return ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
        elif self.ibc_type[0] == "sine":
            xc = torch.tensor(0.0)
            return -torch.sin(math.pi * (x[:, 1:2] - xc))

    def q_bc(self, x):
        if self.ibc_type[1] == "riemann":
            xc = torch.tensor(0.0)
            ul = torch.tensor(1.0)
            ur = torch.tensor(0.0)
            return ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
        elif self.ibc_type[1] == "sine":
            return torch.zeros_like(x[:, 1:2])

    def F_ic(self, x):
        return 0.5 * self.q_ic(x) ** 2

    def F_bc(self, x):
        return 0.5 * self.q_bc(x) ** 2
