import math

import torch
from ml_collections import ConfigDict
from model import basic
from torch.func import jacrev, vmap


class SweNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self._h = basic.Net(
            config.layer_sizes[0], config.activation[0], config.configuration[0]
        )
        self._u = basic.Net(
            config.layer_sizes[0], config.activation[0], config.configuration[0]
        )
        self._flux_hu2p = basic.Net(
            config.layer_sizes[1], config.activation[1], config.configuration[1]
        )
        if config.loss == "MSE":
            self.loss_fn = basic.PDElossfn()
        self.ibc_type = config.ibc_type

    def forward(self, x):
        return torch.cat((self._h(x), self._u(x)), dim=1)

    def q(self, t, x):
        inputs = torch.hstack((t, x))
        h = self._h(inputs)
        u = self._u(inputs)
        momentum = h * u
        return torch.cat((h, momentum), dim=-1)

    def f(self, t, x):
        inputs = torch.hstack((t, x))
        h = self._h(inputs)
        u = self._u(inputs)
        momentum = h * u
        flux_hu2p = self._flux_hu2p(inputs)
        return torch.cat((momentum, flux_hu2p), dim=-1)

    def flux(self, x):
        return self._flux_hu2p(x)

    def flux_true(self, x):
        h = self._h(x)
        u = self._u(x)
        hu2p = h * u**2 + 0.5 * h**2
        return hu2p

    def interior_loss(self, x, weights=[1.0, 1.0, 1.0, 1.0]):
        x = x.to(torch.float32)
        tt, xx = x.hsplit(2)
        q_t = vmap(jacrev(self.q, argnums=0), in_dims=(0, 0))(tt, xx)
        f_x = vmap(jacrev(self.f, argnums=1), in_dims=(0, 0))(tt, xx)
        L_eq = self.loss_fn(q_t, -f_x)
        L_flux = self.loss_fn(self.flux(x), self.flux_true(x))
        return L_eq, L_flux

    def init_loss(self, x_ic, weights=[1.0, 1.0, 1.0, 1.0]):
        x_ic = x_ic.to(torch.float32)
        L_eq = self.loss_fn(self.forward(x_ic), self.q_ic(x_ic))
        L_flux = self.loss_fn(self.flux(x_ic), self.F_ic(x_ic))
        return L_eq, L_flux

    def bc_loss(self, x_bc, weights=[1.0, 1.0, 1.0, 1.0]):
        x_bc = x_bc.to(torch.float32)
        L_eq = self.loss_fn(self.forward(x_bc), self.q_bc(x_bc))
        L_flux = self.loss_fn(self.flux(x_bc), self.F_bc(x_bc))
        return L_eq, L_flux

    def q_ic(self, x):
        if self.ibc_type[0] == "dam-break":
            xc = torch.tensor(0.0)
            hl = torch.tensor(1.0)
            hr = torch.tensor(0.5)
            ul = torch.tensor(0.0)
            ur = torch.tensor(0.0)
            h = hl * (x[:, 1:2] <= xc) + hr * (x[:, 1:2] > xc)
            u = ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
            return torch.cat((h, u), dim=1)
        elif self.ibc_type[0] == "2shock":
            xc = torch.tensor(0.0)
            ul = torch.tensor(1.0)
            ur = torch.tensor(-1.0)
            u = ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
            h = torch.ones_like(u)
            return torch.cat((h, u), dim=1)
        else:
            raise ValueError("other ibc type have not been implemented")

    def q_bc(self, x):
        if self.ibc_type[1] == "dam-break":
            xc = torch.tensor(0.0)
            hl = torch.tensor(1.0)
            hr = torch.tensor(0.5)
            ul = torch.tensor(0.0)
            ur = torch.tensor(0.0)
            h = hl * (x[:, 1:2] <= xc) + hr * (x[:, 1:2] > xc)
            u = ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
            return torch.cat((h, u), dim=1)
        elif self.ibc_type[1] == "2shock":
            xc = torch.tensor(0.0)
            ul = torch.tensor(1.0)
            ur = torch.tensor(-1.0)
            u = ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
            h = torch.ones_like(u)
            return torch.cat((h, u), dim=1)
        else:
            raise ValueError("other ibc type have not been implemented")

    def F_ic(self, x):
        q = self.q_ic(x)
        height = q[:, 0:1]
        velocity = q[:, 1:2]
        F_u = height * velocity
        F_b = height * velocity**2 + 0.5 * height**2
        return torch.cat((F_u, F_b), dim=1)

    def F_bc(self, x):
        q = self.q_bc(x)
        height = q[:, 0:1]
        velocity = q[:, 1:2]
        F_u = height * velocity
        F_b = height * velocity**2 + 0.5 * height**2
        return torch.cat((F_u, F_b), dim=1)
