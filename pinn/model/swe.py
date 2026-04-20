import torch
from ml_collections import ConfigDict
from torch.func import jacrev, vmap

from shared.model import basic


class SweNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self._h = basic.Net(
            config.layer_sizes,
            config.activation,
            config.configuration,
            initialization=config.initialization,
        )
        self._u = basic.Net(
            config.layer_sizes,
            config.activation,
            config.configuration,
            initialization=config.initialization,
        )
        if config.loss == "MSE":
            self.loss_fn = basic.PDElossfn()
        self.ibc_type = config.ibc_type
        self.gravity = 1.0

    def forward(self, x):
        return torch.cat((self._h(x), self._u(x)), dim=-1)

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
        flux_momentum = h * u**2 + 0.5 * self.gravity * h**2
        return torch.cat((momentum, flux_momentum), dim=-1)

    def flux(self, x):
        h = self._h(x)
        u = self._u(x)
        momentum = h * u
        flux_momentum = h * u**2 + 0.5 * self.gravity * h**2
        return torch.cat((momentum, flux_momentum), dim=-1)

    def interior_loss(self, x, weights=[1.0, 1.0]):
        x = x.to(torch.float32)
        tt, xx = x.hsplit(2)
        q_t = vmap(jacrev(self.q, argnums=0), in_dims=(0, 0))(tt, xx)
        f_x = vmap(jacrev(self.f, argnums=1), in_dims=(0, 0))(tt, xx)
        L_eq1 = self.loss_fn(q_t[:, 0:1, :], -f_x[:, 0:1, :])
        L_eq2 = self.loss_fn(q_t[:, 1:2, :], -f_x[:, 1:2, :])
        loss_flux = torch.zeros((), device=x.device, dtype=x.dtype)
        return weights[0] * L_eq1 + weights[1] * L_eq2, loss_flux

    def init_loss(self, x_ic):
        x_ic = x_ic.to(torch.float32)
        loss_eq = self.loss_fn(self.forward(x_ic), self.q_ic(x_ic))
        loss_flux = torch.zeros((), device=x_ic.device, dtype=x_ic.dtype)
        return loss_eq, loss_flux

    def bc_loss(self, x_bc):
        x_bc = x_bc.to(torch.float32)
        loss_eq = self.loss_fn(self.forward(x_bc), self.q_bc(x_bc))
        loss_flux = torch.zeros((), device=x_bc.device, dtype=x_bc.dtype)
        return loss_eq, loss_flux

    def q_ic(self, x):
        if self.ibc_type[0] == "dam_break":
            xc = 0.0
            h_l = 1.0
            h_r = 0.5
            h = h_l * (x[:, 1:2] <= xc) + h_r * (x[:, 1:2] > xc)
            u = torch.zeros_like(h)
            return torch.cat((h, u), dim=-1)
        elif self.ibc_type[0] == "2shock":
            xc = 0.0
            h = torch.ones_like(x[:, 1:2])
            u_l = 1.0
            u_r = -1.0
            u = u_l * (x[:, 1:2] <= xc) + u_r * (x[:, 1:2] > xc)
            return torch.cat((h, u), dim=-1)
        else:
            raise ValueError("Unsupported ibc type {}".format(self.ibc_type[0]))

    def q_bc(self, x):
        if self.ibc_type[1] == "dam_break":
            xc = 0.0
            h_l = 1.0
            h_r = 0.5
            h = h_l * (x[:, 1:2] <= xc) + h_r * (x[:, 1:2] > xc)
            u = torch.zeros_like(h)
            return torch.cat((h, u), dim=-1)
        elif self.ibc_type[1] == "2shock":
            xc = 0.0
            h = torch.ones_like(x[:, 1:2])
            u_l = 1.0
            u_r = -1.0
            u = u_l * (x[:, 1:2] <= xc) + u_r * (x[:, 1:2] > xc)
            return torch.cat((h, u), dim=-1)
        else:
            raise ValueError("Unsupported ibc type {}".format(self.ibc_type[1]))

    def F_ic(self, x):
        q = self.q_ic(x)
        h = q[:, 0:1]
        u = q[:, 1:2]
        flux1 = h * u
        flux2 = h * u**2 + 0.5 * self.gravity * h**2
        return torch.cat((flux1, flux2), dim=-1)

    def F_bc(self, x):
        q = self.q_bc(x)
        h = q[:, 0:1]
        u = q[:, 1:2]
        flux1 = h * u
        flux2 = h * u**2 + 0.5 * self.gravity * h**2
        return torch.cat((flux1, flux2), dim=-1)
