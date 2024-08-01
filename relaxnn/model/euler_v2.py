import math

import torch
from ml_collections import ConfigDict
from model import basic
from torch.func import jacrev, vmap


class EulerNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self._rho = basic.Net(
            config.layer_sizes[0], config.activation[0], config.configuration[0]
        )
        self._u = basic.Net(
            config.layer_sizes[0], config.activation[0], config.configuration[0]
        )
        self._p = basic.Net(
            config.layer_sizes[0], config.activation[0], config.configuration[0]
        )
        self._flux_rhou2p = basic.Net(
            config.layer_sizes[1], config.activation[1], config.configuration[1]
        )
        self._flux_uEp = basic.Net(
            config.layer_sizes[1], config.activation[1], config.configuration[1]
        )
        if config.loss == "MSE":
            self.loss_fn = basic.PDElossfn()
        self.ibc_type = config.ibc_type

    def forward(self, x):
        return torch.cat((self._rho(x), self._u(x), self._p(x)), dim=-1)

    def q(self, t, x):
        inputs = torch.hstack((t, x))
        rho = self._rho(inputs)
        u = self._u(inputs)
        p = self._p(inputs)
        momentum = rho * u
        energy = 0.5 * p + 0.5 * rho * u**2
        return torch.cat((rho, momentum, energy), dim=-1)

    def f(self, t, x):
        inputs = torch.hstack((t, x))
        rho = self._rho(inputs)
        u = self._u(inputs)
        p = self._p(inputs)
        momentum = rho * u
        flux_rhou2p = self._flux_rhou2p(inputs)
        flux_uEp = self._flux_uEp(inputs)
        return torch.cat((momentum, flux_rhou2p, flux_uEp), dim=-1)

    def flux(self, x):
        return torch.cat((self._flux_rhou2p(x), self._flux_uEp(x)), dim=-1)

    def flux_true(self, x):
        rho = self._rho(x)
        u = self._u(x)
        p = self._p(x)
        energy = 0.5 * p + 0.5 * rho * u**2
        rhou2p = rho * u**2 + p
        uEp = u * (energy + p)
        return torch.cat((rhou2p, uEp), dim=-1)

    def interior_loss(self, x, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        x = x.to(torch.float32)
        tt, xx = x.hsplit(2)
        q_t = vmap(jacrev(self.q, argnums=0), in_dims=(0, 0))(tt, xx)
        f_x = vmap(jacrev(self.f, argnums=1), in_dims=(0, 0))(tt, xx)
        L_eq1 = self.loss_fn(q_t[:, :, 0:1], -f_x[:, :, 0:1])
        L_eq2 = self.loss_fn(q_t[:, :, 1:2], -f_x[:, :, 1:2])
        L_eq3 = self.loss_fn(q_t[:, :, 2:3], -f_x[:, :, 2:3])
        flux = self.flux(x)
        flux_true = self.flux_true(x)
        L_flux1 = self.loss_fn(flux[:, 0:1], flux_true[:, 0:1])
        L_flux2 = self.loss_fn(flux[:, 1:2], flux_true[:, 1:2])
        return (
            weights[0] * L_eq1 + weights[1] * L_eq2 + weights[2] * L_eq3,
            weights[3] * L_flux1 + weights[4] * L_flux2,
        )

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
        if self.ibc_type[0] == "shock_tube":
            xc = torch.tensor(0.0)
            rho_l = torch.tensor(1.0)
            rho_r = torch.tensor(0.125)
            pressure_l = torch.tensor(1.0)
            pressure_r = torch.tensor(0.1)
            rho = rho_l * (x[:, 1:2] <= xc) + rho_r * (x[:, 1:2] > xc)
            velocity = torch.zeros_like(rho)
            pressure = pressure_l * (x[:, 1:2] <= xc) + pressure_r * (x[:, 1:2] > xc)
            return torch.cat((rho, velocity, pressure), dim=-1)
        elif self.ibc_type[0] == "lax_tube":
            xc = torch.tensor(0.0)
            rho_l = torch.tensor(0.445)
            rho_r = torch.tensor(0.5)
            u_l = torch.tensor(0.698)
            u_r = torch.tensor(0.0)
            pressure_l = torch.tensor(3.528)
            pressure_r = torch.tensor(0.571)
            rho = rho_l * (x[:, 1:2] <= xc) + rho_r * (x[:, 1:2] > xc)
            velocity = u_l * (x[:, 1:2] <= xc) + u_r * (x[:, 1:2] > xc)
            pressure = pressure_l * (x[:, 1:2] <= xc) + pressure_r * (x[:, 1:2] > xc)
            return torch.cat((rho, velocity, pressure), dim=-1)
        elif self.ibc_type[0] == "shu_osher":
            xc = torch.tensor(-4.0)
            rho_l = torch.tensor(3.857143)
            velocity_l = torch.tensor(2.629369)
            pressure_l = torch.tensor(10.33333)
            pressure_r = torch.tensor(1.0)
            rho = rho_l * (x[:, 1:2] < xc) + (
                1.0 + self.epsilon * torch.sin(5 * xc)
            ) * (x[:, 1:2] >= xc)
            velocity = velocity_l * (x[:, 1:2] < xc)
            pressure = pressure_l * (x[:, 1:2] < xc) + pressure_r * (x[:, 1:2] >= xc)
            return torch.cat((rho, velocity, pressure), dim=-1)
        elif self.ibc_type[0] == "blast":
            xl = -0.1
            xr = 0.1
            pressure_l = torch.tensor(1.0)
            pressure_m = torch.tensor(0.01)
            pressure_r = torch.tensor(1.0)
            pressure = (
                pressure_l * (x[:, 1:2] <= xl)
                + pressure_m * (x[:, 1:2] <= xr) * (x[:, 1:2] > xl)
                + pressure_r * (x[:, 1:2] > xr)
            )
            rho = torch.ones_like(pressure)
            velocity = torch.zeros_like(pressure)
            return torch.cat((rho, velocity, pressure), dim=-1)
        else:
            raise ValueError("other ibc type have not been implemented")

    def q_bc(self, x):
        if self.ibc_type[1] == "shock_tube":
            xc = torch.tensor(0.0)
            rho_l = torch.tensor(1.0)
            rho_r = torch.tensor(0.125)
            pressure_l = torch.tensor(1.0)
            pressure_r = torch.tensor(0.1)
            rho = rho_l * (x[:, 1:2] <= xc) + rho_r * (x[:, 1:2] > xc)
            velocity = torch.zeros_like(rho)
            pressure = pressure_l * (x[:, 1:2] <= xc) + pressure_r * (x[:, 1:2] > xc)
            return torch.cat((rho, velocity, pressure), dim=-1)
        elif self.ibc_type[1] == "lax_tube":
            xc = torch.tensor(0.0)
            rho_l = torch.tensor(0.445)
            rho_r = torch.tensor(0.5)
            u_l = torch.tensor(0.698)
            u_r = torch.tensor(0.0)
            pressure_l = torch.tensor(3.528)
            pressure_r = torch.tensor(0.571)
            rho = rho_l * (x[:, 1:2] <= xc) + rho_r * (x[:, 1:2] > xc)
            velocity = u_l * (x[:, 1:2] <= xc) + u_r * (x[:, 1:2] > xc)
            pressure = pressure_l * (x[:, 1:2] <= xc) + pressure_r * (x[:, 1:2] > xc)
            return torch.cat((rho, velocity, pressure), dim=-1)
        elif self.ibc_type[1] == "shu_osher":
            xc = torch.tensor(-4.0)
            rho_l = torch.tensor(3.857143)
            velocity_l = torch.tensor(2.629369)
            pressure_l = torch.tensor(10.33333)
            pressure_r = torch.tensor(1.0)
            rho = rho_l * (x[:, 1:2] < xc) + (
                1.0 + self.epsilon * torch.sin(5 * xc)
            ) * (x[:, 1:2] >= xc)
            velocity = velocity_l * (x[:, 1:2] < xc)
            pressure = pressure_l * (x[:, 1:2] < xc) + pressure_r * (x[:, 1:2] >= xc)
            return torch.cat((rho, velocity, pressure), dim=-1)
        elif self.ibc_type[1] == "blast":
            xl = -0.1
            xr = 0.1
            pressure_l = torch.tensor(1.0)
            pressure_m = torch.tensor(0.01)
            pressure_r = torch.tensor(1.0)
            pressure = (
                pressure_l * (x[:, 1:2] <= xl)
                + pressure_m * (x[:, 1:2] <= xr) * (x[:, 1:2] > xl)
                + pressure_r * (x[:, 1:2] > xr)
            )
            rho = torch.ones_like(pressure)
            velocity = torch.zeros_like(pressure)
            return torch.cat((rho, velocity, pressure), dim=-1)
        else:
            raise ValueError("other ibc type have not been implemented")

    def F_ic(self, x):
        q = self.q_ic(x)
        rho = q[:, 0:1]
        velocity = q[:, 1:2]
        pressure = q[:, 2:3]
        flux2 = rho * velocity**2 + pressure
        flux3 = velocity * (2.5 * pressure + 0.5 * rho * velocity**2 + pressure)
        return torch.cat((flux2, flux3), dim=-1)

    def F_bc(self, x):
        q = self.q_bc(x)
        rho = q[:, 0:1]
        velocity = q[:, 1:2]
        pressure = q[:, 2:3]
        flux2 = rho * velocity**2 + pressure
        flux3 = velocity * (2.5 * pressure + 0.5 * rho * velocity**2 + pressure)
        return torch.cat((flux2, flux3), dim=-1)
