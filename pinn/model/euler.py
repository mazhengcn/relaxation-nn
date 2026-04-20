import torch
from ml_collections import ConfigDict
from torch.func import jacrev, vmap

from shared.model import basic


class EulerNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self._rho = basic.Net(
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
        self._p = basic.Net(
            config.layer_sizes,
            config.activation,
            config.configuration,
            initialization=config.initialization,
        )
        if config.loss == "MSE":
            self.loss_fn = basic.PDElossfn()
        self.ibc_type = config.ibc_type
        self.gamma = 1.4
        self.shu_osher_amplitude = 0.2

    def forward(self, x):
        return torch.cat((self._rho(x), self._u(x), self._p(x)), dim=-1)

    def q(self, t, x):
        inputs = torch.hstack((t, x))
        rho = self._rho(inputs)
        u = self._u(inputs)
        p = self._p(inputs)
        momentum = rho * u
        energy = p / (self.gamma - 1.0) + 0.5 * rho * u**2
        return torch.cat((rho, momentum, energy), dim=-1)

    def f(self, t, x):
        inputs = torch.hstack((t, x))
        rho = self._rho(inputs)
        u = self._u(inputs)
        p = self._p(inputs)
        momentum = rho * u
        energy = p / (self.gamma - 1.0) + 0.5 * rho * u**2
        flux_rho = momentum
        flux_momentum = rho * u**2 + p
        flux_energy = u * (energy + p)
        return torch.cat((flux_rho, flux_momentum, flux_energy), dim=-1)

    def flux(self, x):
        rho = self._rho(x)
        u = self._u(x)
        p = self._p(x)
        momentum = rho * u
        energy = p / (self.gamma - 1.0) + 0.5 * rho * u**2
        flux_rho = momentum
        flux_momentum = rho * u**2 + p
        flux_energy = u * (energy + p)
        return torch.cat((flux_rho, flux_momentum, flux_energy), dim=-1)

    def interior_loss(self, x, weights=[1.0, 1.0, 1.0]):
        x = x.to(torch.float32)
        tt, xx = x.hsplit(2)
        q_t = vmap(jacrev(self.q, argnums=0), in_dims=(0, 0))(tt, xx)
        f_x = vmap(jacrev(self.f, argnums=1), in_dims=(0, 0))(tt, xx)
        L_eq1 = self.loss_fn(q_t[:, 0:1, :], -f_x[:, 0:1, :])
        L_eq2 = self.loss_fn(q_t[:, 1:2, :], -f_x[:, 1:2, :])
        L_eq3 = self.loss_fn(q_t[:, 2:3, :], -f_x[:, 2:3, :])
        loss_flux = torch.zeros((), device=x.device, dtype=x.dtype)
        return (weights[0] * L_eq1 + weights[1] * L_eq2 + weights[2] * L_eq3, loss_flux)

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
        if self.ibc_type[0] == "shock_tube":
            xc = 0.0
            rho_l = 1.0
            rho_r = 0.125
            pressure_l = 1.0
            pressure_r = 0.1
            rho = rho_l * (x[:, 1:2] <= xc) + rho_r * (x[:, 1:2] > xc)
            velocity = torch.zeros_like(rho)
            pressure = pressure_l * (x[:, 1:2] <= xc) + pressure_r * (x[:, 1:2] > xc)
            return torch.cat((rho, velocity, pressure), dim=-1)
        elif self.ibc_type[0] == "lax_tube":
            xc = 0.0
            rho_l = 0.445
            rho_r = 0.5
            u_l = 0.698
            u_r = 0.0
            pressure_l = 3.528
            pressure_r = 0.571
            rho = rho_l * (x[:, 1:2] <= xc) + rho_r * (x[:, 1:2] > xc)
            velocity = u_l * (x[:, 1:2] <= xc) + u_r * (x[:, 1:2] > xc)
            pressure = pressure_l * (x[:, 1:2] <= xc) + pressure_r * (x[:, 1:2] > xc)
            return torch.cat((rho, velocity, pressure), dim=-1)
        elif self.ibc_type[0] == "shu_osher":
            xc = -4.0
            rho_l = 3.857143
            velocity_l = 2.629369
            pressure_l = 10.33333
            pressure_r = 1.0
            rho = rho_l * (x[:, 1:2] < xc) + (
                1.0 + self.shu_osher_amplitude * torch.sin(5.0 * x[:, 1:2])
            ) * (x[:, 1:2] >= xc)
            velocity = velocity_l * (x[:, 1:2] < xc)
            pressure = pressure_l * (x[:, 1:2] < xc) + pressure_r * (x[:, 1:2] >= xc)
            return torch.cat((rho, velocity, pressure), dim=-1)
        elif self.ibc_type[0] == "blast":
            xl = -0.1
            xr = 0.1
            pressure_l = 1.0
            pressure_m = 0.01
            pressure_r = 1.0
            pressure = (
                pressure_l * (x[:, 1:2] <= xl)
                + pressure_m * (x[:, 1:2] <= xr) * (x[:, 1:2] > xl)
                + pressure_r * (x[:, 1:2] > xr)
            )
            rho = torch.ones_like(pressure)
            velocity = torch.zeros_like(pressure)
            return torch.cat((rho, velocity, pressure), dim=-1)
        else:
            raise ValueError("Unsupported ibc type {}".format(self.ibc_type[0]))

    def q_bc(self, x):
        if self.ibc_type[1] == "shock_tube":
            xc = 0.0
            rho_l = 1.0
            rho_r = 0.125
            pressure_l = 1.0
            pressure_r = 0.1
            rho = rho_l * (x[:, 1:2] <= xc) + rho_r * (x[:, 1:2] > xc)
            velocity = torch.zeros_like(rho)
            pressure = pressure_l * (x[:, 1:2] <= xc) + pressure_r * (x[:, 1:2] > xc)
            return torch.cat((rho, velocity, pressure), dim=-1)
        elif self.ibc_type[1] == "lax_tube":
            xc = 0.0
            rho_l = 0.445
            rho_r = 0.5
            u_l = 0.698
            u_r = 0.0
            pressure_l = 3.528
            pressure_r = 0.571
            rho = rho_l * (x[:, 1:2] <= xc) + rho_r * (x[:, 1:2] > xc)
            velocity = u_l * (x[:, 1:2] <= xc) + u_r * (x[:, 1:2] > xc)
            pressure = pressure_l * (x[:, 1:2] <= xc) + pressure_r * (x[:, 1:2] > xc)
            return torch.cat((rho, velocity, pressure), dim=-1)
        elif self.ibc_type[1] == "shu_osher":
            xc = -4.0
            rho_l = 3.857143
            velocity_l = 2.629369
            pressure_l = 10.33333
            pressure_r = 1.0
            rho = rho_l * (x[:, 1:2] < xc) + (
                1.0 + self.shu_osher_amplitude * torch.sin(5.0 * x[:, 1:2])
            ) * (x[:, 1:2] >= xc)
            velocity = velocity_l * (x[:, 1:2] < xc)
            pressure = pressure_l * (x[:, 1:2] < xc) + pressure_r * (x[:, 1:2] >= xc)
            return torch.cat((rho, velocity, pressure), dim=-1)
        elif self.ibc_type[1] == "blast":
            xl = -0.1
            xr = 0.1
            pressure_l = 1.0
            pressure_m = 0.01
            pressure_r = 1.0
            pressure = (
                pressure_l * (x[:, 1:2] <= xl)
                + pressure_m * (x[:, 1:2] <= xr) * (x[:, 1:2] > xl)
                + pressure_r * (x[:, 1:2] > xr)
            )
            rho = torch.ones_like(pressure)
            velocity = torch.zeros_like(pressure)
            return torch.cat((rho, velocity, pressure), dim=-1)
        else:
            raise ValueError("Unsupported ibc type {}".format(self.ibc_type[1]))

    def F_ic(self, x):
        q = self.q_ic(x)
        rho = q[:, 0:1]
        u = q[:, 1:2]
        p = q[:, 2:3]
        energy = p / (self.gamma - 1.0) + 0.5 * rho * u**2
        flux1 = rho * u
        flux2 = rho * u**2 + p
        flux3 = u * (energy + p)
        return torch.cat((flux1, flux2, flux3), dim=-1)

    def F_bc(self, x):
        q = self.q_bc(x)
        rho = q[:, 0:1]
        u = q[:, 1:2]
        p = q[:, 2:3]
        energy = p / (self.gamma - 1.0) + 0.5 * rho * u**2
        flux1 = rho * u
        flux2 = rho * u**2 + p
        flux3 = u * (energy + p)
        return torch.cat((flux1, flux2, flux3), dim=-1)
