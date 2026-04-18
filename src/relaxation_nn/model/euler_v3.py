import torch
from ml_collections import ConfigDict
from pinns_utils_torch.model import basic
from torch.func import jacrev, vmap


class EulerNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self._rho = basic.Net(
            config.layer_sizes[0],
            config.activation[0],
            config.configuration[0],
            config.initialization[0],
        )
        self._u = basic.Net(
            config.layer_sizes[0],
            config.activation[0],
            config.configuration[0],
            config.initialization[0],
        )
        self._p = basic.Net(
            config.layer_sizes[0],
            config.activation[0],
            config.configuration[0],
            config.initialization[0],
        )
        self._flux_uEp = basic.Net(
            config.layer_sizes[1],
            config.activation[1],
            config.configuration[1],
            config.initialization[1],
        )
        if config.loss == "MSE":
            self.loss_fn = basic.PDElossfn()
        self.ibc_type = config.ibc_type

        self.register_buffer("x_L", torch.tensor(config.x_L, dtype=torch.float32))
        self.register_buffer("x_R", torch.tensor(config.x_R, dtype=torch.float32))
        rho_L, u_L, p_L, rho_R, u_R, p_R = self._get_boundary_values(config.ibc_type)
        self.register_buffer("rho_L", torch.tensor(rho_L, dtype=torch.float32))
        self.register_buffer("rho_R", torch.tensor(rho_R, dtype=torch.float32))
        self.register_buffer("u_L", torch.tensor(u_L, dtype=torch.float32))
        self.register_buffer("u_R", torch.tensor(u_R, dtype=torch.float32))
        self.register_buffer("p_L", torch.tensor(p_L, dtype=torch.float32))
        self.register_buffer("p_R", torch.tensor(p_R, dtype=torch.float32))

    @staticmethod
    def _get_boundary_values(ibc_type):
        bc = ibc_type[1]
        if bc == "shock_tube":
            return 1.0, 0.0, 1.0, 0.125, 0.0, 0.1
        elif bc == "lax_tube":
            return 0.445, 0.698, 3.528, 0.5, 0.0, 0.571
        elif bc == "blast":
            return 1.0, 0.0, 1.0, 1.0, 0.0, 1.0
        else:
            raise ValueError(f"Unknown ibc_type: {bc}")

    def _physics(self, inputs):
        """Apply BC-encoding transformation to raw network outputs."""
        x = inputs[..., 1:2]
        alpha = (self.x_R - x) / (self.x_R - self.x_L)  # ty: ignore
        beta = (x - self.x_L) / (self.x_R - self.x_L)  # ty: ignore
        phi = (x - self.x_L) * (self.x_R - x)

        rho = self.rho_L**alpha * self.rho_R**beta * torch.exp(phi * self._rho(inputs))
        u = torch.sqrt(phi) * self._u(inputs) + alpha * self.u_L + beta * self.u_R
        p = self.p_L**alpha * self.p_R**beta * torch.exp(phi * self._p(inputs))
        return rho, u, p

    def forward(self, x):
        rho, u, p = self._physics(x)
        return torch.cat((rho, u, p), dim=-1)

    def q(self, t, x):
        inputs = torch.hstack((t, x))
        rho, u, p = self._physics(inputs)
        momentum = rho * u
        energy = 2.5 * p + 0.5 * rho * u**2
        return torch.cat((rho, momentum, energy), dim=-1)

    def f(self, t, x):
        inputs = torch.hstack((t, x))
        rho, u, p = self._physics(inputs)
        momentum = rho * u
        rhou2p = rho * u**2 + p
        flux_uEp = self._flux_uEp(inputs)
        return torch.cat((momentum, rhou2p, flux_uEp), dim=-1)

    def flux(self, x):
        return self._flux_uEp(x)

    def flux_true(self, x):
        rho, u, p = self._physics(x)
        energy = 2.5 * p + 0.5 * rho * u**2
        uEp = u * (energy + p)
        return uEp

    def interior_loss(self, x, weights=[1.0, 1.0, 1.0, 1.0]):
        x = x.to(torch.float32)
        tt, xx = x.hsplit(2)
        q_t = vmap(jacrev(self.q, argnums=0), in_dims=(0, 0))(tt, xx)
        f_x = vmap(jacrev(self.f, argnums=1), in_dims=(0, 0))(tt, xx)
        L_eq1 = self.loss_fn(q_t[:, 0:1, :], -f_x[:, 0:1, :])
        L_eq2 = self.loss_fn(q_t[:, 1:2, :], -f_x[:, 1:2, :])
        L_eq3 = self.loss_fn(q_t[:, 2:3, :], -f_x[:, 2:3, :])
        flux = self.flux(x)
        flux_true = self.flux_true(x)
        L_flux = self.loss_fn(flux, flux_true)
        return (
            weights[0] * L_eq1 + weights[1] * L_eq2 + weights[2] * L_eq3,
            weights[3] * L_flux,
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
            raise ValueError("other ibc type have not been implemented")

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
            raise ValueError("other ibc type have not been implemented")

    def F_ic(self, x):
        q = self.q_ic(x)
        rho = q[:, 0:1]
        velocity = q[:, 1:2]
        pressure = q[:, 2:3]
        flux3 = velocity * (2.5 * pressure + 0.5 * rho * velocity**2 + pressure)
        return flux3

    def F_bc(self, x):
        q = self.q_bc(x)
        rho = q[:, 0:1]
        velocity = q[:, 1:2]
        pressure = q[:, 2:3]
        flux3 = velocity * (2.5 * pressure + 0.5 * rho * velocity**2 + pressure)
        return flux3
