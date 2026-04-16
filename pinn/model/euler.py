import torch
from ml_collections import ConfigDict

from shared.model import basic
from torch.func import jacrev, vmap


class EulerNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self._state = basic.Net(
            config.layer_sizes,
            config.activation,
            config.configuration,
            initialization=config.initialization,
        )
        if config.loss == "MSE":
            self.loss_fn = basic.PDElossfn()
        self.ibc_type = config.ibc_type
        self.gamma = float(getattr(config, "gamma", 1.4))
        self.shu_osher_amplitude = float(getattr(config, "shu_osher_amplitude", 0.2))

    def forward(self, x):
        return self._state(x)

    def flux(self, x):
        return self._flux_from_state(self.forward(x))

    def interior_loss(self, x):
        x = x.to(torch.float32)
        tt, xx = x.hsplit(2)

        def q_fn(t, x_coord):
            inputs = torch.cat([t, x_coord], dim=-1)
            return self._conserved_state(self.forward(inputs))

        def f_fn(t, x_coord):
            return self.flux(torch.cat([t, x_coord], dim=-1))

        q_t = vmap(jacrev(q_fn, argnums=0), in_dims=(0, 0))(tt, xx).squeeze(-1)
        f_x = vmap(jacrev(f_fn, argnums=1), in_dims=(0, 0))(tt, xx).squeeze(-1)
        residual = q_t + f_x
        loss_eq = self.loss_fn(residual, torch.zeros_like(residual))
        loss_flux = torch.zeros((), device=x.device, dtype=x.dtype)
        return loss_eq, loss_flux

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
        return self._target_state(x, self.ibc_type[0])

    def q_bc(self, x):
        return self._target_state(x, self.ibc_type[1])

    def compute_loss_terms(self, x_int, x_ic, x_bc, int_weights=None):
        del int_weights
        res_loss, _ = self.interior_loss(x_int)
        u_ic_loss, _ = self.init_loss(x_ic)
        u_bc_loss, _ = self.bc_loss(x_bc)
        return {
            "res_loss": res_loss,
            "u_ic": u_ic_loss,
            "u_bc": u_bc_loss,
        }

    def _conserved_state(self, state):
        rho = state[..., 0:1]
        velocity = state[..., 1:2]
        pressure = state[..., 2:3]
        momentum = rho * velocity
        energy = pressure / (self.gamma - 1.0) + 0.5 * rho * velocity**2
        return torch.cat([rho, momentum, energy], dim=-1)

    def _flux_from_state(self, state):
        rho = state[..., 0:1]
        velocity = state[..., 1:2]
        pressure = state[..., 2:3]
        momentum = rho * velocity
        energy = pressure / (self.gamma - 1.0) + 0.5 * rho * velocity**2
        flux_rho = momentum
        flux_momentum = rho * velocity**2 + pressure
        flux_energy = velocity * (energy + pressure)
        return torch.cat([flux_rho, flux_momentum, flux_energy], dim=-1)

    def _target_state(self, x, ibc_type):
        x_coord = x[..., 1:2]
        normalized = ibc_type.replace("-", "_").lower()

        if normalized in {"shock_tube", "shocktube"}:
            rho = torch.where(x_coord <= 0.0, 1.0, 0.125)
            velocity = torch.zeros_like(rho)
            pressure = torch.where(x_coord <= 0.0, 1.0, 0.1)
            return torch.cat([rho, velocity, pressure], dim=-1)

        if normalized in {"lax_tube", "laxtube"}:
            rho = torch.where(x_coord <= 0.0, 0.445, 0.5)
            velocity = torch.where(x_coord <= 0.0, 0.698, 0.0)
            pressure = torch.where(x_coord <= 0.0, 3.528, 0.571)
            return torch.cat([rho, velocity, pressure], dim=-1)

        if normalized in {"shu_osher", "shocksine", "shock_sine"}:
            left_mask = x_coord < -4.0
            rho = torch.where(
                left_mask,
                3.857143,
                1.0 + self.shu_osher_amplitude * torch.sin(5.0 * x_coord),
            )
            velocity = torch.where(left_mask, 2.629369, 0.0)
            pressure = torch.where(left_mask, 10.33333, 1.0)
            return torch.cat([rho, velocity, pressure], dim=-1)

        if normalized in {"blast", "simplified_blast"}:
            left_mask = x_coord <= -0.1
            middle_mask = (x_coord > -0.1) & (x_coord <= 0.1)
            pressure = torch.where(
                left_mask,
                1.0,
                torch.where(middle_mask, 0.01, 1.0),
            )
            rho = torch.ones_like(pressure)
            velocity = torch.zeros_like(pressure)
            return torch.cat([rho, velocity, pressure], dim=-1)

        raise ValueError("Unsupported ibc type {}".format(ibc_type))
