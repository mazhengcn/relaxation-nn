import torch
from ml_collections import ConfigDict
from pinns_utils_torch.model import basic
from torch.func import jacrev, vmap


class SweNet(torch.nn.Module):
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
        self.gravity = float(getattr(config, "gravity", 1.0))

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
        height = state[..., 0:1]
        velocity = state[..., 1:2]
        momentum = height * velocity
        return torch.cat([height, momentum], dim=-1)

    def _flux_from_state(self, state):
        height = state[..., 0:1]
        velocity = state[..., 1:2]
        momentum = height * velocity
        flux_momentum = height * velocity**2 + 0.5 * self.gravity * height**2
        return torch.cat([momentum, flux_momentum], dim=-1)

    def _target_state(self, x, ibc_type):
        x_coord = x[..., 1:2]
        normalized = ibc_type.replace("_", "-").lower()
        if normalized == "dam-break":
            height = torch.where(x_coord <= 0.0, 1.0, 0.5)
            velocity = torch.zeros_like(height)
            return torch.cat([height, velocity], dim=-1)
        if normalized == "2shock":
            height = torch.ones_like(x_coord)
            velocity = torch.where(x_coord <= 0.0, 1.0, -1.0)
            return torch.cat([height, velocity], dim=-1)
        raise ValueError("Unsupported ibc type {}".format(ibc_type))
