import math

import torch
from ml_collections import ConfigDict
from shared.model import basic
from torch.func import jacrev, vmap


class BurgersNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self._u = basic.Net(
            config.layer_sizes,
            config.activation,
            config.configuration,
            initialization=config.initialization,
        )
        if config.loss == "MSE":
            self.loss_fn = basic.PDElossfn()
        self.ibc_type = config.ibc_type

    def forward(self, x):
        return self._u(x)

    def flux(self, x):
        u = self.forward(x)
        return 0.5 * u**2

    def interior_loss(self, x):
        x = x.to(torch.float32)
        tt, xx = x.hsplit(2)
        
        def q_fn(t, x_coord):
            return self.forward(torch.cat([t, x_coord], dim=-1))

        def f_fn(t, x_coord):
            return self.flux(torch.cat([t, x_coord], dim=-1))

        q_t = vmap(jacrev(q_fn, argnums=0), in_dims=(0, 0))(tt, xx).squeeze(-1)
        f_x = vmap(jacrev(f_fn, argnums=1), in_dims=(0, 0))(tt, xx).squeeze(-1)
        L_eq = self.loss_fn(q_t, -f_x)
        L_flux = torch.zeros((), device=x.device, dtype=x.dtype)
        return L_eq, L_flux

    def init_loss(self, x_ic):
        x_ic = x_ic.to(torch.float32)
        L_eq = self.loss_fn(self.forward(x_ic), self.q_ic(x_ic))
        L_flux = torch.zeros((), device=x_ic.device, dtype=x_ic.dtype)
        return L_eq, L_flux

    def bc_loss(self, x_bc):
        x_bc = x_bc.to(torch.float32)
        L_eq = self.loss_fn(self.forward(x_bc), self.q_bc(x_bc))
        L_flux = torch.zeros((), device=x_bc.device, dtype=x_bc.dtype)
        return L_eq, L_flux

    def q_ic(self, x):
        if self.ibc_type[0] == "riemann":
            xc = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            ul = torch.tensor(1.0, device=x.device, dtype=x.dtype)
            ur = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            return ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
        if self.ibc_type[0] == "sine":
            xc = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            return -torch.sin(math.pi * (x[:, 1:2] - xc))
        raise ValueError("Unsupported ic type {}".format(self.ibc_type[0]))

    def q_bc(self, x):
        if self.ibc_type[1] == "riemann":
            xc = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            ul = torch.tensor(1.0, device=x.device, dtype=x.dtype)
            ur = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            return ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
        if self.ibc_type[1] == "sine":
            return torch.zeros_like(x[:, 1:2])
        raise ValueError("Unsupported bc type {}".format(self.ibc_type[1]))

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
