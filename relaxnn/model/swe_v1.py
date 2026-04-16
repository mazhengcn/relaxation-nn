import torch
from ml_collections import ConfigDict
from torch.func import jacrev, vmap

from relaxnn.model import basic


class SweNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self._state_net = basic.Net(
            config.layer_sizes[0],
            config.activation[0],
            config.configuration[0],
            config.initialization[0],
        )
        self._flux_net = basic.Net(
            config.layer_sizes[1],
            config.activation[1],
            config.configuration[1],
            config.initialization[1],
        )
        if config.loss == "MSE":
            self.loss_fn = basic.PDElossfn()
        self.ibc_type = config.ibc_type

    def forward(self, x):
        return self._state_net(x)

    def q(self, t, x):
        inputs = torch.hstack((t, x))
        h, u = self._split_state(self.forward(inputs))
        momentum = h * u
        return torch.cat((h, momentum), dim=-1)

    def f(self, t, x):
        inputs = torch.hstack((t, x))
        return self._flux_net(inputs)

    def flux(self, x):
        return self._flux_net(x)

    def flux_true(self, x):
        h, u = self._split_state(self.forward(x))
        momentum = h * u
        hu2p = h * u**2 + 0.5 * h**2
        return torch.cat((momentum, hu2p), dim=-1)

    def interior_loss(self, x):
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
        return self._target_state(x, self.ibc_type[0])

    def q_bc(self, x):
        return self._target_state(x, self.ibc_type[1])

    def F_ic(self, x):
        return self._target_flux(self.q_ic(x))

    def F_bc(self, x):
        return self._target_flux(self.q_bc(x))

    @staticmethod
    def _split_state(state):
        return state[..., 0:1], state[..., 1:2]

    def _target_state(self, x, ibc_type):
        x_coord = x[:, 1:2]
        if ibc_type == "dam-break":
            h = torch.where(x_coord <= 0.0, 1.0, 0.5)
            u = torch.zeros_like(h)
            return torch.cat((h, u), dim=-1)
        if ibc_type == "2shock":
            h = torch.ones_like(x_coord)
            u = torch.where(x_coord <= 0.0, 1.0, -1.0)
            return torch.cat((h, u), dim=-1)
        raise ValueError("other ibc type have not been implemented")

    def _target_flux(self, state):
        h, u = self._split_state(state)
        momentum = h * u
        hu2p = h * u**2 + 0.5 * h**2
        return torch.cat((momentum, hu2p), dim=-1)
