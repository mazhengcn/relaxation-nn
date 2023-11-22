"""The version to study the height and velocity"""
import torch
from ml_collections import ConfigDict

import OriginRela.basic


def gradients(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )


class SweNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.u = OriginRela.basic.Net(
            config.layer_sizes[0], config.activation[0], config.configuration[0]
        )
        self.Flux = OriginRela.basic.Net(
            config.layer_sizes[1], config.activation[1], config.configuration[1]
        )
        if config.loss == "MSE":
            self.loss_fn = torch.nn.MSELoss()
        self.ibc_type = config.ibc_type

    def forward(self, x):
        return self.u(x)

    def interior_loss(self, x):
        x = x.to(torch.float32)
        u = self.u(x)
        F = self.Flux(x)
        h = u[:, 0:1]
        velocity = u[:, 1:2]
        m = h * velocity
        E = h * velocity**2 + 0.5 * h**2
        dh_g = gradients(h, x)[0]
        dm_g = gradients(m, x)[0]
        dF_g = gradients(F, x)[0]
        dE_g = gradients(E, x)[0]
        h_t = dh_g[:, 0:1]
        m_t = dm_g[:, 0:1]
        F_x = dF_g[:, 1:2]
        E_x = dE_g[:, 1:2]
        f1 = self.loss_fn(h_t + F_x, torch.zeros_like(h_t))
        f2 = self.loss_fn(m_t + E_x, torch.zeros_like(m_t))
        f3 = self.loss_fn(F, m)
        return f1 + f2, f3

    def init_loss(self, x_ic):
        x_ic = x_ic.to(torch.float32)
        u_ic_nn = self.u(x_ic)
        F_ic_nn = self.Flux(x_ic)
        f1 = self.loss_fn(u_ic_nn, self.u_ic(x_ic))
        f2 = self.loss_fn(F_ic_nn, self.F_ic(x_ic))
        return f1, f2

    def bc_loss(self, x_bc):
        x_bc = x_bc.to(torch.float32)
        u_bc_nn = self.u(x_bc)
        F_bc_nn = self.Flux(x_bc)
        f1 = self.loss_fn(u_bc_nn, self.u_bc(x_bc))
        f2 = self.loss_fn(F_bc_nn, self.F_bc(x_bc))
        return f1, f2

    def u_ic(self, x):
        if self.ibc_type[0] == "dam-break":
            xc = torch.tensor(0.0)
            hl = torch.tensor(1.0)
            hr = torch.tensor(0.5)
            h = hl * (x[:, 1:2] <= xc) + hr * (x[:, 1:2] > xc)
            u = torch.zeros_like(h)
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

    def u_bc(self, x):
        if self.ibc_type[1] == "dam-break":
            xc = torch.tensor(0.0)
            hl = torch.tensor(1.0)
            hr = torch.tensor(0.5)
            h = hl * (x[:, 1:2] <= xc) + hr * (x[:, 1:2] > xc)
            u = torch.zeros_like(h)
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
        q = self.u_ic(x)
        height = q[:, 0:1]
        velocity = q[:, 1:2]
        F_true = height * velocity
        return F_true

    def F_bc(self, x):
        q = self.u_bc(x)
        height = q[:, 0:1]
        velocity = q[:, 1:2]
        F_true = height * velocity
        return F_true
