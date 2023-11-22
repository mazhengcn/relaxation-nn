import torch
from ml_collections import ConfigDict

import OriginPINN.basic


def gradients(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )


class SweNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.u = OriginPINN.basic.Net(
            config.layer_sizes, config.activation, config.configuration
        )
        if config.loss == "MSE":
            self.loss_fn = torch.nn.MSELoss()
        self.ibc_type = config.ibc_type

    def forward(self, x):
        return self.u(x)

    def interior_loss(self, x):
        x = x.to(torch.float32)
        h = self.u(x)[:, 0:1]
        u = self.u(x)[:, 1:2]
        M = h * u
        E = h * u**2 + 0.5 * h**2
        dh_g = gradients(h, x)[0]
        dM_g = gradients(M, x)[0]
        dE_g = gradients(E, x)[0]
        h_t = dh_g[:, 0:1]
        M_t = dM_g[:, 0:1]
        M_x = dM_g[:, 1:2]
        E_x = dE_g[:, 1:2]
        res1 = self.loss_fn(h_t + M_x, torch.zeros_like(h_t))
        res2 = self.loss_fn(M_t + E_x, torch.zeros_like(M_t))
        return res1 + res2

    def init_loss(self, x_ic):
        x_ic = x_ic.to(torch.float32)
        u_ic_nn = self.u(x_ic)
        u_ic = self.u_ic(x_ic)
        f = self.loss_fn(u_ic_nn, u_ic)
        return f

    def bc_loss(self, x_bc):
        x_bc = x_bc.to(torch.float32)
        u_bc_nn = self.u(x_bc)
        u_bc = self.u_bc(x_bc)
        f = self.loss_fn(u_bc_nn, u_bc)
        return f

    def u_ic(self, x):
        if self.ibc_type[0] == "dam-break":
            xc = torch.tensor(0.0)
            hl = torch.tensor(1.0)
            hr = torch.tensor(0.5)
            ul = torch.tensor(0.0)
            ur = torch.tensor(0.0)
            h = hl * (x[:, 1:2] <= xc) + hr * (x[:, 1:2] > xc)
            u = ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
            return torch.cat([h, u], dim=1)
        elif self.ibc_type[0] == "2shock":
            xc = torch.tensor(0.0)
            ul = torch.tensor(1.0)
            ur = torch.tensor(-1.0)
            u = ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
            h = torch.ones_like(u)
            return torch.cat([h, u], dim=1)
        else:
            raise ValueError("other ibc type have not been implemented")

    def u_bc(self, x):
        if self.ibc_type[1] == "dam-break":
            xc = torch.tensor(0.0)
            hl = torch.tensor(1.0)
            hr = torch.tensor(0.5)
            ul = torch.tensor(0.0)
            ur = torch.tensor(0.0)
            h = hl * (x[:, 1:2] <= xc) + hr * (x[:, 1:2] > xc)
            u = ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
            return torch.cat([h, u], dim=1)
        elif self.ibc_type[1] == "2shock":
            xc = torch.tensor(0.0)
            ul = torch.tensor(1.0)
            ur = torch.tensor(-1.0)
            u = ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
            h = torch.ones_like(u)
            return torch.cat([h, u], dim=1)
        else:
            raise ValueError("other ibc type have not been implemented")
