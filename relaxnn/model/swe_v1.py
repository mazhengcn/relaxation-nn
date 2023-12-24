import torch
from ml_collections import ConfigDict
from model import basic
from utils import grad

gradients = grad.gradients

class SweNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.u = basic.Net(
            config.layer_sizes[0], config.activation[0], config.configuration[0]
        )
        self.flux = basic.Net(
            config.layer_sizes[1], config.activation[1], config.configuration[1]
        )
        if config.loss == "MSE":
            self.loss_fn = torch.nn.MSELoss()
        self.num_eq = config.layer_sizes[0][-1] + config.layer_sizes[1][-1]
        self.ibc_type = config.ibc_type

    def forward(self, x):
        return self.u(x)

    def interior_loss(self, x, weights=[1.0, 1.0, 1.0, 1.0]):
        if len(weights) != self.num_eq:
            raise ValueError("length of weights should be equal to number of equations")
        x = x.to(torch.float32)
        u = self.u(x)
        flux = self.flux(x)
        h = u[:, 0:1]
        velocity = u[:, 1:2]
        m = h * velocity
        flux1 = flux[:, 0:1]
        flux2 = flux[:, 1:2]
        flux1_true = m
        flux2_true = h * velocity**2 + 0.5 * h**2
        dh_g = gradients(h, x)[0]
        dm_g = gradients(m, x)[0]
        dflux1_g = gradients(flux1, x)[0]
        dflux2_g = gradients(flux2, x)[0]
        h_t = dh_g[:, 0:1]
        m_t = dm_g[:, 0:1]
        flux1_x = dflux1_g[:, 1:2]
        flux2_x = dflux2_g[:, 1:2]
        zeros = torch.zeros_like(h_t)
        L_eq1 = self.loss_fn(h_t + flux1_x, zeros)
        L_eq2 = self.loss_fn(m_t + flux2_x, zeros)
        L_flux1 = self.loss_fn(flux1, flux1_true)
        L_flux2 = self.loss_fn(flux2, flux2_true)
        return (
            weights[0] * L_eq1 + weights[1] * L_eq2,
            weights[2] * L_flux1 + weights[3] * L_flux2,
        )

    def init_loss(self, x_ic, weights=[1.0, 1.0, 1.0, 1.0]):
        x_ic = x_ic.to(torch.float32)
        u_ic_nn = self.u(x_ic)
        u_ic_true = self.u_ic(x_ic)
        F_ic_nn = self.flux(x_ic)
        F_ic_true = self.F_ic(x_ic)
        eq1 = self.loss_fn(u_ic_nn[:, 0:1], u_ic_true[:, 0:1])
        eq2 = self.loss_fn(u_ic_nn[:, 1:2], u_ic_true[:, 1:2])
        flux1 = self.loss_fn(F_ic_nn[:, 0:1], F_ic_true[:, 0:1])
        flux2 = self.loss_fn(F_ic_nn[:, 1:2], F_ic_true[:, 1:2])
        return (
            weights[0] * eq1 + weights[1] * eq2,
            weights[2] * flux1 + weights[3] * flux2,
        )

    def bc_loss(self, x_bc, weights=[1.0, 1.0, 1.0, 1.0]):
        x_bc = x_bc.to(torch.float32)
        u_bc_nn = self.u(x_bc)
        u_bc_true = self.u_bc(x_bc)
        F_bc_nn = self.flux(x_bc)
        F_bc_true = self.F_bc(x_bc)
        eq1 = self.loss_fn(u_bc_nn[:, 0:1], u_bc_true[:, 0:1])
        eq2 = self.loss_fn(u_bc_nn[:, 1:2], u_bc_true[:, 1:2])
        flux1 = self.loss_fn(F_bc_nn[:, 0:1], F_bc_true[:, 0:1])
        flux2 = self.loss_fn(F_bc_nn[:, 1:2], F_bc_true[:, 1:2])
        return (
            weights[0] * eq1 + weights[1] * eq2,
            weights[2] * flux1 + weights[3] * flux2,
        )

    def u_ic(self, x):
        if self.ibc_type[0] == "dam-break":
            xc = torch.tensor(0.0)
            hl = torch.tensor(1.0)
            hr = torch.tensor(0.5)
            ul = torch.tensor(0.0)
            ur = torch.tensor(0.0)
            h = hl * (x[:, 1:2] <= xc) + hr * (x[:, 1:2] > xc)
            u = ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
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
            ul = torch.tensor(0.0)
            ur = torch.tensor(0.0)
            h = hl * (x[:, 1:2] <= xc) + hr * (x[:, 1:2] > xc)
            u = ul * (x[:, 1:2] <= xc) + ur * (x[:, 1:2] > xc)
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
        F_u = height * velocity
        F_b = height * velocity**2 + 0.5 * height**2
        return torch.cat((F_u, F_b), dim=1)

    def F_bc(self, x):
        q = self.u_bc(x)
        height = q[:, 0:1]
        velocity = q[:, 1:2]
        F_u = height * velocity
        F_b = height * velocity**2 + 0.5 * height**2
        return torch.cat((F_u, F_b), dim=1)
