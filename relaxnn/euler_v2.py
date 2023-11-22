import torch
from ml_collections import ConfigDict

import OriginRela.basic


def gradients(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )


class EulerNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.u = OriginRela.basic.Net(
            config.layer_sizes[0], config.activation[0], config.configuration[0]
        )
        self.flux = OriginRela.basic.Net(
            config.layer_sizes[1], config.activation[1], config.configuration[1]
        )
        self.gamma = 1.4
        self.epsilon = 0.2
        if config.loss == "MSE":
            self.loss_fn = torch.nn.MSELoss()
        self.num_eq = config.layer_sizes[0][-1] + config.layer_sizes[1][-1]
        self.ibc_type = config.ibc_type

    def forward(self, x):
        return self.u(x)

    def interior_loss(self, x, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        if len(weights) != self.num_eq:
            raise ValueError("length of weights should be equal to number of equations")
        x = x.to(torch.float32)
        u = self.u(x)
        flux = self.flux(x)
        rho = u[:, 0:1]
        velocity = u[:, 1:2]
        pressure = u[:, 2:3]
        momentum = rho * velocity
        energy = pressure / (self.gamma - 1) + 0.5 * rho * velocity**2
        flux1_true = rho * velocity**2 + pressure
        flux2_true = velocity * (energy + pressure)
        flux1 = flux[:, 0:1]
        flux2 = flux[:, 1:2]
        dg_rho = gradients(rho, x)[0]
        dg_momentum = gradients(momentum, x)[0]
        dg_energy = gradients(energy, x)[0]
        dg_flux1 = gradients(flux1, x)[0]
        dg_flux2 = gradients(flux2, x)[0]
        rho_dt = dg_rho[:, 0:1]
        momentum_dt = dg_momentum[:, 0:1]
        momentum_dx = dg_momentum[:, 1:2]
        energy_dt = dg_energy[:, 0:1]
        flux1_dx = dg_flux1[:, 1:2]
        flux2_dx = dg_flux2[:, 1:2]
        zeros = torch.zeros_like(rho_dt)
        eq1 = self.loss_fn(rho_dt + momentum_dx, zeros)
        eq2 = self.loss_fn(momentum_dt + flux1_dx, zeros)
        eq3 = self.loss_fn(energy_dt + flux2_dx, zeros)
        eqflux1 = self.loss_fn(flux1 - flux1_true, zeros)
        eqflux2 = self.loss_fn(flux2 - flux2_true, zeros)
        relaxate = weights[0] * eq1 + weights[1] * eq2 + weights[2] * eq3
        flux = weights[3] * eqflux1 + weights[4] * eqflux2
        return relaxate, flux

    def init_loss(self, x_ic, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        x_ic = x_ic.to(torch.float32)
        u_ic_nn = self.u(x_ic)
        u_ic_true = self.u_ic(x_ic)
        F_ic_nn = self.flux(x_ic)
        F_ic_true = self.F_ic(x_ic)
        L_rho = self.loss_fn(u_ic_nn[:, 0:1], u_ic_true[:, 0:1])
        L_velocity = self.loss_fn(u_ic_nn[:, 1:2], u_ic_true[:, 1:2])
        L_pressure = self.loss_fn(u_ic_nn[:, 2:3], u_ic_true[:, 2:3])
        L_flux1 = self.loss_fn(F_ic_nn[:, 0:1], F_ic_true[:, 0:1])
        L_flux2 = self.loss_fn(F_ic_nn[:, 1:2], F_ic_true[:, 1:2])
        init_u = weights[0] * L_rho + weights[1] * L_velocity + weights[2] * L_pressure
        init_F = weights[3] * L_flux1 + weights[4] * L_flux2
        return init_u, init_F

    def bc_loss(self, x_bc, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        x_bc = x_bc.to(torch.float32)
        u_bc_nn = self.u(x_bc)
        u_bc_true = self.u_bc(x_bc)
        F_bc_nn = self.flux(x_bc)
        F_bc_true = self.F_bc(x_bc)
        L_rho = self.loss_fn(u_bc_nn[:, 0:1], u_bc_true[:, 0:1])
        L_velocity = self.loss_fn(u_bc_nn[:, 1:2], u_bc_true[:, 1:2])
        L_pressure = self.loss_fn(u_bc_nn[:, 2:3], u_bc_true[:, 2:3])
        L_flux1 = self.loss_fn(F_bc_nn[:, 0:1], F_bc_true[:, 0:1])
        L_flux2 = self.loss_fn(F_bc_nn[:, 1:2], F_bc_true[:, 1:2])
        bc_u = weights[0] * L_rho + weights[1] * L_velocity + weights[2] * L_pressure
        bc_F = weights[3] * L_flux1 + weights[4] * L_flux2
        return bc_u, bc_F

    def u_ic(self, x):
        if self.ibc_type[0] == "shock_tube":
            xc = torch.tensor(0.0)
            rho_l = torch.tensor(1.0)
            rho_r = torch.tensor(0.125)
            pressure_l = torch.tensor(1.0)
            pressure_r = torch.tensor(0.1)
            rho = rho_l * (x[:, 1:2] <= xc) + rho_r * (x[:, 1:2] > xc)
            velocity = torch.zeros_like(rho)
            pressure = pressure_l * (x[:, 1:2] <= xc) + pressure_r * (x[:, 1:2] > xc)
            return torch.cat((rho, velocity, pressure), dim=1)
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
            return torch.cat((rho, velocity, pressure), dim=1)
        elif self.ibc_type[0] == "shu_osher":
            xc = torch.tensor(-4.0)
            rho_l = torch.tensor(3.857143)
            velocity_l = torch.tensor(2.629369)
            pressure_l = torch.tensor(10.33333)
            pressure_r = torch.tensor(1.0)
            rho = rho_l * (x[:, 1:2] < xc) + (
                1.0 + self.epsilon * torch.sin(5 * x[:, 1:2])
            ) * (x[:, 1:2] >= xc)
            velocity = velocity_l * (x[:, 1:2] < xc)
            pressure = pressure_l * (x[:, 1:2] < xc) + pressure_r * (x[:, 1:2] >= xc)
            return torch.cat((rho, velocity, pressure), dim=1)
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
            return torch.cat((rho, velocity, pressure), dim=1)
        else:
            raise ValueError("other ibc type have not been implemented")

    def u_bc(self, x):
        if self.ibc_type[1] == "shock_tube":
            xc = torch.tensor(0.0)
            rho_l = torch.tensor(1.0)
            rho_r = torch.tensor(0.125)
            pressure_l = torch.tensor(1.0)
            pressure_r = torch.tensor(0.1)
            rho = rho_l * (x[:, 1:2] <= xc) + rho_r * (x[:, 1:2] > xc)
            velocity = torch.zeros_like(rho)
            pressure = pressure_l * (x[:, 1:2] <= xc) + pressure_r * (x[:, 1:2] > xc)
            return torch.cat((rho, velocity, pressure), dim=1)
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
            return torch.cat((rho, velocity, pressure), dim=1)
        elif self.ibc_type[1] == "shu_osher":
            xc = torch.tensor(-4.0)
            rho_l = torch.tensor(3.857143)
            velocity_l = torch.tensor(2.629369)
            pressure_l = torch.tensor(10.33333)
            pressure_r = torch.tensor(1.0)
            rho = rho_l * (x[:, 1:2] < xc) + (
                1.0 + self.epsilon * torch.sin(5 * x[:, 1:2])
            ) * (x[:, 1:2] >= xc)
            velocity = velocity_l * (x[:, 1:2] < xc)
            pressure = pressure_l * (x[:, 1:2] < xc) + pressure_r * (x[:, 1:2] >= xc)
            return torch.cat((rho, velocity, pressure), dim=1)
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
            return torch.cat((rho, velocity, pressure), dim=1)
        else:
            raise ValueError("other ibc type have not been implemented")

    def F_ic(self, x):
        q = self.u_ic(x)
        rho = q[:, 0:1]
        velocity = q[:, 1:2]
        pressure = q[:, 2:3]
        energy = pressure / (self.gamma - 1) + 0.5 * rho * velocity**2
        flux1 = rho * velocity**2 + pressure
        flux2 = velocity * (energy + pressure)
        return torch.cat((flux1, flux2), dim=1)

    def F_bc(self, x):
        q = self.u_bc(x)
        rho = q[:, 0:1]
        velocity = q[:, 1:2]
        pressure = q[:, 2:3]
        energy = pressure / (self.gamma - 1) + 0.5 * rho * velocity**2
        flux1 = rho * velocity**2 + pressure
        flux2 = velocity * (energy + pressure)
        return torch.cat((flux1, flux2), dim=1)
