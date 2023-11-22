import torch
from ml_collections import ConfigDict

import OriginPINN.basic


def gradients(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )


class EulerNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        self.u = OriginPINN.basic.Net(
            config.layer_sizes, config.activation, config.configuration
        )

        self.gamma = 1.4
        self.epsilon = 0.2
        if config.loss == "MSE":
            self.loss_fn = torch.nn.MSELoss()
        self.ibc_type = config.ibc_type

    def forward(self, x):
        return self.u(x)

    def interior_loss(self, x, weights=[5.0, 5.0, 1.0]):
        x = x.to(torch.float32)
        u = self.u(x)
        rho = u[:, 0:1]
        velocity = u[:, 1:2]
        pressure = u[:, 2:3]
        momentum = rho * velocity
        energy = pressure / (self.gamma - 1) + 0.5 * rho * velocity**2
        second_term = rho * velocity**2 + pressure
        third_term = (energy + pressure) * velocity
        dg_rho = gradients(rho, x)[0]
        dg_momentum = gradients(momentum, x)[0]
        dg_energy = gradients(energy, x)[0]
        dg_second = gradients(second_term, x)[0]
        dg_third = gradients(third_term, x)[0]
        rho_dt = dg_rho[:, 0:1]
        momentum_dt = dg_momentum[:, 0:1]
        momentum_dx = dg_momentum[:, 1:2]
        energy_dt = dg_energy[:, 0:1]
        second_dx = dg_second[:, 1:2]
        third_dx = dg_third[:, 1:2]
        eq1 = self.loss_fn(rho_dt + momentum_dx, torch.zeros_like(rho_dt))
        eq2 = self.loss_fn(momentum_dt + second_dx, torch.zeros_like(momentum_dt))
        eq3 = self.loss_fn(energy_dt + third_dx, torch.zeros_like(energy_dt))
        return weights[0] * eq1 + weights[1] * eq2 + weights[2] * eq3

    def init_loss(self, x_ic):
        x_ic = x_ic.to(torch.float32)
        u_ic_nn = self.u(x_ic)
        u_ic_true = self.u_ic(x_ic)
        L_rho = self.loss_fn(u_ic_nn[:, 0:1], u_ic_true[:, 0:1])
        L_velocity = self.loss_fn(u_ic_nn[:, 1:2], u_ic_true[:, 1:2])
        L_pressure = self.loss_fn(u_ic_nn[:, 2:3], u_ic_true[:, 2:3])
        return L_rho + L_velocity + L_pressure

    def bc_loss(self, x_bc):
        x_bc = x_bc.to(torch.float32)
        u_bc_nn = self.u(x_bc)
        u_bc_true = self.u_bc(x_bc)
        L_rho = self.loss_fn(u_bc_nn[:, 0:1], u_bc_true[:, 0:1])
        L_velocity = self.loss_fn(u_bc_nn[:, 1:2], u_bc_true[:, 1:2])
        L_pressure = self.loss_fn(u_bc_nn[:, 2:3], u_bc_true[:, 2:3])
        return L_rho + L_velocity + L_pressure

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
                1.0 + self.epsilon * torch.sin(5 * xc)
            ) * (x[:, 1:2] >= xc)
            velocity = velocity_l * (x[:, 1:2] < xc)
            pressure = pressure_l * (x[:, 1:2] < xc) + pressure_r * (x[:, 1:2] >= xc)
            return torch.cat((rho, velocity, pressure), dim=1)
        elif self.ibc_type[0] == "blast":
            xl = -0.2
            xr = 0.2
            pressure_l = torch.tensor(0.01)
            pressure_m = torch.tensor(1)
            pressure_r = torch.tensor(0.01)
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
                1.0 + self.epsilon * torch.sin(5 * xc)
            ) * (x[:, 1:2] >= xc)
            velocity = velocity_l * (x[:, 1:2] < xc)
            pressure = pressure_l * (x[:, 1:2] < xc) + pressure_r * (x[:, 1:2] >= xc)
            return torch.cat((rho, velocity, pressure), dim=1)
        elif self.ibc_type[1] == "blast":
            xl = -0.2
            xr = 0.2
            pressure_l = torch.tensor(0.01)
            pressure_m = torch.tensor(1)
            pressure_r = torch.tensor(0.01)
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
