import math

import torch
import torch.nn.functional as F
from ml_collections import ConfigDict
from shared.model import basic


def _lp_parameter_regularization(module: torch.nn.Module, order: int = 2):
    reg_loss = None
    for name, param in module.named_parameters():
        if "weight" not in name and "bias" not in name:
            continue
        term = torch.norm(param, p=order)
        reg_loss = term if reg_loss is None else reg_loss + term
    if reg_loss is None:
        return torch.tensor(0.0)
    return reg_loss


class BurgersNet(torch.nn.Module):
    def __init__(self, config: ConfigDict):
        super().__init__()
        solution_activation = getattr(config, "solution_activation", config.activation)
        test_activation = getattr(config, "test_activation", solution_activation)
        solution_configuration = getattr(
            config, "solution_configuration", config.configuration
        )
        test_configuration = getattr(config, "test_configuration", solution_configuration)
        initialization = getattr(config, "initialization", "xavier_uniform")
        test_initialization = getattr(config, "test_initialization", initialization)

        solution_layers = getattr(
            config, "solution_layer_sizes", getattr(config, "layer_sizes", None)
        )
        if solution_layers is None:
            raise ValueError("NetConfig must provide solution_layer_sizes or layer_sizes")
        test_layers = getattr(
            config, "test_layer_sizes", getattr(config, "test_layer_sizes", None)
        )
        if test_layers is None:
            raise ValueError("NetConfig must provide test_layer_sizes")

        self._solution_net = basic.Net(
            solution_layers,
            solution_activation,
            solution_configuration,
            initialization=initialization,
        )
        self._test_net = basic.Net(
            test_layers,
            test_activation,
            test_configuration,
            initialization=test_initialization,
        )
        self._test_initialization = test_initialization
        self.ibc_type = list(config.ibc_type)
        self.loss_fn = basic.PDElossfn()
        self.entropy_norm = getattr(config, "entropy_norm", "H1")
        self.cutoff = getattr(config, "cutoff", "def_max")
        self.weak_form = getattr(config, "weak_form", "partial")
        self.c_mode = getattr(config, "c_mode", "max")
        self.entropy_c_samples = int(getattr(config, "entropy_c_samples", 101))
        self.c_range_factor = float(getattr(config, "c_range_factor", 2.0))
        self.sign_smoothing = float(getattr(config, "sign_smoothing", 1e-2))
        self.use_relu = bool(getattr(config, "use_relu", True))
        self.p = int(getattr(config, "p", 2))
        self.eps = float(getattr(config, "eps", 1e-12))
        self.data_weight = float(getattr(config, "data_weight", 10.0))
        self.residual_weight = float(getattr(config, "residual_weight", 1.0))
        self.solution_regularization = float(
            getattr(config, "solution_regularization", 0.0)
        )
        self.test_regularization = float(getattr(config, "test_regularization", 0.0))
        self.domain_min = torch.tensor(getattr(config, "domain_min", [0.0, -1.0]))
        self.domain_max = torch.tensor(getattr(config, "domain_max", [1.0, 1.0]))
        self._entropy_c_cache = {}

    def forward(self, x):
        return self._solution_net(x)

    def flux(self, x):
        u = self.forward(x)
        return 0.5 * u**2

    def test_network(self, x):
        return self._test_net(x)

    def solution_parameters(self):
        return self._solution_net.parameters()

    def test_parameters(self):
        return self._test_net.parameters()

    def reset_test_network(self):
        self._test_net._initialize_layers(self._test_initialization)

    def q_ic(self, x):
        if self.ibc_type[0] == "sine":
            return -torch.sin(math.pi * x[:, 1:2])
        if self.ibc_type[0] == "riemann":
            return torch.where(
                x[:, 1:2] <= 0.0,
                torch.ones_like(x[:, 1:2]),
                torch.zeros_like(x[:, 1:2]),
            )
        raise ValueError("Unsupported ic type {}".format(self.ibc_type[0]))

    def q_bc(self, x):
        if self.ibc_type[1] == "sine":
            return torch.zeros_like(x[:, 1:2])
        if self.ibc_type[1] == "riemann":
            return torch.where(
                x[:, 1:2] <= 0.0,
                torch.ones_like(x[:, 1:2]),
                torch.zeros_like(x[:, 1:2]),
            )
        raise ValueError("Unsupported bc type {}".format(self.ibc_type[1]))

    def initial_loss(self, x_ic):
        return self.loss_fn(self.forward(x_ic), self.q_ic(x_ic))

    def boundary_loss(self, x_bc):
        return self.loss_fn(self.forward(x_bc), self.q_bc(x_bc))

    def _smooth_sign(self, x):
        return 2.0 * torch.sigmoid(x / self.sign_smoothing) - 1.0

    def _smooth_abs(self, x):
        mu = self.sign_smoothing
        return 2.0 * mu * (F.softplus(x / mu) + math.log(0.5)) - x

    def _phi(self, x):
        theta = self.test_network(x).reshape(-1)
        return self._cutoff_weight(x).reshape(-1) * theta.square()

    def _cutoff_weight(self, x):
        device = x.device
        dtype = x.dtype
        domain_min = self.domain_min.to(device=device, dtype=dtype)
        domain_max = self.domain_max.to(device=device, dtype=dtype)
        spatial_coords = x[:, 1:]
        spatial_min = domain_min[1:]
        spatial_max = domain_max[1:]
        half_width = 0.5 * (spatial_max - spatial_min)
        center = 0.5 * (spatial_max + spatial_min)
        normalized = (spatial_coords - center) / half_width

        cutoff_name = str(self.cutoff).strip().lower()
        if cutoff_name in {"def_max", "bump"}:
            inside = (1.0 - normalized.abs()) > 0.0
            values = torch.zeros_like(normalized)
            safe = inside & (normalized.abs() < 1.0)
            values = torch.where(
                safe,
                torch.exp(1.0 / (normalized.square() - 1.0)),
                values,
            )
            weight = values.prod(dim=-1, keepdim=True)
            return weight / weight.amax().clamp_min(self.eps)

        if cutoff_name in {"quad", "quadratic"}:
            weight = (1.0 - normalized.square()).clamp_min(0.0).prod(dim=-1, keepdim=True)
            return weight / weight.amax().clamp_min(self.eps)

        raise ValueError("Unsupported cutoff {}".format(self.cutoff))

    def _norm(self, phi, phi_x):
        if self.entropy_norm == "H1":
            return torch.mean(torch.abs(phi) ** self.p + torch.abs(phi_x) ** self.p) ** (
                1.0 / self.p
            )
        if self.entropy_norm == "L2":
            return torch.mean(torch.abs(phi) ** self.p) ** (1.0 / self.p)
        if self.entropy_norm == "H1s":
            return torch.mean(torch.abs(phi_x) ** self.p) ** (1.0 / self.p)
        if self.entropy_norm in {None, "None"}:
            return torch.tensor(1.0, device=phi.device, dtype=phi.dtype)
        raise ValueError("Unsupported entropy_norm {}".format(self.entropy_norm))

    def _entropy_c_values(self, device, dtype):
        cache_key = (device.type, device.index, str(dtype), self.entropy_c_samples)
        cached = self._entropy_c_cache.get(cache_key)
        if cached is not None:
            return cached

        probe = torch.linspace(-1.0, 1.0, 2049, device=device, dtype=dtype).reshape(-1, 1)
        u0 = self.q_ic(torch.cat([torch.zeros_like(probe), probe], dim=-1))
        min_c = self.c_range_factor * torch.min(u0)
        max_c = self.c_range_factor * torch.max(u0)
        c_values = torch.linspace(
            min_c.item(),
            max_c.item(),
            self.entropy_c_samples,
            device=device,
            dtype=dtype,
        )
        self._entropy_c_cache[cache_key] = c_values
        return c_values

    def adversarial_residual(self, x_int):
        x_int = x_int.to(torch.float32)
        x_int.requires_grad_(True)

        u = self.forward(x_int).reshape(-1)
        phi = self._phi(x_int)

        grad_u = torch.autograd.grad(
            u,
            x_int,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]
        grad_phi = torch.autograd.grad(
            phi,
            x_int,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
        )[0]

        grad_u_t = grad_u[:, 0]
        grad_phi_t = grad_phi[:, 0]
        grad_phi_x = grad_phi[:, 1]

        norm_test = self._norm(phi, grad_phi_x).square().clamp_min(self.eps)
        c_values = self._entropy_c_values(x_int.device, x_int.dtype)

        c_column = c_values.reshape(-1, 1)
        u_row = u.reshape(1, -1)
        phi_row = phi.reshape(1, -1)
        grad_u_t_row = grad_u_t.reshape(1, -1)
        grad_phi_t_row = grad_phi_t.reshape(1, -1)
        grad_phi_x_row = grad_phi_x.reshape(1, -1)
        flux_gap = 0.5 * u_row.square() - 0.5 * c_column.square()
        sign_term = self._smooth_sign(u_row - c_column)

        if self.weak_form == "partial":
            residuals = torch.mean(
                sign_term
                * (grad_u_t_row * phi_row - grad_phi_x_row * flux_gap),
                dim=1,
            )
            if self.use_relu:
                residuals = torch.relu(residuals) + self.eps
            residuals = residuals.square()
        elif self.weak_form == "full":
            residuals = torch.relu(
                -torch.mean(
                    self._smooth_abs(u_row - c_column) * grad_phi_t_row
                    + sign_term * grad_phi_x_row * flux_gap,
                    dim=1,
                )
            ).square()
        else:
            raise ValueError("Unsupported weak_form {}".format(self.weak_form))

        res_mean = residuals.mean()
        res_max = residuals.amax()

        if self.c_mode == "max":
            res_raw = res_max
        elif self.c_mode == "mean":
            res_raw = res_mean
        elif self.c_mode == "both":
            res_raw = res_mean + res_max
        else:
            raise ValueError("Unsupported c_mode {}".format(self.c_mode))

        return {
            "res_loss": res_raw / norm_test,
            "res_raw": res_raw,
            "test_norm": norm_test,
        }

    def compute_min_loss_terms(self, x_int, x_ic, x_bc):
        residual_terms = self.adversarial_residual(x_int)
        u_ic = self.initial_loss(x_ic)
        u_bc = self.boundary_loss(x_bc)
        num_ic = int(x_ic.shape[0])
        num_bc = int(x_bc.shape[0])
        total_data_points = num_ic + num_bc
        if total_data_points > 0:
            data_loss = (
                u_ic * float(num_ic) + u_bc * float(num_bc)
            ) / float(total_data_points)
        else:
            data_loss = torch.tensor(0.0, device=x_int.device, dtype=x_int.dtype)
        sol_reg = _lp_parameter_regularization(self._solution_net)
        test_reg = _lp_parameter_regularization(self._test_net)

        total = (
            self.data_weight * data_loss
            + self.residual_weight * residual_terms["res_loss"]
            + self.solution_regularization * sol_reg
            + self.test_regularization * test_reg
        )
        return {
            "total": total,
            "data_loss": data_loss,
            "res_loss": residual_terms["res_loss"],
            "res_raw": residual_terms["res_raw"],
            "u_ic": u_ic,
            "u_bc": u_bc,
            "test_norm": residual_terms["test_norm"],
            "sol_reg": sol_reg,
            "test_reg": test_reg,
        }

    def compute_max_loss_terms(self, x_int):
        residual_terms = self.adversarial_residual(x_int)
        max_loss = -torch.log(residual_terms["res_loss"].clamp_min(self.eps))
        return {
            "total": max_loss,
            "max_loss": max_loss,
            "res_loss": residual_terms["res_loss"],
            "res_raw": residual_terms["res_raw"],
            "test_norm": residual_terms["test_norm"],
        }
