import csv
import inspect
from pathlib import Path

import numpy as np
import torch
from absl import logging
from ml_collections import ConfigDict


def train(
    device: torch.device,
    datagenerator,
    x_test: np.ndarray,
    q_test: np.ndarray,
    model: torch.nn.Module,
    config: ConfigDict,
    csv_path: Path,
    model_dir: Path,
    lr_dir: Path,
):
    int_weights = None
    if hasattr(config, "int_weights"):
        int_weights = torch.tensor(config.int_weights, dtype=torch.float32).to(device)
    history_every = getattr(config, "history_every", 1000)
    log_every = getattr(config, "log_every", 1000)
    checkpoint_every = getattr(config, "checkpoint_every", 1000)

    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    q_test = torch.tensor(q_test, dtype=torch.float32).to(device)

    logging.info("Created the csv file")
    logging.info("----------Start training-----------")

    optimization_schedule = _build_optimization_schedule(model, config)
    active_phase = _phase_for_epoch(optimization_schedule, 0)
    optimizer = active_phase["optimizer"]

    initial_terms = _compute_loss_terms(
        model,
        *datagenerator.samples(),
        int_weights=int_weights,
        config=config,
    )
    history_terms = _resolve_history_terms(config, initial_terms)
    _write_history_header(csv_path, history_terms)
    _record_state(
        epoch=0,
        loss_terms=initial_terms,
        model=model,
        x_test=x_test,
        q_test=q_test,
        optimizer=optimizer,
        csv_path=csv_path,
        model_dir=model_dir,
        lr_dir=lr_dir,
        history_every=history_every,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        history_terms=history_terms,
    )

    for epoch in range(1, config.epochs + 1):
        active_phase = _phase_for_epoch(optimization_schedule, epoch)
        optimizer = active_phase["optimizer"]
        scheduler = active_phase["scheduler"]
        scheduler_every = active_phase["scheduler_every"]
        batch = datagenerator.samples()
        if active_phase["mode"] == "Adam":
            loss_terms = _compute_loss_terms(
                model,
                *batch,
                int_weights=int_weights,
                config=config,
            )
            optimizer.zero_grad()
            loss_terms["total"].backward()
            optimizer.step()
            if scheduler is not None and epoch % scheduler_every == 0:
                scheduler.step()
        elif active_phase["mode"] == "LBFGS":
            closure_cache = {}

            def closure():
                optimizer.zero_grad()
                current_terms = _compute_loss_terms(
                    model,
                    *batch,
                    int_weights=int_weights,
                    config=config,
                )
                current_terms["total"].backward()
                closure_cache["terms"] = _detach_loss_terms(current_terms)
                return current_terms["total"]

            optimizer.step(closure)
            loss_terms = closure_cache["terms"]
        else:
            raise ValueError("other optimizer have not been implemented")

        _record_state(
            epoch=epoch,
            loss_terms=loss_terms,
            model=model,
            x_test=x_test,
            q_test=q_test,
            optimizer=optimizer,
            csv_path=csv_path,
            model_dir=model_dir,
            lr_dir=lr_dir,
            history_every=history_every,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            history_terms=history_terms,
        )


def _build_optimization_schedule(model: torch.nn.Module, config: ConfigDict):
    if config.optimizer == "Adam":
        scheduler_every = _resolve_scheduler_every(
            getattr(config, "scheduler_every", 1000)
        )
        optimizer, scheduler = _build_adam_optimizer(
            model=model,
            lr=config.lr,
            decay=getattr(config, "decay", "Exponential"),
            decay_rate=getattr(config, "decay_rate", 0.99),
            total_epochs=config.epochs,
            scheduler_every=scheduler_every,
            scheduler_config=config,
        )
        return [
            dict(
                mode="Adam",
                start_epoch=1,
                end_epoch=config.epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                scheduler_every=scheduler_every,
            )
        ]

    if config.optimizer == "LBFGS":
        optimizer = _build_lbfgs_optimizer(
            model=model,
            lr=config.lr,
            config=config,
        )
        return [
            dict(
                mode="LBFGS",
                start_epoch=1,
                end_epoch=config.epochs,
                optimizer=optimizer,
                scheduler=None,
                scheduler_every=None,
            )
        ]

    if config.optimizer == "Adam_LBFGS":
        adam_epochs = int(getattr(config, "adam_epochs", 0))
        if adam_epochs <= 0 or adam_epochs >= config.epochs:
            raise ValueError(
                "Adam_LBFGS requires adam_epochs between 1 and epochs-1, got {}".format(
                    adam_epochs
                )
            )

        adam_scheduler_every = _resolve_scheduler_every(
            getattr(
                config,
                "adam_scheduler_every",
                getattr(config, "scheduler_every", 1000),
            )
        )
        adam_optimizer, adam_scheduler = _build_adam_optimizer(
            model=model,
            lr=getattr(config, "adam_lr", 1e-3),
            decay=getattr(
                config, "adam_decay", getattr(config, "decay", "Exponential")
            ),
            decay_rate=getattr(
                config, "adam_decay_rate", getattr(config, "decay_rate", 0.99)
            ),
            total_epochs=adam_epochs,
            scheduler_every=adam_scheduler_every,
            scheduler_config=config,
        )
        lbfgs_optimizer = _build_lbfgs_optimizer(
            model=model,
            lr=getattr(config, "lbfgs_lr", 1.0),
            config=config,
        )
        return [
            dict(
                mode="Adam",
                start_epoch=1,
                end_epoch=adam_epochs,
                optimizer=adam_optimizer,
                scheduler=adam_scheduler,
                scheduler_every=adam_scheduler_every,
            ),
            dict(
                mode="LBFGS",
                start_epoch=adam_epochs + 1,
                end_epoch=config.epochs,
                optimizer=lbfgs_optimizer,
                scheduler=None,
                scheduler_every=None,
            ),
        ]

    raise ValueError("other optimizer have not been implemented")


def _build_adam_optimizer(
    model: torch.nn.Module,
    lr: float,
    decay: str,
    decay_rate: float,
    total_epochs: int,
    scheduler_every: int,
    scheduler_config: ConfigDict,
):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    decay_name = str(decay).strip().lower()
    if decay_name == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    elif decay_name in {"cosine", "cosineannealing", "cosine_annealing"}:
        t_max = int(
            getattr(scheduler_config, "cosine_t_max", 0)
            or _default_cosine_t_max(
                total_epochs=total_epochs,
                scheduler_every=scheduler_every,
            )
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, t_max),
            eta_min=float(getattr(scheduler_config, "cosine_eta_min", 0.0)),
        )
    elif decay_name in {"none", "constant"}:
        scheduler = None
    else:
        raise ValueError("other decay have not been implemented")
    return optimizer, scheduler


def _resolve_scheduler_every(value):
    return max(1, int(value))


def _default_cosine_t_max(total_epochs: int, scheduler_every: int):
    return max(
        1, (int(total_epochs) + int(scheduler_every) - 1) // int(scheduler_every)
    )


def _build_lbfgs_optimizer(
    model: torch.nn.Module,
    lr: float,
    config: ConfigDict,
):
    line_search_fn = getattr(config, "lbfgs_line_search_fn", None)
    if isinstance(line_search_fn, str) and line_search_fn.lower() == "none":
        line_search_fn = None
    return torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=getattr(config, "lbfgs_max_iter", 1),
        max_eval=getattr(config, "lbfgs_max_eval", None),
        history_size=getattr(config, "lbfgs_history_size", 100),
        tolerance_grad=getattr(config, "lbfgs_tolerance_grad", 1e-7),
        tolerance_change=getattr(config, "lbfgs_tolerance_change", 1e-9),
        line_search_fn=line_search_fn,
    )


def _compute_loss_terms(
    model: torch.nn.Module,
    x_int: torch.Tensor,
    x_ic: torch.Tensor,
    x_bc: torch.Tensor,
    int_weights: torch.Tensor,
    config: ConfigDict,
):
    x_int = _prepare_batch_tensor(x_int)
    x_ic = _prepare_batch_tensor(x_ic)
    x_bc = _prepare_batch_tensor(x_bc)

    if hasattr(model, "compute_loss_terms"):
        loss_terms = model.compute_loss_terms(
            x_int=x_int,
            x_ic=x_ic,
            x_bc=x_bc,
            int_weights=int_weights,
        )
    else:
        res_loss, flux_loss = _call_interior_loss(
            model=model,
            x_int=x_int,
            int_weights=int_weights,
        )
        u_ic_loss, F_ic_loss = model.init_loss(x_ic)
        u_bc_loss, F_bc_loss = model.bc_loss(x_bc)
        loss_terms = {
            "res_loss": res_loss,
            "flux_loss": flux_loss,
            "u_ic": u_ic_loss,
            "F_ic": F_ic_loss,
            "u_bc": u_bc_loss,
            "F_bc": F_bc_loss,
        }

    total_loss = _sum_weighted_losses(loss_terms, _resolve_loss_weights(config))
    return {"total": total_loss, **loss_terms}


def _record_state(
    epoch: int,
    loss_terms,
    model: torch.nn.Module,
    x_test: torch.Tensor,
    q_test: torch.Tensor,
    optimizer,
    csv_path: Path,
    model_dir: Path,
    lr_dir: Path,
    history_every: int,
    log_every: int,
    checkpoint_every: int,
    history_terms: list[str],
):
    should_write_history = epoch % history_every == 0
    should_log = epoch % log_every == 0
    should_checkpoint = epoch % checkpoint_every == 0

    if not (should_write_history or should_log or should_checkpoint):
        return

    mae, l2re = _evaluate_test_metrics(model, x_test, q_test)

    lr = optimizer.param_groups[0]["lr"]

    row = [epoch, _to_float(loss_terms["total"])]
    for name in history_terms:
        row.append(_to_float(loss_terms[name]))
    row.extend([_to_float(mae), _to_float(l2re), lr])

    if should_write_history:
        with open(csv_path, "a+", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    if should_log:
        loss_log = " | ".join(
            "{} : {}".format(name, _to_float(loss_terms[name]))
            for name in history_terms
        )
        logging.info(
            "epoch : {}  | total : {} | {} | lr : {} | MAE : {} | L2RE : {}".format(
                epoch,
                _to_float(loss_terms["total"]),
                loss_log,
                lr,
                _to_float(mae),
                _to_float(l2re),
            )
        )

    if should_checkpoint:
        checkpoint_name = "model_{:02d}".format(epoch)
        torch.save(model.state_dict(), model_dir / checkpoint_name)
        torch.save(optimizer.state_dict(), lr_dir / checkpoint_name)


def _call_interior_loss(
    model: torch.nn.Module,
    x_int: torch.Tensor,
    int_weights: torch.Tensor | None,
):
    signature = inspect.signature(model.interior_loss)
    positional_params = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]

    supports_weights = len(positional_params) >= 2
    if supports_weights and int_weights is not None:
        return model.interior_loss(x_int, int_weights)
    return model.interior_loss(x_int)


def _evaluate_test_metrics(
    model: torch.nn.Module, x_test: torch.Tensor, q_test: torch.Tensor
):
    with torch.inference_mode():
        q_pred = model(x_test)
    mae = torch.nn.L1Loss()(q_test, q_pred)
    error_norm = torch.linalg.vector_norm((q_pred - q_test).reshape(-1), ord=2)
    reference_norm = torch.linalg.vector_norm(q_test.reshape(-1), ord=2)
    l2re = error_norm / reference_norm
    return mae, l2re


def _detach_loss_terms(loss_terms):
    detached = {}
    for name, value in loss_terms.items():
        if isinstance(value, torch.Tensor):
            detached[name] = value.detach()
        else:
            detached[name] = value
    return detached


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return float(value)


def _prepare_batch_tensor(x: torch.Tensor):
    x = x.detach()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    return x.requires_grad_(True)


def _resolve_loss_weights(config: ConfigDict):
    if hasattr(config, "loss_weights"):
        return {name: float(value) for name, value in dict(config.loss_weights).items()}

    if hasattr(config, "ratio"):
        ratio = list(config.ratio)
        return {
            "res_loss": float(ratio[0]),
            "flux_loss": float(ratio[1]),
            "u_ic": float(ratio[2]),
            "u_bc": float(ratio[3]),
        }

    raise ValueError("TrainConfig must provide loss_weights or ratio")


def _resolve_history_terms(config: ConfigDict, loss_terms: dict):
    if hasattr(config, "history_terms"):
        history_terms = list(config.history_terms)
    else:
        history_terms = [name for name in loss_terms.keys() if name != "total"]

    missing_terms = [name for name in history_terms if name not in loss_terms]
    if missing_terms:
        raise KeyError(
            "History terms {} not found in model outputs".format(missing_terms)
        )
    return history_terms


def _sum_weighted_losses(loss_terms: dict, loss_weights: dict):
    total = None
    for name, weight in loss_weights.items():
        if name not in loss_terms:
            raise KeyError("Loss term {} not found in model outputs".format(name))
        term = weight * loss_terms[name]
        total = term if total is None else total + term

    if total is None:
        raise ValueError("No weighted loss terms were configured")
    return total


def _write_history_header(csv_path: Path, history_terms: list[str]):
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total", *history_terms, "MAE", "L2RE", "lr"])


def _phase_for_epoch(schedule: list[dict], epoch: int):
    if epoch == 0:
        return schedule[0]

    for phase in schedule:
        if phase["start_epoch"] <= epoch <= phase["end_epoch"]:
            return phase
    raise ValueError("No optimization phase configured for epoch {}".format(epoch))
