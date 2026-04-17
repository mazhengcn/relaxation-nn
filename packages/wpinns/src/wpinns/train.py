import csv
import time
from pathlib import Path

import torch
from absl import logging
from ml_collections import ConfigDict


def train(
    device: torch.device,
    datagenerator,
    x_test,
    q_test,
    model: torch.nn.Module,
    config: ConfigDict,
    csv_path: Path,
    model_dir: Path,
    lr_dir: Path,
):
    history_every = int(getattr(config, "history_every", 100))
    log_every = int(getattr(config, "log_every", 100))
    checkpoint_every = int(getattr(config, "checkpoint_every", 1000))
    maximize_steps = int(getattr(config, "maximize_steps", 1))
    minimize_steps = int(getattr(config, "minimize_steps", 1))
    test_reset_every = int(getattr(config, "test_reset_every", 0))

    x_test = torch.tensor(x_test, dtype=torch.float32, device=device)
    q_test = torch.tensor(q_test, dtype=torch.float32, device=device)

    solution_optimizer = torch.optim.Adam(
        model.solution_parameters(),
        lr=float(getattr(config, "solution_lr", 1e-3)),
        amsgrad=bool(getattr(config, "solution_amsgrad", True)),
    )
    test_optimizer = torch.optim.Adam(
        model.test_parameters(),
        lr=float(getattr(config, "test_lr", 1e-3)),
        amsgrad=bool(getattr(config, "test_amsgrad", True)),
    )
    solution_scheduler = _build_scheduler(
        solution_optimizer,
        getattr(config, "solution_decay", "CosineAnnealing"),
        total_epochs=int(config.epochs),
        eta_min=float(getattr(config, "solution_cosine_eta_min", 0.0)),
    )
    test_scheduler = _build_scheduler(
        test_optimizer,
        getattr(config, "test_decay", "CosineAnnealing"),
        total_epochs=int(config.epochs),
        eta_min=float(getattr(config, "test_cosine_eta_min", 0.0)),
    )

    logging.info("Created the csv file")
    logging.info("----------Start adversarial training-----------")
    start_time = time.time()

    history_terms = [
        "data_loss",
        "res_loss",
        "res_raw",
        "u_ic",
        "u_bc",
        "test_norm",
        "sol_reg",
        "test_reg",
        "max_loss",
    ]
    _write_history_header(csv_path, history_terms)

    summary = {
        "best_mae": float("inf"),
        "best_l2re": float("inf"),
        "best_mae_epoch": None,
        "best_l2re_epoch": None,
        "status": "running",
    }

    initial_batch = _prepare_batch(datagenerator.samples(), device)
    initial_metrics = _evaluate_epoch(
        model=model,
        batch=initial_batch,
        x_test=x_test,
        q_test=q_test,
        solution_optimizer=solution_optimizer,
        test_optimizer=test_optimizer,
        epoch=0,
    )
    _record_state(
        metrics=initial_metrics,
        csv_path=csv_path,
        model=model,
        model_dir=model_dir,
        lr_dir=lr_dir,
        solution_optimizer=solution_optimizer,
        test_optimizer=test_optimizer,
        solution_scheduler=solution_scheduler,
        test_scheduler=test_scheduler,
        history_every=history_every,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        is_final=False,
        history_terms=history_terms,
        summary=summary,
    )

    completed_normally = True
    for epoch in range(1, int(config.epochs) + 1):
        if test_reset_every > 0 and epoch % test_reset_every == 0:
            model.reset_test_network()
            test_optimizer.state.clear()

        batch = _prepare_batch(datagenerator.samples(), device)
        min_terms = None
        max_terms = None

        for _ in range(maximize_steps):
            test_optimizer.zero_grad()
            max_terms = model.compute_max_loss_terms(
                x_int=batch[0].detach().clone().requires_grad_(True)
            )
            max_terms["total"].backward()
            test_optimizer.step()

        for _ in range(minimize_steps):
            solution_optimizer.zero_grad()
            min_terms = model.compute_min_loss_terms(
                x_int=batch[0].detach().clone().requires_grad_(True),
                x_ic=batch[1].detach().clone().requires_grad_(True),
                x_bc=batch[2].detach().clone().requires_grad_(True),
            )
            min_terms["total"].backward()
            solution_optimizer.step()

        if solution_scheduler is not None:
            solution_scheduler.step()
        if test_scheduler is not None:
            test_scheduler.step()

        metrics = _merge_metrics(
            epoch=epoch,
            min_terms=min_terms,
            max_terms=max_terms,
            x_test=x_test,
            q_test=q_test,
            model=model,
            solution_optimizer=solution_optimizer,
            test_optimizer=test_optimizer,
        )

        if not _all_finite(metrics):
            logging.warning("Encountered non-finite metrics at epoch %s", epoch)
            completed_normally = False
            summary["status"] = "nonfinite_loss"
            _record_state(
                metrics=metrics,
                csv_path=csv_path,
                model=model,
                model_dir=model_dir,
                lr_dir=lr_dir,
                solution_optimizer=solution_optimizer,
                test_optimizer=test_optimizer,
                solution_scheduler=solution_scheduler,
                test_scheduler=test_scheduler,
                history_every=history_every,
                log_every=log_every,
                checkpoint_every=checkpoint_every,
                is_final=True,
                history_terms=history_terms,
                summary=summary,
            )
            break

        _record_state(
            metrics=metrics,
            csv_path=csv_path,
            model=model,
            model_dir=model_dir,
            lr_dir=lr_dir,
            solution_optimizer=solution_optimizer,
            test_optimizer=test_optimizer,
            solution_scheduler=solution_scheduler,
            test_scheduler=test_scheduler,
            history_every=history_every,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            is_final=epoch == int(config.epochs),
            history_terms=history_terms,
            summary=summary,
        )

    if completed_normally:
        summary["status"] = "completed"
    summary["elapsed_seconds"] = time.time() - start_time
    return summary


def _evaluate_epoch(
    model,
    batch,
    x_test,
    q_test,
    solution_optimizer,
    test_optimizer,
    epoch,
):
    with torch.enable_grad():
        min_terms = model.compute_min_loss_terms(
            x_int=batch[0].detach().clone().requires_grad_(True),
            x_ic=batch[1].detach().clone().requires_grad_(True),
            x_bc=batch[2].detach().clone().requires_grad_(True),
        )
        max_terms = model.compute_max_loss_terms(
            x_int=batch[0].detach().clone().requires_grad_(True)
        )
    return _merge_metrics(
        epoch=epoch,
        min_terms=min_terms,
        max_terms=max_terms,
        x_test=x_test,
        q_test=q_test,
        model=model,
        solution_optimizer=solution_optimizer,
        test_optimizer=test_optimizer,
    )


def _merge_metrics(
    epoch,
    min_terms,
    max_terms,
    x_test,
    q_test,
    model,
    solution_optimizer,
    test_optimizer,
):
    mae, l2re = _evaluate_test_metrics(model, x_test, q_test)
    return {
        "epoch": epoch,
        "total": min_terms["total"].detach(),
        "data_loss": min_terms["data_loss"].detach(),
        "res_loss": min_terms["res_loss"].detach(),
        "res_raw": min_terms["res_raw"].detach(),
        "u_ic": min_terms["u_ic"].detach(),
        "u_bc": min_terms["u_bc"].detach(),
        "test_norm": min_terms["test_norm"].detach(),
        "sol_reg": min_terms["sol_reg"].detach(),
        "test_reg": min_terms["test_reg"].detach(),
        "max_loss": max_terms["max_loss"].detach(),
        "mae": mae.detach(),
        "l2re": l2re.detach(),
        "solution_lr": solution_optimizer.param_groups[0]["lr"],
        "test_lr": test_optimizer.param_groups[0]["lr"],
    }


def _build_scheduler(optimizer, decay, total_epochs, eta_min):
    decay_name = str(decay).strip().lower()
    if decay_name in {"none", "constant"}:
        return None
    if decay_name in {"cosine", "cosineannealing", "cosine_annealing"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(total_epochs)),
            eta_min=float(eta_min),
        )
    raise ValueError("Unsupported decay {}".format(decay))


def _prepare_batch(batch, device):
    return tuple(
        sample.detach().clone().to(device=device, dtype=torch.float32) for sample in batch
    )


def _write_history_header(csv_path: Path, history_terms: list[str]):
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "total", *history_terms, "MAE", "L2RE", "solution_lr", "test_lr"]
        )


def _record_state(
    metrics,
    csv_path,
    model,
    model_dir,
    lr_dir,
    solution_optimizer,
    test_optimizer,
    solution_scheduler,
    test_scheduler,
    history_every,
    log_every,
    checkpoint_every,
    is_final,
    history_terms,
    summary,
):
    epoch = int(metrics["epoch"])
    row = [epoch, _to_float(metrics["total"])]
    for name in history_terms:
        row.append(_to_float(metrics[name]))
    row.extend(
        [
            _to_float(metrics["mae"]),
            _to_float(metrics["l2re"]),
            float(metrics["solution_lr"]),
            float(metrics["test_lr"]),
        ]
    )

    if epoch % history_every == 0 or is_final:
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    if epoch % log_every == 0 or is_final:
        logging.info(
            "epoch : %s | total : %s | data_loss : %s | res_loss : %s | "
            "res_raw : %s | max_loss : %s | MAE : %s | L2RE : %s | "
            "sol_lr : %s | test_lr : %s",
            epoch,
            _to_float(metrics["total"]),
            _to_float(metrics["data_loss"]),
            _to_float(metrics["res_loss"]),
            _to_float(metrics["res_raw"]),
            _to_float(metrics["max_loss"]),
            _to_float(metrics["mae"]),
            _to_float(metrics["l2re"]),
            float(metrics["solution_lr"]),
            float(metrics["test_lr"]),
        )

    if epoch % checkpoint_every == 0 or is_final:
        checkpoint_name = "model_{:02d}".format(epoch)
        torch.save(model.state_dict(), model_dir / checkpoint_name)
        torch.save(
            {
                "solution_optimizer": solution_optimizer.state_dict(),
                "test_optimizer": test_optimizer.state_dict(),
                "solution_scheduler": (
                    None
                    if solution_scheduler is None
                    else solution_scheduler.state_dict()
                ),
                "test_scheduler": (
                    None if test_scheduler is None else test_scheduler.state_dict()
                ),
            },
            lr_dir / checkpoint_name,
        )

    mae_value = _to_float(metrics["mae"])
    l2re_value = _to_float(metrics["l2re"])
    if mae_value < summary["best_mae"]:
        summary["best_mae"] = mae_value
        summary["best_mae_epoch"] = epoch
    if l2re_value < summary["best_l2re"]:
        summary["best_l2re"] = l2re_value
        summary["best_l2re_epoch"] = epoch
    summary["final_epoch"] = epoch
    summary["final_mae"] = mae_value
    summary["final_l2re"] = l2re_value
    summary["final_total_loss"] = _to_float(metrics["total"])


def _evaluate_test_metrics(model, x_test, q_test):
    with torch.no_grad():
        q_pred = model(x_test)
    mae = torch.nn.L1Loss()(q_test, q_pred)
    error_norm = torch.linalg.vector_norm((q_pred - q_test).reshape(-1), ord=2)
    reference_norm = torch.linalg.vector_norm(q_test.reshape(-1), ord=2)
    l2re = error_norm / reference_norm
    return mae, l2re


def _all_finite(metrics: dict):
    for value in metrics.values():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            return False
        if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
            return False
    return True


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return float(value)
