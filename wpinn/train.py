import csv
import time
from pathlib import Path

import torch
from absl import logging
from ml_collections import ConfigDict

_HISTORY_TERMS = [
    "data_loss",
    "res_loss",
    "res_raw",
    "res_raw_after_max",
    "u_ic",
    "u_bc",
    "test_norm",
    "sol_reg",
    "test_reg",
    "max_loss",
]

_MIN_TERM_NAMES = (
    "total",
    "data_loss",
    "res_loss",
    "res_raw",
    "u_ic",
    "u_bc",
    "test_norm",
    "sol_reg",
    "test_reg",
)

_MAX_TERM_NAMES = (
    "total",
    "max_loss",
    "res_loss",
    "res_raw",
    "test_norm",
)


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
    start_epoch: int = 0,
    optimizer_state: dict | None = None,
):
    total_epochs = int(config.epochs)
    history_every = int(getattr(config, "history_every", 1000))
    log_every = int(getattr(config, "log_every", 1000))
    eval_every = int(getattr(config, "eval_every", max(history_every, log_every)))
    maximize_steps = int(getattr(config, "maximize_steps", 1))
    minimize_steps = int(getattr(config, "minimize_steps", 1))
    test_reset_every = int(getattr(config, "test_reset_every", 0))
    test_reset_frequency = float(getattr(config, "test_reset_frequency", 0.0))
    reset_test_optimizer_state = bool(
        getattr(config, "reset_test_optimizer_state", True)
    )
    if test_reset_every <= 0 and test_reset_frequency > 0.0:
        test_reset_every = max(1, int(round(test_reset_frequency * total_epochs)))

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
        total_epochs=total_epochs,
        eta_min=float(getattr(config, "solution_cosine_eta_min", 0.0)),
        t_max=getattr(config, "solution_cosine_t_max", None),
    )
    test_scheduler = _build_scheduler(
        test_optimizer,
        getattr(config, "test_decay", "CosineAnnealing"),
        total_epochs=total_epochs,
        eta_min=float(getattr(config, "test_cosine_eta_min", 0.0)),
        t_max=getattr(config, "test_cosine_t_max", None),
    )

    if optimizer_state:
        solution_state = optimizer_state.get("solution_optimizer")
        test_state = optimizer_state.get("test_optimizer")
        solution_scheduler_state = optimizer_state.get("solution_scheduler")
        test_scheduler_state = optimizer_state.get("test_scheduler")
        if solution_state is not None:
            solution_optimizer.load_state_dict(solution_state)
        if test_state is not None:
            test_optimizer.load_state_dict(test_state)
        if solution_scheduler is not None and solution_scheduler_state is not None:
            solution_scheduler.load_state_dict(solution_scheduler_state)
        if test_scheduler is not None and test_scheduler_state is not None:
            test_scheduler.load_state_dict(test_scheduler_state)

        resume_solution_lr = getattr(config, "resume_solution_lr", None)
        resume_test_lr = getattr(config, "resume_test_lr", None)
        if resume_solution_lr not in (None, 0):
            for group in solution_optimizer.param_groups:
                group["lr"] = float(resume_solution_lr)
        if resume_test_lr not in (None, 0):
            for group in test_optimizer.param_groups:
                group["lr"] = float(resume_test_lr)

    compile_requested = _compile_requested(config)
    minimize_step, maximize_step, compiled_training, compile_disable_reason = (
        _prepare_train_steps(
            model=model,
            config=config,
            datagenerator=datagenerator,
            device=device,
        )
    )

    logging.info("Created the csv file")
    logging.info("----------Start adversarial training-----------")
    if compiled_training:
        logging.info(
            "Compiled WPINN loss and residual steps with torch.compile "
            "(mode=%s, cudagraphs=%s)",
            getattr(config, "compile_mode", "reduce-overhead"),
            bool(getattr(config, "compile_cudagraphs", False)),
        )
    elif compile_requested and compile_disable_reason:
        logging.warning(
            "torch.compile was requested for WPINN training steps, "
            "but the current runtime could not compile this higher-order "
            "gradient path. Reason: %s",
            compile_disable_reason,
        )
    start_time = time.time()

    _write_history_header(csv_path, _HISTORY_TERMS)
    summary = {
        "status": "running",
        "eval_every": eval_every,
        "compile_requested": compile_requested,
        "compiled_training": compiled_training,
        "compile_mode": str(getattr(config, "compile_mode", "reduce-overhead")),
        "compile_cudagraphs": bool(getattr(config, "compile_cudagraphs", False)),
    }
    if compile_disable_reason:
        summary["compile_disable_reason"] = compile_disable_reason

    completed_normally = True
    last_epoch = int(start_epoch)

    for epoch in range(int(start_epoch) + 1, total_epochs + 1):
        last_epoch = epoch

        if test_reset_every > 0 and epoch % test_reset_every == 0:
            model.reset_test_network()
            if reset_test_optimizer_state:
                test_optimizer.state.clear()

        batch = _prepare_batch(datagenerator.samples(), device)
        min_terms = None
        max_terms = None
        res_raw_after_max = None

        for _ in range(maximize_steps):
            test_optimizer.zero_grad(set_to_none=True)
            max_terms = _max_values_to_terms(maximize_step(batch[0]))
            current_res_raw = max_terms["res_raw"].detach()
            if res_raw_after_max is None:
                res_raw_after_max = current_res_raw
            else:
                res_raw_after_max = res_raw_after_max + current_res_raw
            max_terms["total"].backward()
            test_optimizer.step()

        if res_raw_after_max is None:
            res_raw_after_max = torch.tensor(0.0, device=device, dtype=torch.float32)

        for _ in range(minimize_steps):
            solution_optimizer.zero_grad(set_to_none=True)
            min_terms = _min_values_to_terms(minimize_step(*batch))
            min_terms["total"].backward()
            solution_optimizer.step()

        if solution_scheduler is not None:
            solution_scheduler.step()
        if test_scheduler is not None:
            test_scheduler.step()

        training_metrics = _merge_training_metrics(
            epoch=epoch,
            min_terms=min_terms,
            max_terms=max_terms,
            res_raw_after_max=res_raw_after_max,
            solution_optimizer=solution_optimizer,
            test_optimizer=test_optimizer,
        )
        _update_summary_training(summary, training_metrics)

        if not _all_finite(training_metrics):
            logging.warning("Encountered non-finite metrics at epoch %s", epoch)
            completed_normally = False
            summary["status"] = "nonfinite_loss"
            if _should_run_interval(epoch, eval_every, is_final=True):
                eval_metrics = _evaluate_test_metrics(model, x_test, q_test)
                _update_summary_evaluation(summary, epoch, eval_metrics)
                _record_state(
                    training_metrics=training_metrics,
                    eval_metrics=eval_metrics,
                    csv_path=csv_path,
                    history_every=history_every,
                    log_every=log_every,
                    is_final=True,
                )
            break

        is_final = epoch == total_epochs
        if _should_run_interval(epoch, eval_every, is_final=is_final):
            eval_metrics = _evaluate_test_metrics(model, x_test, q_test)
            _update_summary_evaluation(summary, epoch, eval_metrics)
            _record_state(
                training_metrics=training_metrics,
                eval_metrics=eval_metrics,
                csv_path=csv_path,
                history_every=history_every,
                log_every=log_every,
                is_final=is_final,
            )

    if completed_normally:
        summary["status"] = "completed"

    if last_epoch > int(start_epoch):
        _save_checkpoint(
            epoch=last_epoch,
            model=model,
            model_dir=model_dir,
            lr_dir=lr_dir,
            solution_optimizer=solution_optimizer,
            test_optimizer=test_optimizer,
            solution_scheduler=solution_scheduler,
            test_scheduler=test_scheduler,
        )

    summary["elapsed_seconds"] = time.time() - start_time
    return summary


def _prepare_train_steps(model, config, datagenerator, device):
    _set_model_gradient_mode(model, "autograd")
    eager_minimize_step = _build_minimize_step(model, config, compile_step=False)
    eager_maximize_step = _build_maximize_step(model, config, compile_step=False)

    if not _compile_requested(config):
        return eager_minimize_step, eager_maximize_step, False, None

    _set_model_gradient_mode(model, "torch_func")
    compiled_minimize_step = _build_minimize_step(model, config, compile_step=True)
    compiled_maximize_step = _build_maximize_step(model, config, compile_step=True)
    try:
        batch = _prepare_batch(datagenerator.samples(), device)
        model.zero_grad(set_to_none=True)
        max_terms = _max_values_to_terms(compiled_maximize_step(batch[0]))
        max_terms["total"].backward()
        model.zero_grad(set_to_none=True)
        min_terms = _min_values_to_terms(compiled_minimize_step(*batch))
        min_terms["total"].backward()
        model.zero_grad(set_to_none=True)
        return compiled_minimize_step, compiled_maximize_step, True, None
    except RuntimeError as exc:
        model.zero_grad(set_to_none=True)
        if _is_compile_fallback_error(exc):
            _set_model_gradient_mode(model, "autograd")
            if not bool(getattr(config, "compile_fallback_to_eager", False)):
                raise RuntimeError(
                    "WPINN compile preflight failed and eager fallback is disabled. "
                    "Reason: {}".format(exc)
                ) from exc
            return eager_minimize_step, eager_maximize_step, False, str(exc)
        raise


def _build_minimize_step(
    model: torch.nn.Module, config: ConfigDict, compile_step: bool
):
    def minimize_step(x_int: torch.Tensor, x_ic: torch.Tensor, x_bc: torch.Tensor):
        terms = model.compute_min_loss_terms(x_int=x_int, x_ic=x_ic, x_bc=x_bc)
        return tuple(terms[name] for name in _MIN_TERM_NAMES)

    return _maybe_compile(minimize_step, config, enabled=compile_step)


def _build_maximize_step(
    model: torch.nn.Module, config: ConfigDict, compile_step: bool
):
    def maximize_step(x_int: torch.Tensor):
        terms = model.compute_max_loss_terms(x_int=x_int)
        return tuple(terms[name] for name in _MAX_TERM_NAMES)

    return _maybe_compile(maximize_step, config, enabled=compile_step)


def _maybe_compile(step_fn, config: ConfigDict, enabled: bool):
    if not enabled:
        return step_fn
    compile_kwargs = {
        "fullgraph": bool(getattr(config, "compile_fullgraph", False)),
        "dynamic": bool(getattr(config, "compile_dynamic", False)),
    }
    if bool(getattr(config, "compile_cudagraphs", False)):
        compile_kwargs["mode"] = str(getattr(config, "compile_mode", "reduce-overhead"))
    else:
        compile_kwargs["options"] = {"triton.cudagraphs": False}
    return torch.compile(step_fn, **compile_kwargs)


def _compile_requested(config: ConfigDict):
    return bool(getattr(config, "compile_training", True)) and hasattr(torch, "compile")


def _is_compile_fallback_error(exc: RuntimeError):
    message = str(exc).lower()
    return (
        "double backward" in message
        or "higher order gradients" in message
        or "higher-order gradients" in message
        or "torch.compile" in message
        or "backendcompilerfailed" in message
        or "cudagraph" in message
    )


def _set_model_gradient_mode(model: torch.nn.Module, mode: str):
    setter = getattr(model, "set_gradient_mode", None)
    if setter is not None:
        setter(mode)


def _merge_training_metrics(
    epoch,
    min_terms,
    max_terms,
    res_raw_after_max,
    solution_optimizer,
    test_optimizer,
):
    return {
        "epoch": epoch,
        "total": min_terms["total"].detach(),
        "data_loss": min_terms["data_loss"].detach(),
        "res_loss": min_terms["res_loss"].detach(),
        "res_raw": min_terms["res_raw"].detach(),
        "res_raw_after_max": res_raw_after_max.detach(),
        "u_ic": min_terms["u_ic"].detach(),
        "u_bc": min_terms["u_bc"].detach(),
        "test_norm": min_terms["test_norm"].detach(),
        "sol_reg": min_terms["sol_reg"].detach(),
        "test_reg": min_terms["test_reg"].detach(),
        "max_loss": max_terms["max_loss"].detach(),
        "solution_lr": solution_optimizer.param_groups[0]["lr"],
        "test_lr": test_optimizer.param_groups[0]["lr"],
    }


def _update_summary_training(summary: dict, training_metrics: dict):
    summary["final_epoch"] = int(training_metrics["epoch"])
    summary["final_total_loss"] = _to_float(training_metrics["total"])
    summary["final_data_loss"] = _to_float(training_metrics["data_loss"])
    summary["final_res_loss"] = _to_float(training_metrics["res_loss"])
    summary["final_res_raw"] = _to_float(training_metrics["res_raw"])
    summary["final_res_raw_after_max"] = _to_float(
        training_metrics["res_raw_after_max"]
    )
    summary["final_u_ic"] = _to_float(training_metrics["u_ic"])
    summary["final_u_bc"] = _to_float(training_metrics["u_bc"])
    summary["final_test_norm"] = _to_float(training_metrics["test_norm"])
    summary["final_sol_reg"] = _to_float(training_metrics["sol_reg"])
    summary["final_test_reg"] = _to_float(training_metrics["test_reg"])
    summary["final_max_loss"] = _to_float(training_metrics["max_loss"])
    summary["final_solution_lr"] = float(training_metrics["solution_lr"])
    summary["final_test_lr"] = float(training_metrics["test_lr"])


def _update_summary_evaluation(summary: dict, epoch: int, eval_metrics: dict):
    summary["last_evaluated_epoch"] = int(epoch)
    summary["final_mae"] = _to_float(eval_metrics["mae"])
    summary["final_l1re"] = _to_float(eval_metrics["l1re"])
    summary["final_l2re"] = _to_float(eval_metrics["l2re"])


def _evaluate_test_metrics(model, x_test, q_test):
    with torch.no_grad():
        q_pred = model(x_test)
    mae = torch.nn.L1Loss()(q_test, q_pred)
    reference_l1 = torch.mean(torch.abs(q_test)).clamp_min(
        torch.finfo(q_test.dtype).eps
    )
    l1re = mae / reference_l1
    error_norm = torch.linalg.vector_norm((q_pred - q_test).reshape(-1), ord=2)
    reference_norm = torch.linalg.vector_norm(q_test.reshape(-1), ord=2)
    l2re = error_norm / reference_norm
    return {
        "mae": mae.detach(),
        "l1re": l1re.detach(),
        "l2re": l2re.detach(),
    }


def _record_state(
    training_metrics,
    eval_metrics,
    csv_path,
    history_every,
    log_every,
    is_final,
):
    epoch = int(training_metrics["epoch"])
    should_write_history = _should_run_interval(epoch, history_every, is_final=is_final)
    should_log = _should_run_interval(epoch, log_every, is_final=is_final)

    row = [epoch, _to_float(training_metrics["total"])]
    for name in _HISTORY_TERMS:
        row.append(_to_float(training_metrics[name]))
    row.extend(
        [
            _to_float(eval_metrics["mae"]),
            _to_float(eval_metrics["l1re"]),
            _to_float(eval_metrics["l2re"]),
            float(training_metrics["solution_lr"]),
            float(training_metrics["test_lr"]),
        ]
    )

    if should_write_history:
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    if should_log:
        logging.info(
            "epoch : %s | total : %s | data_loss : %s | res_loss : %s | "
            "res_raw : %s | max_loss : %s | MAE : %s | L1RE : %s | L2RE : %s | "
            "sol_lr : %s | test_lr : %s",
            epoch,
            _to_float(training_metrics["total"]),
            _to_float(training_metrics["data_loss"]),
            _to_float(training_metrics["res_loss"]),
            _to_float(training_metrics["res_raw"]),
            _to_float(training_metrics["max_loss"]),
            _to_float(eval_metrics["mae"]),
            _to_float(eval_metrics["l1re"]),
            _to_float(eval_metrics["l2re"]),
            float(training_metrics["solution_lr"]),
            float(training_metrics["test_lr"]),
        )


def _save_checkpoint(
    epoch,
    model,
    model_dir,
    lr_dir,
    solution_optimizer,
    test_optimizer,
    solution_scheduler,
    test_scheduler,
):
    checkpoint_name = "model_{:02d}".format(int(epoch))
    torch.save(model.state_dict(), model_dir / checkpoint_name)
    torch.save(
        {
            "solution_optimizer": solution_optimizer.state_dict(),
            "test_optimizer": test_optimizer.state_dict(),
            "solution_scheduler": (
                None if solution_scheduler is None else solution_scheduler.state_dict()
            ),
            "test_scheduler": (
                None if test_scheduler is None else test_scheduler.state_dict()
            ),
        },
        lr_dir / checkpoint_name,
    )


def _min_values_to_terms(loss_values):
    return {
        name: value for name, value in zip(_MIN_TERM_NAMES, loss_values, strict=True)
    }


def _max_values_to_terms(loss_values):
    return {
        name: value for name, value in zip(_MAX_TERM_NAMES, loss_values, strict=True)
    }


def _build_scheduler(optimizer, decay, total_epochs, eta_min, t_max=None):
    decay_name = str(decay).strip().lower()
    if decay_name in {"none", "constant"}:
        return None
    if decay_name in {"inversetime", "inverse_time", "lambda"}:
        total_epochs = max(1, int(total_epochs))
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0 / (1.0 + (float(epoch) / float(total_epochs))),
        )
    if decay_name in {"cosine", "cosineannealing", "cosine_annealing"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(total_epochs if t_max in (None, 0) else t_max)),
            eta_min=float(eta_min),
        )
    raise ValueError("Unsupported decay {}".format(decay))


def _prepare_batch(batch, device):
    return tuple(
        sample.detach().clone().to(device=device, dtype=torch.float32)
        for sample in batch
    )


def _write_history_header(csv_path: Path, history_terms: list[str]):
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "total",
                *history_terms,
                "MAE",
                "L1RE",
                "L2RE",
                "solution_lr",
                "test_lr",
            ]
        )


def _should_run_interval(epoch: int, interval: int, is_final: bool):
    if is_final:
        return True
    if interval <= 0:
        return False
    return epoch % interval == 0


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
