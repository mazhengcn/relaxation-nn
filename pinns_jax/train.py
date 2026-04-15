import dataclasses

import jax
import jax.numpy as jnp
from absl import logging
from clu import metric_writers, periodic_actions
from etils import epath
from flax import nnx
from orbax.checkpoint import v1 as ocp

from .equations import BaseEquation, Burgers1D, Solution
from .input_pipeline import get_datasets
from .optimizers import get_optimizer, get_scheduler

training = ocp.training


@nnx.jit(static_argnames=["equation"])
def train_step(
    model: Solution,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    equation: BaseEquation,
    batch,
):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(equation.loss, has_aux=True)
    (total_loss, loss_dict), grads = grad_fn(model, batch)
    metrics.update(total_loss=total_loss, **loss_dict)  # In-place updates.
    optimizer.update(model, grads)  # In-place updates.


@nnx.jit
def eval_step(model: Solution, metrics: nnx.MultiMetric, data):
    """Evaluate for a single step."""
    y = model(*data["inputs"])
    labels = data["labels"]
    mse = jnp.mean((y - labels) ** 2)
    relative_mse = mse / jnp.mean(labels**2)
    metrics.update(mse=mse, relative_mse=relative_mse)  # In-place updates.


def init_training_states(config, init_key):
    # Create an abstract model
    model = Solution(config, rngs=nnx.Rngs(init_key))
    opt = get_optimizer(config)
    optimizer = nnx.Optimizer(model, opt, wrt=nnx.Param)
    return model, optimizer


def init_or_restore(
    ckptr: training.Checkpointer, ckpt_path: str, config, init_key
) -> tuple[Solution, nnx.Optimizer, int]:

    if ckpt_path or ckptr.latest:
        # If a checkpoint already exists, we restore it.
        model, optimizer = nnx.eval_shape(
            lambda: init_training_states(config, init_key=0)
        )
        abs_state = nnx.state({"params": model, "optimizer": optimizer})
        if ckptr.latest:
            loaded_state = ckptr.load_pytree(abstract_pytree=abs_state)
        else:
            loaded_state = ocp.load_pytree(path=ckpt_path, abstract_pytree=abs_state)
        # Update model and optimizer separately
        nnx.update(model, loaded_state["params"])  # ty: ignore
        nnx.update(optimizer, loaded_state["optimizer"])  # ty: ignore
        last_step = int(loaded_state["optimizer"]["step"][...])  # ty: ignore
    else:
        last_step = 0
        model, optimizer = init_training_states(config, init_key)

    return model, optimizer, last_step


def train_and_evaluate(config, workdir: str | epath.Path):
    """Runs a training and evaluation loop.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    workdir = epath.Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # Load Dataset
    # ---------------------------------------------------------------------------
    logging.info("Initializing dataset.")
    train_ds, eval_ds = get_datasets(config)
    train_iter = iter(train_ds)

    logging.info("Initializing equation.")
    equation = Burgers1D(nu=config.nu, weights=config.loss_weights)

    logging.info("Initializing model, optimizer, and step functions.")
    init_key = jax.random.key(config.seed)

    learning_rate_fn = get_scheduler(config)

    with ocp.training.Checkpointer(workdir) as ckptr:
        # Build Model and Optimizer
        # ---------------------------------------------------------------------------
        model, optimizer, start_step = init_or_restore(
            ckptr, config.restore_checkpoints, config, init_key
        )
        writer = metric_writers.create_default_writer(
            workdir, just_logging=jax.process_index() > 0
        )
        if start_step == 0:
            writer.write_hparams(dataclasses.asdict(config))

        # Main Train Loop
        # ---------------------------------------------------------------------------

        # We init the first set of dropout PRNG keys, but update it afterwards inside
        # the main pmap'd training update for performance.
        logging.info("Starting training loop.")
        hooks = []
        report_progress = periodic_actions.ReportProgress(
            num_train_steps=config.num_train_steps, writer=writer
        )
        if jax.process_index() == 0:
            hooks += [
                report_progress,
                periodic_actions.Profile(logdir=workdir, num_profile_steps=5),
            ]
        train_metrics = nnx.MultiMetric(
            total_loss=nnx.metrics.Average("total_loss"),
            res_loss=nnx.metrics.Average("res_loss"),
            bc_loss=nnx.metrics.Average("bc_loss"),
            ic_loss=nnx.metrics.Average("ic_loss"),
        )
        eval_metrics = nnx.MultiMetric(
            mse=nnx.metrics.Average("mse"),
            relative_mse=nnx.metrics.Average("relative_mse"),
        )
        with metric_writers.ensure_flushes(writer):
            for step in range(start_step, config.num_train_steps):
                is_last_step = step == config.num_train_steps - 1

                # Shard data to devices and do a training step.
                with jax.profiler.StepTraceAnnotation("train", step_num=step):
                    with report_progress.timed("data"):
                        batch = next(train_iter)

                    with report_progress.timed("train_step"):
                        train_step(
                            model=model,
                            optimizer=optimizer,
                            metrics=train_metrics,
                            equation=equation,
                            batch=batch,
                        )

                # Quick indication that training is happening.
                logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
                for h in hooks:
                    h(step)

                # Periodic metric handling.
                if (step > 0 and step % config.eval_every_steps == 0) or is_last_step:
                    with report_progress.timed("training_metrics"):
                        logging.info("Gathering training metrics.")
                        summary = {
                            "train_" + k: v for k, v in train_metrics.compute().items()
                        }
                        summary["learning_rate"] = learning_rate_fn(step)
                        writer.write_scalars(step, summary)
                        train_metrics.reset()

                    with report_progress.timed("eval"):
                        eval_step(model, eval_metrics, eval_ds)
                        writer.write_scalars(
                            step,
                            {"eval_" + k: v for k, v in eval_metrics.compute().items()},
                        )
                        eval_metrics.reset()

                # Save a checkpoint on one host after every checkpoint_freq steps.
                save_checkpoint = (
                    step % config.checkpoint_every_steps == 0 or is_last_step
                )
                if config.save_checkpoints and save_checkpoint:
                    logging.info("Saving checkpoint step %d.", step)
                    with report_progress.timed("checkpoint"):
                        ckptr.save_pytree_async(
                            nnx.state(
                                {
                                    "params": model,
                                    "optimizer": optimizer,
                                }
                            )
                        )
