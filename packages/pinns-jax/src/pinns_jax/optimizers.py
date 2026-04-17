import optax


def get_optimizer(config):
    lr_scheduler = get_scheduler(config)
    if config.optimizer == "adam":
        return optax.adam(lr_scheduler)
    elif config.optimizer == "sgd":
        return optax.sgd(lr_scheduler)
    elif config.optimizer == "lbfgs":
        return optax.lbfgs(lr_scheduler)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def get_scheduler(config) -> optax.Schedule:
    if config.scheduler == "exponential_decay":
        return optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.scheduler_transition_steps,
            decay_rate=config.scheduler_decay_rate,
            staircase=config.scheduler_staircase,
        )
    elif config.scheduler == "cosine_decay":
        return optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=config.scheduler_decay_steps,
        )
    elif config.scheduler == "constant":
        return optax.constant_schedule(config.learning_rate)
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")
