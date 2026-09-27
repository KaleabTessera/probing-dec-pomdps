"""Learning-rate warmup and cosine decay used for Overcooked v2."""

import optax


def create_learning_rate_fn(config):
    base_learning_rate = config["LR"]

    lr_warmup = config["LR_WARMUP"]
    update_steps = config["NUM_UPDATES"]
    warmup_steps = int(lr_warmup * update_steps)

    steps_per_epoch = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]

    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=warmup_steps * steps_per_epoch,
    )
    cosine_epochs = max(update_steps - warmup_steps, 1)

    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps * steps_per_epoch],
    )
    return schedule_fn
