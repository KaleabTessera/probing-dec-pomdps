"""IPPO with independent feed-forward policies.

Use ``python -m baselines.run`` to train and export diagnostic trajectories.
"""

from typing import NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from baselines.env_utils import action_dim, make_env
from baselines.rollout import evaluate
from baselines.train_utils import create_learning_rate_fn


class ActorCritic(nn.Module):
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, x, avail_actions=None, return_intermediates=False):
        act_str = self.config.get("ACTIVATION", "relu")
        if act_str == "relu":
            activation = nn.relu
        elif act_str == "tanh":
            activation = nn.tanh

        # Actor network
        actor_mean = nn.Dense(
            self.config.get("FC_DIM_SIZE", 64),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.config.get("FC_DIM_SIZE", 64),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        # Pre-softmax logits (before action masking)
        pre_softmax_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        if self.config.get("CONTINUOUS_ACTIONS", False):
            # MaBrax uses an independent Gaussian with learned, state-independent
            # scale. PPO log probabilities sum over the action coordinates.
            log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(pre_softmax_logits, jnp.exp(log_std))
        else:
            # Apply action masking if available actions are provided
            if avail_actions is not None:
                unavail_actions = 1 - avail_actions
                action_logits = pre_softmax_logits - (unavail_actions * 1e10)
            else:
                action_logits = pre_softmax_logits

            pi = distrax.Categorical(logits=action_logits)

        # Critic network
        critic = nn.Dense(
            self.config.get("FC_DIM_SIZE", 64),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.config.get("FC_DIM_SIZE", 64),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        if return_intermediates:
            return pi, jnp.squeeze(critic, axis=-1), pre_softmax_logits
        else:
            return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def run_eval(rng, train_states, network, test_env, config, collect_data=False):
    """Evaluate greedy policies and track episode resets in collected trajectories."""
    return evaluate(
        rng,
        train_states,
        test_env,
        config,
        recurrent=False,
        independent=True,
        heterogeneous=False,
        collect_data=collect_data,
    )


def make_train(config):
    env = make_env(config, use_log_wrapper=True)
    test_env = make_env(
        config, use_log_wrapper=True
    )  # Used for in-training eval metrics

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    if "TEST_NUM_ENVS" not in config:
        config["TEST_NUM_ENVS"] = config["NUM_ENVS"]
    if "TEST_NUM_STEPS" not in config:
        config["TEST_NUM_STEPS"] = config["NUM_STEPS"]
    if "TEST_INTERVAL" not in config:
        config["TEST_INTERVAL"] = 0.1

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # Reward shaping annealing schedule (optional)
    rew_shaping_anneal = None
    if config.get("REW_SHAPING_HORIZON") is not None:
        rew_shaping_anneal = optax.linear_schedule(
            init_value=1.0,
            end_value=0.0,
            transition_steps=config["REW_SHAPING_HORIZON"],
        )

    def train(rng):
        num_agents = len(env.agents)
        # Initialize network for all agents (no parameter sharing)
        rng, *rng_inits = jax.random.split(rng, num_agents + 1)
        init_rngs = jnp.array(rng_inits)
        network = ActorCritic(action_dim(env, env.agents[0]), config=config)

        def init_single_agent(rng_key):
            try:
                init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
            except (TypeError, AttributeError):
                # overcooked
                init_x = jnp.zeros(env.observation_space().shape).flatten()
            return network.init(rng_key, init_x)

        # Vectorized parameter initialization - Shape: (num_agents, ...)
        all_params = jax.vmap(init_single_agent)(init_rngs)

        # Create optimizer
        if config["ANNEAL_LR"]:
            # warmup for overcooked v2
            if config.get("LR_WARMUP") is not None:
                lr_fn = create_learning_rate_fn(config)
            else:
                lr_fn = linear_schedule
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr_fn, eps=config.get("ADAM_EPS", 1e-5)),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=config.get("ADAM_EPS", 1e-5)),
            )

        def create_single_train_state(params):
            return TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        # Create independent train states for each agent
        train_states = jax.vmap(create_single_train_state)(all_params)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        def stack_agent_data(data_dict, size=config["NUM_ENVS"]):
            stacked_data = jnp.stack([data_dict[agent] for agent in env.agents])
            return stacked_data.reshape(num_agents, size, -1)

        obs_batch = stack_agent_data(obsv)

        def _update_step(runner_state, update_step):
            train_states, env_state, obs_batch, rng = runner_state

            # Environment rollout
            def _env_step(runner_state, step_idx):
                train_states, env_state, obs_batch, rng = runner_state
                try:
                    all_avail_actions = jax.vmap(env.get_avail_actions)(
                        env_state.env_state
                    )
                    avail_batch = stack_agent_data(
                        jax.lax.stop_gradient(all_avail_actions)
                    )
                except (AttributeError, NotImplementedError):
                    avail_batch = jnp.ones(
                        (
                            num_agents,
                            config["NUM_ENVS"],
                            action_dim(env, env.agents[0]),
                        )
                    )

                rng, *_rngs = jax.random.split(rng, num_agents + 1)
                action_rngs = jnp.array(_rngs)

                def single_agent_forward(
                    train_state, obs, avail_actions_agent, rng_key
                ):
                    pi, value = network.apply(
                        train_state.params, obs, avail_actions_agent
                    )
                    action = pi.sample(seed=rng_key)
                    log_prob = pi.log_prob(action)
                    return action, log_prob, value

                vmapped_forward = jax.vmap(single_agent_forward, in_axes=(0, 0, 0, 0))
                actions_batch, log_probs_batch, values_batch = vmapped_forward(
                    train_states, obs_batch, avail_batch, action_rngs
                )

                actions = {
                    agent: actions_batch[i] for i, agent in enumerate(env.agents)
                }
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, actions
                )

                # Apply reward shaping if configured
                if rew_shaping_anneal is not None and "shaped_reward" in info:
                    current_timestep = (
                        update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                        + step_idx * config["NUM_ENVS"]
                    )
                    shaping_factor = rew_shaping_anneal(current_timestep)
                    # Add shaped rewards to base rewards
                    reward = jax.tree.map(
                        lambda base_rew, shaped_rew: (
                            base_rew + shaped_rew * shaping_factor
                        ),
                        reward,
                        info["shaped_reward"],
                    )

                    # Remove shaped_reward from info since it's now incorporated into reward
                    info = {k: v for k, v in info.items() if k != "shaped_reward"}

                new_obs_batch = stack_agent_data(obsv)
                done_batch = stack_agent_data(done)
                rewards_batch = stack_agent_data(reward)
                info_batch = jax.tree.map(lambda x: x.T, info)

                # Create vectorized transition
                # need squeeze for overcooked
                transition = Transition(
                    done=done_batch.squeeze(-1),
                    action=actions_batch,
                    value=values_batch,
                    reward=rewards_batch.squeeze(-1),
                    log_prob=log_probs_batch,
                    obs=obs_batch,
                    info=info_batch,
                    avail_actions=avail_batch,
                )
                runner_state = (train_states, env_state, new_obs_batch, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, jnp.arange(config["NUM_STEPS"])
            )
            train_states, env_state, obs_batch, rng = runner_state

            # GAE Calculation
            def get_last_value(train_state, obs, avail_actions_agent):
                _, last_val = network.apply(
                    train_state.params, obs, avail_actions_agent
                )
                return last_val

            vmapped_last_value = jax.vmap(get_last_value, in_axes=(0, 0, 0))
            try:
                last_avail_actions = jax.vmap(env.get_avail_actions)(
                    env_state.env_state
                )
                last_avail_batch = stack_agent_data(last_avail_actions)
            except (AttributeError, NotImplementedError):
                last_avail_batch = jnp.ones(
                    (num_agents, config["NUM_ENVS"], action_dim(env, env.agents[0]))
                )
            last_vals = vmapped_last_value(train_states, obs_batch, last_avail_batch)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    delta = (
                        transition.reward
                        + config["GAMMA"] * next_value * (1 - transition.done)
                        - transition.value
                    )
                    gae = (
                        delta
                        + config["GAMMA"]
                        * config["GAE_LAMBDA"]
                        * (1 - transition.done)
                        * gae
                    )
                    return (gae, transition.value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_vals)

            # PPO Update
            def _single_agent_update(
                agent_train_state, agent_traj, agent_advantages, agent_targets, rng
            ):
                def _update_epoch(update_state, _):
                    def _update_minibatch(train_state, batch_info):
                        traj_batch, advantages, targets = batch_info

                        def _loss_fn(params, traj_batch, gae, targets):
                            pi, value = network.apply(
                                params, traj_batch.obs, traj_batch.avail_actions
                            )
                            log_prob = pi.log_prob(traj_batch.action)
                            value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(
                                value_pred_clipped - targets
                            )
                            value_loss = (
                                0.5
                                * jnp.maximum(value_losses, value_losses_clipped).mean()
                            )
                            ratio = jnp.exp(log_prob - traj_batch.log_prob)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                            entropy = pi.entropy().mean()
                            total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy)

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets
                        )
                        train_state = train_state.apply_gradients(grads=grads)

                        loss_info = {
                            "total_loss": total_loss[0],
                            "actor_loss": total_loss[1][1],
                            "critic_loss": total_loss[1][0],
                            "entropy": total_loss[1][2],
                        }

                        return train_state, loss_info

                    (
                        agent_train_state,
                        agent_traj,
                        agent_advantages,
                        agent_targets,
                        rng,
                    ) = update_state
                    rng, _rng = jax.random.split(rng)
                    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                    flat_traj = jax.tree.map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), agent_traj
                    )
                    flat_advantages = agent_advantages.reshape(batch_size)
                    flat_targets = agent_targets.reshape(batch_size)

                    permutation = jax.random.permutation(_rng, batch_size)
                    batch = (flat_traj, flat_advantages, flat_targets)
                    batch = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                    )
                    shuffled_batch = jax.tree.map(
                        lambda x: jnp.reshape(
                            x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                        ),
                        batch,
                    )

                    agent_train_state, loss_info = jax.lax.scan(
                        _update_minibatch, agent_train_state, shuffled_batch
                    )
                    update_state = (
                        agent_train_state,
                        agent_traj,
                        agent_advantages,
                        agent_targets,
                        rng,
                    )
                    return update_state, loss_info

                update_state = (
                    agent_train_state,
                    agent_traj,
                    agent_advantages,
                    agent_targets,
                    rng,
                )
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )
                return update_state[0], loss_info

            # Reshape data to (agent, timestep, env, ...)
            agent_traj = jax.tree.map(
                lambda x: x.transpose(1, 0, 2, *range(3, len(x.shape))), traj_batch
            )
            agent_advantages = advantages.transpose(1, 0, 2)
            agent_targets = targets.transpose(1, 0, 2)

            # Generate separate RNG for each agent
            rng, *agent_rngs = jax.random.split(rng, num_agents + 1)
            agent_rngs = jnp.array(agent_rngs)

            # Vmap across agents
            vmapped_update = jax.vmap(_single_agent_update, in_axes=(0, 0, 0, 0, 0))
            train_states, loss_info = vmapped_update(
                train_states, agent_traj, agent_advantages, agent_targets, agent_rngs
            )

            metric = traj_batch.info

            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric = {**metric}
            metric["loss"] = loss_info
            metric["update_steps"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

            # Add reward shaping metrics if configured
            if rew_shaping_anneal is not None and "shaped_reward" in traj_batch.info:
                current_timestep = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                shaping_factor = rew_shaping_anneal(current_timestep)
                shaped_reward_mean = jax.tree.map(
                    lambda x: x.mean(), traj_batch.info["shaped_reward"]
                )
                metric["shaped_reward"] = shaped_reward_mean
                metric["shaped_reward_annealed"] = jax.tree.map(
                    lambda x: x * shaping_factor, shaped_reward_mean
                )
                metric["reward_shaping_factor"] = shaping_factor

            # Evaluate without retaining trajectories.
            if config.get("TEST_DURING_TRAINING", True):
                eval_interval = max(
                    1, int(config["NUM_UPDATES"] * config.get("TEST_INTERVAL", 0.1))
                )
                should_evaluate = (update_step == config["NUM_UPDATES"] - 1) | (
                    update_step % eval_interval == 0
                )

                def do_eval(_):
                    rng_eval, _ = jax.random.split(rng)
                    return run_eval(
                        rng_eval,
                        train_states,
                        network,
                        test_env,
                        config,
                        collect_data=False,
                    )

                def skip_eval(_):
                    # Match the evaluation structure with NaN placeholders.
                    return jax.tree.map(
                        lambda x: (
                            jnp.full_like(x, jnp.nan) if hasattr(x, "shape") else None
                        ),
                        initial_test_metrics,
                    )

                test_metrics = jax.lax.cond(should_evaluate, do_eval, skip_eval, None)
                # Mark updates without evaluation using NaNs.
                metric.update({"test_" + k: v for k, v in test_metrics.items()})

            runner_state = (train_states, env_state, obs_batch, rng)
            return runner_state, metric

        # Evaluate once to determine the metric shapes for lax.cond.
        rng, _rng_eval = jax.random.split(rng)
        initial_test_metrics = run_eval(
            _rng_eval, train_states, network, test_env, config, collect_data=False
        )

        # Initial runner state
        runner_state = (train_states, env_state, obs_batch, rng)
        # Run training
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train
