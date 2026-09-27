"""MAPPO with independent feed-forward policies.

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


class ActorFF(nn.Module):
    """
    Feed-forward actor with separate parameters for each agent.
    """

    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, x, return_intermediates=False):
        act_str = self.config.get("ACTIVATION", "relu")
        if act_str == "relu":
            activation = nn.relu
        elif act_str == "tanh":
            activation = nn.tanh
        obs, avail_actions = x

        # Initial embedding
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        # Hidden layers
        hidden = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        hidden = activation(hidden)

        # Action logits
        pre_softmax_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(hidden)

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

        if return_intermediates:
            return pi, pre_softmax_logits, hidden
        else:
            return pi


class CriticFF(nn.Module):
    """
    Feed-forward critic with separate parameters for each agent.
    """

    config: dict

    @nn.compact
    def __call__(self, x, return_intermediates=False):
        act_str = self.config.get("ACTIVATION", "relu")
        if act_str == "relu":
            activation = nn.relu
        elif act_str == "tanh":
            activation = nn.tanh

        world_state = x

        # Initial embedding
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        embedding = activation(embedding)

        # Hidden layers
        hidden = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        hidden = activation(hidden)

        # Value output
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            hidden
        )

        if return_intermediates:
            return jnp.squeeze(value, axis=-1), hidden
        else:
            return jnp.squeeze(value, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def run_eval(rng, train_states, test_env, config, collect_data=False):
    """Evaluate greedy policies and track episode resets in collected trajectories."""
    return evaluate(
        rng,
        train_states,
        test_env,
        config,
        recurrent=False,
        independent=False,
        heterogeneous=False,
        collect_data=collect_data,
    )


def make_train(config):
    env = make_env(config, use_log_wrapper=True, use_state_wrapper=True)
    # Create test environment
    test_env = make_env(config, use_log_wrapper=True, use_state_wrapper=True)
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

    # Set default test parameters if not provided
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
        rng[0]
        num_agents = len(env.agents)

        # Stack agent data function
        def stack_agent_data(data_dict, size):
            """Stack agent data, handling world_state separately."""
            if "world_state" in data_dict:
                # Keep global critic inputs separate from local actor inputs.
                agent_obs = jnp.stack([data_dict[agent] for agent in env.agents])
                world_state = data_dict["world_state"]
                return {
                    "agent_obs": agent_obs.reshape(num_agents, size, -1),
                    "world_state": world_state,
                }
            else:
                # Stack rewards and actions along the agent axis.
                return jnp.stack([data_dict[agent] for agent in env.agents])

        # Initialize separate networks for each agent
        rng, *rng_inits = jax.random.split(rng, num_agents * 2 + 1)
        actor_init_rngs = jnp.array(rng_inits[:num_agents])
        critic_init_rngs = jnp.array(rng_inits[num_agents:])

        # Network definitions
        actor_network = ActorFF(action_dim(env, env.agents[0]), config=config)
        critic_network = CriticFF(config=config)

        def init_single_actor(rng_key):
            try:
                init_x = (
                    jnp.zeros(
                        (
                            config["NUM_ENVS"],
                            env.observation_space(env.agents[0]).shape[0],
                        )
                    ),
                    jnp.zeros((config["NUM_ENVS"], action_dim(env, env.agents[0]))),
                )
            except (TypeError, AttributeError):  # overcooked v2
                init_x = (
                    jnp.zeros(
                        (config["NUM_ENVS"], np.prod(env.observation_space().shape))
                    ),
                    jnp.zeros((config["NUM_ENVS"], action_dim(env, env.agents[0]))),
                )
            return actor_network.init(rng_key, init_x)

        def init_single_critic(rng_key):
            init_x = jnp.zeros((config["NUM_ENVS"], env.world_state_size()))
            return critic_network.init(rng_key, init_x)

        # Initialize parameters for all agents
        all_actor_params = jax.vmap(init_single_actor)(actor_init_rngs)
        all_critic_params = jax.vmap(init_single_critic)(critic_init_rngs)

        # Create optimizers
        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(
                    learning_rate=linear_schedule, eps=config.get("ADAM_EPS", 1e-5)
                ),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(
                    learning_rate=linear_schedule, eps=config.get("ADAM_EPS", 1e-5)
                ),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=config.get("ADAM_EPS", 1e-5)),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=config.get("ADAM_EPS", 1e-5)),
            )

        # Create separate train states for each agent (MAPPO - no parameter sharing)
        def create_single_actor_train_state(params):
            return TrainState.create(
                apply_fn=actor_network.apply,
                params=params,
                tx=actor_tx,
            )

        def create_single_critic_train_state(params):
            return TrainState.create(
                apply_fn=critic_network.apply,
                params=params,
                tx=critic_tx,
            )

        # Create independent train states for each agent
        actor_train_states = jax.vmap(create_single_actor_train_state)(all_actor_params)
        critic_train_states = jax.vmap(create_single_critic_train_state)(
            all_critic_params
        )

        # Initialize environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        obs_data = stack_agent_data(obsv, size=config["NUM_ENVS"])

        # Main training loop
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, step_idx):
                train_states, env_state, obs_data, rng = runner_state

                # Vectorized action selection
                rng, *_rngs = jax.random.split(rng, num_agents + 1)
                action_rngs = jnp.array(_rngs)

                # Get available actions if environment supports it
                try:
                    all_avail_actions = jax.vmap(env.get_avail_actions)(
                        env_state.env_state
                    )
                    all_avail_actions = jax.lax.stop_gradient(all_avail_actions)
                    avail_batch = stack_agent_data(
                        all_avail_actions, size=config["NUM_ENVS"]
                    )
                except (AttributeError, NotImplementedError):
                    # Fallback: all actions available
                    avail_batch = jnp.ones(
                        (
                            num_agents,
                            config["NUM_ENVS"],
                            action_dim(env, env.agents[0]),
                        )
                    )

                # Extract agent observations for actors
                obs_batch = obs_data["agent_obs"]

                def single_agent_forward(params, obs, avail, rng_key):
                    ac_in = (obs, avail)
                    pi = actor_network.apply(params, ac_in)
                    action = pi.sample(seed=rng_key)
                    log_prob = pi.log_prob(action)
                    return action, log_prob

                # Vectorize across all agents for actors
                vmapped_forward = jax.vmap(single_agent_forward, in_axes=(0, 0, 0, 0))
                actions_batch, log_probs_batch = vmapped_forward(
                    train_states[0].params, obs_batch, avail_batch, action_rngs
                )

                # Get values from centralized critics
                world_state = obs_data["world_state"]
                world_state = world_state.swapaxes(
                    0, 1
                )  # (num_agents, num_envs, world_state_size)

                def single_critic_forward(params, world_state):
                    value = critic_network.apply(params, world_state)
                    return value

                vmapped_critic_forward = jax.vmap(single_critic_forward, in_axes=(0, 0))
                values_batch = vmapped_critic_forward(
                    train_states[1].params, world_state
                )

                # Convert to dict format for environment
                actions = {
                    agent: actions_batch[i] for i, agent in enumerate(env.agents)
                }

                # Environment step
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, actions)

                # Apply reward shaping if configured
                if rew_shaping_anneal is not None and "shaped_reward" in info:
                    current_timestep = (
                        update_steps * config["NUM_STEPS"] * config["NUM_ENVS"]
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

                # Stack new data
                new_obs_data = stack_agent_data(obsv, size=config["NUM_ENVS"])
                done_batch = stack_agent_data(done, size=config["NUM_ENVS"])
                rewards_batch = stack_agent_data(reward, size=config["NUM_ENVS"])

                # infos - convert - {"key": [num_envs, num_agents]} to {"key": [num_agents, num_envs]}
                info = jax.tree.map(lambda x: x.T, info)

                # Create vectorized transition
                transition = Transition(
                    global_done=jnp.tile(done["__all__"], (num_agents, 1)),
                    done=done_batch,
                    action=actions_batch,
                    value=values_batch,
                    reward=rewards_batch,
                    log_prob=log_probs_batch,
                    obs=obs_batch,
                    world_state=world_state,
                    info=info,
                    avail_actions=avail_batch,
                )

                runner_state = (train_states, env_state, new_obs_data, rng)
                return runner_state, transition

            # Collect trajectory
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, jnp.arange(config["NUM_STEPS"])
            )

            train_states, env_state, obs_data, rng = runner_state

            # Calculate last values for bootstrapping
            last_world_state = obs_data["world_state"]
            last_world_state = last_world_state.swapaxes(0, 1)

            def get_last_value(params, world_state):
                last_val = critic_network.apply(params, world_state)
                return last_val

            vmapped_last_value = jax.vmap(get_last_value, in_axes=(0, 0))
            last_vals = vmapped_last_value(train_states[1].params, last_world_state)

            # Vectorized GAE calculation for all agents
            def calculate_gae_single_agent(traj_agent, last_val_agent):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val_agent), last_val_agent),
                    traj_agent,
                    reverse=True,
                )
                return advantages, advantages + traj_agent.value

            # Calculate GAE for all agents
            vmapped_gae = jax.vmap(
                calculate_gae_single_agent, in_axes=(1, 0), out_axes=(1, 1)
            )
            advantages, targets = vmapped_gae(traj_batch, last_vals)

            # PPO Update for all agents with minibatching
            def _single_agent_update(
                agent_actor_state,
                agent_critic_state,
                agent_traj,
                agent_advantages,
                agent_targets,
                rng,
            ):
                """Update one agent; vmap applies this across agents."""

                def _update_epoch(update_state, unused):
                    def _update_minibatch(train_states, batch_info):
                        actor_state, critic_state = train_states
                        traj_batch, advantages, targets = batch_info

                        def _actor_loss_fn(params, traj_batch, gae):
                            # Re-evaluate the actor
                            pi = actor_network.apply(
                                params, (traj_batch.obs, traj_batch.avail_actions)
                            )
                            log_prob = pi.log_prob(traj_batch.action)

                            # Calculate actor loss
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
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()
                            entropy = pi.entropy().mean()

                            total_loss = loss_actor - config["ENT_COEF"] * entropy
                            return total_loss, (loss_actor, entropy, ratio)

                        def _critic_loss_fn(params, traj_batch, targets):
                            # Re-evaluate the critic
                            value = critic_network.apply(params, traj_batch.world_state)

                            # Calculate value loss
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

                            total_loss = config["VF_COEF"] * value_loss
                            return total_loss, value_loss

                        # Compute gradients
                        actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                        critic_grad_fn = jax.value_and_grad(
                            _critic_loss_fn, has_aux=True
                        )

                        (actor_loss, actor_aux), actor_grads = actor_grad_fn(
                            actor_state.params, traj_batch, advantages
                        )
                        (critic_loss, critic_aux), critic_grads = critic_grad_fn(
                            critic_state.params, traj_batch, targets
                        )

                        # Update states
                        actor_state = actor_state.apply_gradients(grads=actor_grads)
                        critic_state = critic_state.apply_gradients(grads=critic_grads)

                        loss_info = {
                            "total_loss": actor_loss + critic_loss,
                            "actor_loss": actor_aux[0],
                            "critic_loss": critic_aux,
                            "entropy": actor_aux[1],
                            "ratio": actor_aux[2],
                        }

                        return (actor_state, critic_state), loss_info

                    (
                        agent_actor_state,
                        agent_critic_state,
                        agent_traj,
                        agent_advantages,
                        agent_targets,
                        rng,
                    ) = update_state
                    rng, _rng = jax.random.split(rng)

                    # Flatten across timesteps and environments for this agent
                    batch_size = (
                        agent_traj.obs.shape[0] * agent_traj.obs.shape[1]
                    )  # timesteps * envs
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

                    (agent_actor_state, agent_critic_state), loss_info = jax.lax.scan(
                        _update_minibatch,
                        (agent_actor_state, agent_critic_state),
                        shuffled_batch,
                    )
                    update_state = (
                        agent_actor_state,
                        agent_critic_state,
                        agent_traj,
                        agent_advantages,
                        agent_targets,
                        rng,
                    )
                    return update_state, loss_info

                update_state = (
                    agent_actor_state,
                    agent_critic_state,
                    agent_traj,
                    agent_advantages,
                    agent_targets,
                    rng,
                )
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )

                return update_state[0], update_state[1], loss_info

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
            vmapped_update = jax.vmap(_single_agent_update, in_axes=(0, 0, 0, 0, 0, 0))
            actor_train_states, critic_train_states, loss_info = vmapped_update(
                train_states[0],
                train_states[1],
                agent_traj,
                agent_advantages,
                agent_targets,
                agent_rngs,
            )

            train_states = (actor_train_states, critic_train_states)

            # Process loss info
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric = traj_batch.info

            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric = {**metric}
            metric["loss"] = loss_info
            metric["update_steps"] = update_steps
            metric["env_step"] = update_steps * config["NUM_STEPS"] * config["NUM_ENVS"]

            # Periodic evaluation (no data collection)
            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                eval_interval = max(
                    1, int(config["NUM_UPDATES"] * config.get("TEST_INTERVAL", 0.1))
                )
                should_evaluate = (update_steps == config["NUM_UPDATES"] - 1) | (
                    update_steps % eval_interval == 0
                )
                test_metrics = jax.lax.cond(
                    should_evaluate,
                    lambda _: run_eval(
                        _rng,
                        train_states[0],
                        test_env,
                        config,
                        collect_data=False,
                    ),
                    lambda _: jax.tree.map(
                        lambda x: (
                            jnp.full_like(x, jnp.nan) if hasattr(x, "shape") else None
                        ),
                        initial_test_metrics,
                    ),
                    operand=None,
                )
                metric.update({"test_" + k: v for k, v in test_metrics.items()})

            update_steps = update_steps + 1
            runner_state = (train_states, env_state, obs_data, rng)
            return (runner_state, update_steps), metric

        # Initial evaluation (placeholder, not passed in runner_state)
        rng, _rng_eval = jax.random.split(rng)
        initial_test_metrics = run_eval(
            _rng_eval,
            actor_train_states,
            test_env,
            config,
            collect_data=False,
        )

        # Initial runner state (no test_state)
        runner_state = (
            (actor_train_states, critic_train_states),
            env_state,
            obs_data,
            _rng,
        )

        # Run training
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train
