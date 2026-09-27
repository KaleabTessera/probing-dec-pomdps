"""IPPO with independent recurrent policies.

Use ``python -m baselines.run`` to train and export diagnostic trajectories.
"""

import functools
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


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Reset finished episodes and advance the GRU."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(rnn_state.shape[0], rnn_state.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, hidden, x, return_intermediates=False):

        act_str = self.config.get("ACTIVATION", "relu")
        if act_str == "relu":
            activation = nn.relu
        elif act_str == "tanh":
            activation = nn.tanh

        obs, dones, avail_actions = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = activation(actor_mean)

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

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        if return_intermediates:
            return (
                hidden,
                pi,
                jnp.squeeze(critic, axis=-1),
                pre_softmax_logits,
                embedding,
            )
        else:
            return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def run_eval(rng, train_states, test_env, config, collect_data=False):
    """Evaluate greedy policies and track episode resets in collected trajectories."""
    return evaluate(
        rng,
        train_states,
        test_env,
        config,
        recurrent=True,
        independent=True,
        heterogeneous=False,
        collect_data=collect_data,
    )


def make_train(config):
    env = make_env(config, use_log_wrapper=True)
    # Create test environment
    test_env = make_env(config, use_log_wrapper=True)
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
        num_agents = len(env.agents)

        # Initialize network for all agents (no parameter sharing)
        rng, *rng_inits = jax.random.split(rng, num_agents + 1)
        init_rngs = jnp.array(rng_inits)

        network = ActorCriticRNN(action_dim(env, env.agents[0]), config=config)

        def init_single_agent(rng_key):
            try:
                obs_shape = env.observation_space(env.agents[0]).shape
                obs_size = int(np.prod(obs_shape))  # Flatten the observation space
            except (AttributeError, NotImplementedError):
                obs_shape = env.observation_space().shape
                obs_size = int(np.prod(obs_shape))  # Flatten the observation space
            init_x = (
                jnp.zeros(
                    (
                        1,
                        config["NUM_ENVS"],
                        obs_size,
                    )
                ),
                jnp.zeros((1, config["NUM_ENVS"])),
                jnp.zeros((1, config["NUM_ENVS"], action_dim(env, env.agents[0]))),
            )
            init_hstate = ScannedRNN.initialize_carry(
                config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
            )
            return network.init(rng_key, init_hstate, init_x)

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

        # Create separate train states for each agent (IPPO - no parameter sharing)
        def create_single_train_state(params):
            return TrainState.create(
                apply_fn=network.apply,
                params=params,
                tx=tx,
            )

        # Create independent train states for each agent
        train_states = jax.vmap(create_single_train_state)(all_params)

        # Initialize environments
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # Stack observations - Shape: (num_agents, num_envs, ...)
        def stack_agent_data(data_dict, size=config["NUM_ENVS"]):
            stacked_data = jnp.stack([data_dict[agent] for agent in env.agents])
            return stacked_data.reshape(num_agents, size, -1)

        obs_batch = stack_agent_data(obsv)

        # Initialize hidden states - Shape: (num_agents, num_envs, hidden_dim)
        init_hstates = jnp.stack(
            [
                ScannedRNN.initialize_carry(
                    config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
                )
                for _ in range(num_agents)
            ]
        )

        # Training loop
        def _update_step(runner_state, update_step):
            # Replay each sequence from its rollout-start memory, including mid-episode batches.
            rollout_hidden = runner_state[4]

            def _env_step(runner_state, step_idx):
                train_states, env_state, obs_batch, done_batch, hstates_batch, rng = (
                    runner_state
                )

                # Vectorized action selection
                rng, *_rngs = jax.random.split(rng, num_agents + 1)
                action_rngs = jnp.array(_rngs)

                # Get available actions if environment supports it
                try:
                    all_avail_actions = jax.vmap(env.get_avail_actions)(
                        env_state.env_state
                    )
                    avail_batch = jax.lax.stop_gradient(
                        stack_agent_data(all_avail_actions)
                    )
                except (AttributeError, NotImplementedError):
                    # Create all-ones tensor (all actions available)
                    avail_batch = jnp.ones(
                        (
                            num_agents,
                            config["NUM_ENVS"],
                            action_dim(env, env.agents[0]),
                        )
                    )

                def single_agent_forward(
                    train_state, hstate, obs, done, avail, rng_key
                ):
                    ac_in = (
                        obs[np.newaxis, :],
                        done[np.newaxis, :],
                        avail[np.newaxis, :],
                    )
                    new_hstate, pi, value = network.apply(
                        train_state.params, hstate, ac_in
                    )
                    action = pi.sample(seed=rng_key)
                    log_prob = pi.log_prob(action)
                    return (
                        new_hstate,
                        action.squeeze(0),
                        log_prob.squeeze(0),
                        value.squeeze(0),
                    )

                # Vectorize across all agents
                vmapped_forward = jax.vmap(
                    single_agent_forward, in_axes=(0, 0, 0, 0, 0, 0)
                )
                new_hstates, actions_batch, log_probs_batch, values_batch = (
                    vmapped_forward(
                        train_states,
                        hstates_batch,
                        obs_batch,
                        done_batch,
                        avail_batch,
                        action_rngs,
                    )
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

                # Stack new data
                new_obs_batch = stack_agent_data(obsv)
                new_done_batch = stack_agent_data(done)
                rewards_batch = stack_agent_data(reward)

                # infos - convert - {"key": [num_envs, num_agents]} to {"key": [num_agents, num_envs]}
                info = jax.tree.map(lambda x: x.T, info)

                # Create vectorized transition
                transition = Transition(
                    global_done=jnp.tile(done["__all__"], (num_agents, 1)),
                    done=done_batch,
                    action=actions_batch,
                    value=values_batch,
                    reward=rewards_batch.squeeze(-1),
                    log_prob=log_probs_batch,
                    obs=obs_batch,
                    info=info,
                    avail_actions=avail_batch,
                )

                runner_state = (
                    train_states,
                    env_state,
                    new_obs_batch,
                    new_done_batch.squeeze(-1),
                    new_hstates,
                    rng,
                )
                return runner_state, transition

            # Collect trajectory
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, jnp.arange(config["NUM_STEPS"])
            )
            train_states, env_state, obs_batch, done_batch, hstates_batch, rng = (
                runner_state
            )

            # Calculate last values for all agents
            try:
                all_avail_final = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_final = jax.lax.stop_gradient(stack_agent_data(all_avail_final))
            except (AttributeError, NotImplementedError):
                avail_final = jnp.ones(
                    (num_agents, config["NUM_ENVS"], action_dim(env, env.agents[0]))
                )

            def get_last_value(train_state, hstate, obs, done, avail):
                # Flatten observations for the network
                obs_flat = obs
                ac_in = (
                    obs_flat[np.newaxis, :],
                    done[np.newaxis, :],
                    avail[np.newaxis, :],
                )
                _, _, last_val = network.apply(train_state.params, hstate, ac_in)
                return last_val.squeeze(0)

            vmapped_last_value = jax.vmap(get_last_value, in_axes=(0, 0, 0, 0, 0))
            last_vals = vmapped_last_value(
                train_states, hstates_batch, obs_batch, done_batch, avail_final
            )

            # Calculate advantages for all agents
            def _calculate_gae(traj_batch, last_val):
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
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_vals)

            # Update each agent independently.
            def _single_agent_update(
                agent_train_state,
                agent_traj,
                agent_advantages,
                agent_targets,
                rng,
                initial_hidden,
            ):
                """Update one agent; vmap applies this across agents."""

                def _update_epoch(update_state, unused):
                    def _update_minibatch(train_state, batch_info):
                        traj_batch, advantages, targets, init_hstate = batch_info

                        def _loss_fn(params, traj_batch, gae, targets, init_hstate):
                            # For RNN, preserve sequence structure: (timesteps, envs, ...)
                            obs_input = traj_batch.obs  # (timesteps, envs, obs_dim)
                            done_input = traj_batch.done  # (timesteps, envs)
                            avail_input = (
                                traj_batch.avail_actions
                            )  # (timesteps, envs, action_dim)

                            # Run network forward with proper sequence
                            _, pi, value = network.apply(
                                params,
                                init_hstate,
                                (obs_input, done_input, avail_input),
                            )

                            log_prob = pi.log_prob(traj_batch.action)

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

                            total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy, ratio)

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params,
                            traj_batch,
                            advantages,
                            targets,
                            init_hstate,
                        )
                        train_state = train_state.apply_gradients(grads=grads)

                        loss_info = {
                            "total_loss": total_loss[0],
                            "actor_loss": total_loss[1][1],
                            "critic_loss": total_loss[1][0],
                            "entropy": total_loss[1][2],
                            "ratio": total_loss[1][3],
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

                    # For RNN, preserve sequence structure and only permute across environments
                    _seq_len, batch_size = agent_traj.obs.shape[:2]  # timesteps, envs

                    # Permute environments only
                    permutation = jax.random.permutation(_rng, batch_size)

                    # Permute across the environment dimension (axis=1)
                    permuted_traj = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=1), agent_traj
                    )
                    permuted_advantages = jnp.take(
                        agent_advantages, permutation, axis=1
                    )
                    permuted_targets = jnp.take(agent_targets, permutation, axis=1)

                    # Split into minibatches across environments
                    mb_size = batch_size // config["NUM_MINIBATCHES"]

                    def split_minibatch(x, axis=1):
                        # Reshape to (seq_len, num_minibatches, mb_size, ...)
                        new_shape = (
                            x.shape[:axis]
                            + (config["NUM_MINIBATCHES"], mb_size)
                            + x.shape[axis + 1 :]
                        )
                        reshaped = jnp.reshape(x, new_shape)
                        # Swap to (num_minibatches, seq_len, mb_size, ...)
                        return jnp.swapaxes(reshaped, 0, axis)

                    mb_traj = jax.tree.map(split_minibatch, permuted_traj)
                    mb_advantages = split_minibatch(permuted_advantages)
                    mb_targets = split_minibatch(permuted_targets)

                    mb_hstates = jnp.take(initial_hidden, permutation, axis=0).reshape(
                        config["NUM_MINIBATCHES"], mb_size, config["GRU_HIDDEN_DIM"]
                    )

                    # Combine into minibatch format: (num_minibatches, ...)
                    minibatches = (mb_traj, mb_advantages, mb_targets, mb_hstates)

                    agent_train_state, loss_info = jax.lax.scan(
                        _update_minibatch, agent_train_state, minibatches
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
            vmapped_update = jax.vmap(_single_agent_update, in_axes=(0, 0, 0, 0, 0, 0))
            train_states, loss_info = vmapped_update(
                train_states,
                agent_traj,
                agent_advantages,
                agent_targets,
                agent_rngs,
                rollout_hidden,
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

            # Periodic evaluation
            if config.get("TEST_DURING_TRAINING", True):
                eval_interval = max(
                    1, int(config["NUM_UPDATES"] * config.get("TEST_INTERVAL", 0.1))
                )
                should_evaluate = (update_step == config["NUM_UPDATES"] - 1) | (
                    update_step % eval_interval == 0
                )

                def do_eval(_):
                    rng_eval, _ = jax.random.split(rng)
                    return run_eval(rng_eval, train_states, test_env, config)

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

            runner_state = (
                train_states,
                env_state,
                obs_batch,
                done_batch,
                hstates_batch,
                rng,
            )
            return runner_state, metric

        # Evaluate once to determine the metric shapes for lax.cond.
        rng, _rng_eval = jax.random.split(rng)
        initial_test_metrics = run_eval(_rng_eval, train_states, test_env, config)

        # Initial runner state
        runner_state = (
            train_states,
            env_state,
            obs_batch,
            jnp.zeros((num_agents, config["NUM_ENVS"]), dtype=bool),
            init_hstates,
            _rng,
        )

        # Run training
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
