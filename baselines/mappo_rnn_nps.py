"""MAPPO with independent recurrent policies.

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


class ScannedRNN(nn.Module):
    """
    Apply a GRU across a sequence, resetting at episode boundaries.
    """

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x

        # Handle both scalar and array resets to avoid indexing errors
        if resets.ndim == 0:
            # Scalar case - broadcast to match rnn_state batch dimension
            reset_mask = jnp.broadcast_to(resets, (rnn_state.shape[0],))[:, np.newaxis]
        else:
            # Array case - add newaxis for hidden dimension
            reset_mask = resets[:, np.newaxis]

        rnn_state = jnp.where(
            reset_mask,
            self.initialize_carry(rnn_state.shape[0], rnn_state.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    """
    Recurrent actor with separate parameters for each agent.
    """

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

        # Initial embedding
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        # RNN layer
        rnn_in = (embedding, dones)
        hidden, rnn_output = ScannedRNN()(hidden, rnn_in)

        # Actor layers
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(rnn_output)
        actor_mean = activation(actor_mean)

        # Action logits
        pre_softmax_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        if self.config.get("CONTINUOUS_ACTIONS", False):
            # MaBrax uses an independent Gaussian with learned, state-independent
            # scale. PPO log probabilities sum over the action coordinates.
            log_std = self.param("log_std", nn.initializers.zeros, (1, self.action_dim))
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
            return hidden, pi, pre_softmax_logits, rnn_output
        else:
            return hidden, pi


class CriticRNN(nn.Module):
    """
    Recurrent critic with separate parameters for each agent.
    """

    config: dict

    @nn.compact
    def __call__(self, hidden, x, return_intermediates=False):
        act_str = self.config.get("ACTIVATION", "relu")
        if act_str == "relu":
            activation = nn.relu
        elif act_str == "tanh":
            activation = nn.tanh

        world_state, dones = x

        # Initial embedding
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        embedding = activation(embedding)

        # RNN layer
        rnn_in = (embedding, dones)
        hidden, rnn_output = ScannedRNN()(hidden, rnn_in)

        # Critic layers
        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(rnn_output)
        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        if return_intermediates:
            return hidden, jnp.squeeze(critic, axis=-1), rnn_output
        else:
            return hidden, jnp.squeeze(critic, axis=-1)


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
        train_states[0],
        test_env,
        config,
        recurrent=True,
        independent=False,
        heterogeneous=False,
        collect_data=collect_data,
    )


def make_train(config):
    env = make_env(config, use_log_wrapper=True, use_state_wrapper=True)
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
        def stack_agent_data(data_dict, size=config["NUM_ENVS"]):
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
        actor_network = ActorRNN(action_dim(env, env.agents[0]), config=config)
        critic_network = CriticRNN(config=config)

        def init_single_actor(rng_key):
            try:
                init_x = (
                    jnp.zeros(
                        (
                            1,
                            config["NUM_ENVS"],
                            env.observation_space(env.agents[0]).shape[0],
                        )
                    ),
                    jnp.zeros((1, config["NUM_ENVS"])),
                    jnp.zeros((1, config["NUM_ENVS"], action_dim(env, env.agents[0]))),
                )
            except (TypeError, AttributeError):  # overcooked
                init_x = (
                    jnp.zeros(
                        (
                            1,
                            config["NUM_ENVS"],
                            np.prod(env.observation_space().shape),
                        )
                    ),
                    jnp.zeros((1, config["NUM_ENVS"])),
                    jnp.zeros((1, config["NUM_ENVS"], action_dim(env, env.agents[0]))),
                )

            init_hstate = ScannedRNN.initialize_carry(
                config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
            )
            return actor_network.init(rng_key, init_hstate, init_x)

        def init_single_critic(rng_key):
            init_x = (
                jnp.zeros((1, config["NUM_ENVS"], env.world_state_size())),
                jnp.zeros((1, config["NUM_ENVS"])),
            )
            init_hstate = ScannedRNN.initialize_carry(
                config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
            )
            return critic_network.init(rng_key, init_hstate, init_x)

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

        # Create separate train states for each agent.
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

        # Create independent train states for each agent.
        actor_train_states = jax.vmap(create_single_actor_train_state)(all_actor_params)
        critic_train_states = jax.vmap(create_single_critic_train_state)(
            all_critic_params
        )
        train_states = (actor_train_states, critic_train_states)

        # Initialize environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        obs_data = stack_agent_data(
            obsv
        )  # Returns {"agent_obs": ..., "world_state": ...}.

        # Initialize hidden states for all agents
        init_actor_hstates = jnp.stack(
            [
                ScannedRNN.initialize_carry(
                    config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
                )
                for _ in range(num_agents)
            ]
        )
        init_critic_hstates = jnp.stack(
            [
                ScannedRNN.initialize_carry(
                    config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
                )
                for _ in range(num_agents)
            ]
        )

        # Main training loop
        def _update_step(runner_state, update_step):
            # Preserve rollout-start actor and critic memories for PPO sequence replay.
            rollout_actor_hidden, rollout_critic_hidden = runner_state[4:6]

            def _env_step(runner_state, step_idx):
                (
                    train_states,
                    env_state,
                    obs_data,
                    done_batch,
                    actor_hstates,
                    critic_hstates,
                    rng,
                ) = runner_state

                # Vectorized action selection
                rng, *_rngs = jax.random.split(rng, num_agents + 1)
                action_rngs = jnp.array(_rngs)

                # Get available actions
                try:
                    all_avail_actions = jax.vmap(env.get_avail_actions)(
                        env_state.env_state
                    )
                    avail_batch = jax.lax.stop_gradient(
                        stack_agent_data(all_avail_actions)
                    )
                except (AttributeError, NotImplementedError):
                    avail_batch = jnp.ones(
                        (
                            num_agents,
                            config["NUM_ENVS"],
                            action_dim(env, env.agents[0]),
                        )
                    )

                # Extract agent observations for actors
                obs_batch = obs_data["agent_obs"]

                def single_agent_forward(params, hstate, obs, done, avail, rng_key):
                    ac_in = (
                        obs[np.newaxis, :],
                        done[np.newaxis, :],
                        avail[np.newaxis, :],
                    )
                    new_hstate, pi = actor_network.apply(params, hstate, ac_in)
                    action = pi.sample(seed=rng_key)
                    log_prob = pi.log_prob(action)
                    return new_hstate, action.squeeze(0), log_prob.squeeze(0)

                # Vectorize across all agents for actors
                vmapped_forward = jax.vmap(
                    single_agent_forward, in_axes=(0, 0, 0, 0, 0, 0)
                )
                new_actor_hstates, actions_batch, log_probs_batch = vmapped_forward(
                    train_states[0].params,
                    actor_hstates,
                    obs_batch,
                    done_batch,
                    avail_batch,
                    action_rngs,
                )

                # Get values from centralized critics - use wrapper's world_state
                world_state = obs_data["world_state"]
                # world_state from wrapper is (num_envs, num_agents, world_state_size)
                # Put agents first: (num_agents, num_envs, world_state_size).
                world_state = world_state.swapaxes(0, 1)

                def single_critic_forward(params, hstate, world_state, done):
                    cr_in = (world_state[np.newaxis, :], done[np.newaxis, :])
                    new_hstate, value = critic_network.apply(params, hstate, cr_in)
                    return new_hstate, value.squeeze(0)

                vmapped_critic_forward = jax.vmap(
                    single_critic_forward, in_axes=(0, 0, 0, 0)
                )
                new_critic_hstates, values_batch = vmapped_critic_forward(
                    train_states[1].params, critic_hstates, world_state, done_batch
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
                new_obs_data = stack_agent_data(obsv)
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
                    reward=rewards_batch,
                    log_prob=log_probs_batch,
                    obs=obs_batch,  # Agent observations only
                    world_state=world_state,  # Centralized world state
                    info=info,
                    avail_actions=avail_batch,
                )

                runner_state = (
                    train_states,
                    env_state,
                    new_obs_data,
                    new_done_batch,
                    new_actor_hstates,
                    new_critic_hstates,
                    rng,
                )
                return runner_state, transition

            # Collect trajectory
            (runner_state[4], runner_state[5])
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, jnp.arange(config["NUM_STEPS"])
            )

            (
                train_states,
                env_state,
                obs_data,
                done_batch,
                actor_hstates,
                critic_hstates,
                rng,
            ) = runner_state

            # Calculate last values for bootstrapping
            last_world_state = obs_data["world_state"]
            # Reshape properly for critic input
            last_world_state = last_world_state.swapaxes(0, 1)

            def get_last_value(params, hstate, world_state, done):
                cr_in = (world_state[np.newaxis, :], done[np.newaxis, :])
                _, last_val = critic_network.apply(params, hstate, cr_in)
                return last_val.squeeze(0)

            vmapped_last_value = jax.vmap(get_last_value, in_axes=(0, 0, 0, 0))
            last_vals = vmapped_last_value(
                train_states[1].params, critic_hstates, last_world_state, done_batch
            )

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

            # Update each agent's actor and critic.
            def _single_agent_actor_update(
                agent_actor_train_state,
                agent_traj,
                agent_advantages,
                rng,
                initial_hidden,
            ):
                """Update one actor; vmap applies this across agents."""

                def _update_epoch(update_state, unused):
                    def _update_minibatch(train_state, batch_info):
                        traj_batch, advantages, init_hstate = batch_info

                        def _loss_fn(params, traj_batch, gae, init_hstate):
                            # For RNN, preserve sequence structure
                            obs_input = traj_batch.obs  # (timesteps, envs, obs_dim)
                            done_input = traj_batch.done  # (timesteps, envs)
                            avail_input = (
                                traj_batch.avail_actions
                            )  # (timesteps, envs, action_dim)

                            # Run network forward with proper sequence
                            _, pi = actor_network.apply(
                                params,
                                init_hstate,
                                (obs_input, done_input, avail_input),
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

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, traj_batch, advantages, init_hstate
                        )
                        train_state = train_state.apply_gradients(grads=grads)

                        loss_info = {
                            "total_loss": total_loss[0],
                            "actor_loss": total_loss[1][0],
                            "entropy": total_loss[1][1],
                            "ratio": total_loss[1][2],
                        }

                        return train_state, loss_info

                    agent_actor_train_state, agent_traj, agent_advantages, rng = (
                        update_state
                    )
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

                    # Create hidden states for each minibatch
                    mb_hstates = jnp.take(initial_hidden, permutation, axis=0).reshape(
                        config["NUM_MINIBATCHES"], mb_size, config["GRU_HIDDEN_DIM"]
                    )

                    # Combine into minibatch format: (num_minibatches, ...)
                    minibatches = (mb_traj, mb_advantages, mb_hstates)

                    agent_actor_train_state, loss_info = jax.lax.scan(
                        _update_minibatch, agent_actor_train_state, minibatches
                    )
                    update_state = (
                        agent_actor_train_state,
                        agent_traj,
                        agent_advantages,
                        rng,
                    )
                    return update_state, loss_info

                update_state = (
                    agent_actor_train_state,
                    agent_traj,
                    agent_advantages,
                    rng,
                )
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )

                return update_state[0], loss_info

            def _single_agent_critic_update(
                agent_critic_train_state, agent_traj, agent_targets, rng, initial_hidden
            ):
                """Update one critic; vmap applies this across agents."""

                def _update_epoch(update_state, unused):
                    def _update_minibatch(train_state, batch_info):
                        traj_batch, targets, init_hstate = batch_info

                        def _loss_fn(params, traj_batch, targets, init_hstate):
                            # For RNN, preserve sequence structure
                            world_state_input = (
                                traj_batch.world_state
                            )  # (timesteps, envs, world_state_dim)
                            done_input = traj_batch.done  # (timesteps, envs)

                            # Run network forward with proper sequence
                            _, value = critic_network.apply(
                                params,
                                init_hstate,
                                (world_state_input, done_input),
                            )

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

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, traj_batch, targets, init_hstate
                        )
                        train_state = train_state.apply_gradients(grads=grads)

                        loss_info = {
                            "critic_loss": total_loss[1],
                        }

                        return train_state, loss_info

                    agent_critic_train_state, agent_traj, agent_targets, rng = (
                        update_state
                    )
                    rng, _rng = jax.random.split(rng)

                    # For RNN, preserve sequence structure and only permute across environments
                    _seq_len, batch_size = agent_traj.world_state.shape[
                        :2
                    ]  # timesteps, envs

                    # Permute environments only
                    permutation = jax.random.permutation(_rng, batch_size)

                    # Permute across the environment dimension (axis=1)
                    permuted_traj = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=1), agent_traj
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
                    mb_targets = split_minibatch(permuted_targets)

                    # Create hidden states for each minibatch
                    mb_hstates = jnp.take(initial_hidden, permutation, axis=0).reshape(
                        config["NUM_MINIBATCHES"], mb_size, config["GRU_HIDDEN_DIM"]
                    )

                    # Combine into minibatch format: (num_minibatches, ...)
                    minibatches = (mb_traj, mb_targets, mb_hstates)

                    agent_critic_train_state, loss_info = jax.lax.scan(
                        _update_minibatch, agent_critic_train_state, minibatches
                    )
                    update_state = (
                        agent_critic_train_state,
                        agent_traj,
                        agent_targets,
                        rng,
                    )
                    return update_state, loss_info

                update_state = (
                    agent_critic_train_state,
                    agent_traj,
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
            rng, *agent_rngs = jax.random.split(rng, num_agents * 2 + 1)
            actor_rngs = jnp.array(agent_rngs[:num_agents])
            critic_rngs = jnp.array(agent_rngs[num_agents:])

            # Update actors independently across agents.
            vmapped_actor_update = jax.vmap(
                _single_agent_actor_update, in_axes=(0, 0, 0, 0, 0)
            )
            new_actor_train_states, actor_loss_info = vmapped_actor_update(
                train_states[0],
                agent_traj,
                agent_advantages,
                actor_rngs,
                rollout_actor_hidden,
            )

            # Update critics independently across agents.
            vmapped_critic_update = jax.vmap(
                _single_agent_critic_update, in_axes=(0, 0, 0, 0, 0)
            )
            new_critic_train_states, critic_loss_info = vmapped_critic_update(
                train_states[1],
                agent_traj,
                agent_targets,
                critic_rngs,
                rollout_critic_hidden,
            )

            # Store the updated actor and critic states.
            train_states = (new_actor_train_states, new_critic_train_states)

            # Aggregate loss info from all agents
            actor_loss_info_mean = jax.tree.map(lambda x: x.mean(), actor_loss_info)
            critic_loss_info_mean = jax.tree.map(lambda x: x.mean(), critic_loss_info)

            loss_info_mean = {
                "total_loss": actor_loss_info_mean["total_loss"]
                + critic_loss_info_mean["critic_loss"],
                "actor_loss": actor_loss_info_mean["actor_loss"],
                "critic_loss": critic_loss_info_mean["critic_loss"],
                "entropy": actor_loss_info_mean["entropy"],
                "ratio": actor_loss_info_mean["ratio"],
            }

            # Aggregate metrics
            metric = jax.tree.map(lambda x: jnp.mean(x), traj_batch.info)
            metric["loss"] = loss_info_mean

            metric["update_steps"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

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
                obs_data,
                done_batch,
                actor_hstates,
                critic_hstates,
                rng,
            )
            return runner_state, metric

        # Initial evaluation
        rng, _rng_eval = jax.random.split(rng)
        initial_test_metrics = run_eval(_rng_eval, train_states, test_env, config)

        # Initial runner state
        runner_state = (
            train_states,
            env_state,
            obs_data,  # Changed from obs_batch to obs_data
            jnp.zeros((num_agents, config["NUM_ENVS"]), dtype=bool),
            init_actor_hstates,
            init_critic_hstates,
            _rng,
        )

        # Run training
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )
        return {
            "runner_state": runner_state,
            "metrics": metric,
        }

    return train
