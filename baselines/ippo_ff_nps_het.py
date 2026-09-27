"""IPPO with separate feed-forward policies for unequal agent spaces.

Use ``python -m baselines.run`` to train and export diagnostic trajectories.
"""

from typing import Any, NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from baselines.env_utils import make_env
from baselines.rollout import evaluate


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
        else:
            raise ValueError(f"Unsupported ACTIVATION {act_str}")

        # Flatten if needed
        if x.ndim > 2:
            x = x.reshape((x.shape[0], -1))

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
    done: dict[str, jnp.ndarray]  # (E,)
    action: dict[str, jnp.ndarray]  # (E,)
    value: dict[str, jnp.ndarray]  # (E,)
    reward: dict[str, jnp.ndarray]  # (E,)
    log_prob: dict[str, jnp.ndarray]  # (E,)
    obs: dict[str, jnp.ndarray]  # (E, obs_dim_a)
    info: dict[str, Any]
    avail_actions: dict[str, jnp.ndarray]  # (E, act_n_a)


def run_eval(rng, train_states, network, test_env, config, collect_data=False):
    """Evaluate greedy policies and track episode resets in collected trajectories."""
    return evaluate(
        rng,
        train_states,
        test_env,
        config,
        recurrent=False,
        independent=True,
        heterogeneous=True,
        collect_data=collect_data,
    )


def make_train(config):
    env = make_env(config, use_log_wrapper=True)
    test_env = make_env(
        config, use_log_wrapper=True
    )  # Used for in-training eval metrics

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / len(env.agents)
        if config.get("SCALE_CLIP_EPS", False)
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
        agents = tuple(env.agents)
        num_agents = len(agents)

        # Initialize per-agent networks (no parameter sharing)
        network = {a: ActorCritic(env.action_space(a).n, config=config) for a in agents}

        # Per-agent params init
        rng, *rng_inits = jax.random.split(rng, num_agents + 1)
        init_rngs = dict(zip(agents, rng_inits))

        def init_single_agent(agent):
            try:
                init_x = jnp.zeros(env.observation_space(agent).shape)
            except (AttributeError, NotImplementedError):
                # overcooked
                init_x = jnp.zeros(env.observation_space().shape).flatten()
            return network[agent].init(init_rngs[agent], init_x)

        all_params = {a: init_single_agent(a) for a in agents}

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(
                learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"],
                eps=1e-5,
            ),
        )

        def create_single_train_state(agent):
            return TrainState.create(
                apply_fn=network[agent].apply, params=all_params[agent], tx=tx
            )

        # Create independent train states for each agent
        train_states = {a: create_single_train_state(a) for a in agents}

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        obs_batch = {a: obsv[a] for a in agents}

        def _update_step(runner_state, update_step):
            train_states, env_state, obs_batch, rng = runner_state

            # Environment rollout
            def _env_step(runner_state, step_idx):
                train_states, env_state, obs_batch, rng = runner_state

                # Avail actions per agent
                try:
                    all_avail_actions = jax.vmap(env.get_avail_actions)(
                        env_state.env_state
                    )
                    avail_batch = {
                        a: jax.lax.stop_gradient(all_avail_actions[a]) for a in agents
                    }
                except (AttributeError, NotImplementedError, KeyError, TypeError):
                    avail_batch = {
                        a: jnp.ones((config["NUM_ENVS"], env.action_space(a).n))
                        for a in agents
                    }

                rng, *agent_rngs = jax.random.split(rng, num_agents + 1)
                agent_rngs = dict(zip(agents, agent_rngs))

                actions_batch, log_probs_batch, values_batch = {}, {}, {}

                for a in agents:

                    def single_env_forward(o, av, rk):
                        pi, value = network[a].apply(train_states[a].params, o, av)
                        action = pi.sample(seed=rk)
                        log_prob = pi.log_prob(action)
                        return action, log_prob, value

                    ks = jax.random.split(agent_rngs[a], config["NUM_ENVS"])
                    ab = jax.vmap(single_env_forward)(obs_batch[a], avail_batch[a], ks)
                    actions_batch[a], log_probs_batch[a], values_batch[a] = ab

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, actions_batch
                )

                # Apply reward shaping if configured
                if rew_shaping_anneal is not None and "shaped_reward" in info:
                    current_timestep = (
                        update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                        + step_idx * config["NUM_ENVS"]
                    )
                    shaping_factor = rew_shaping_anneal(current_timestep)
                    # Add shaped rewards to base rewards
                    reward = {
                        a: reward[a] + info["shaped_reward"][a] * shaping_factor
                        for a in agents
                    }
                    # Remove shaped_reward from info since it's now incorporated into reward
                    info = {k: v for k, v in info.items() if k != "shaped_reward"}

                new_obs_batch = {a: obsv[a] for a in agents}
                done_batch = {a: done[a] for a in agents}
                rewards_batch = {a: reward[a] for a in agents}
                info_batch = info  # leave as dict (env-defined)

                # Create Transition (dict per agent)
                transition = Transition(
                    done=done_batch,
                    action=actions_batch,
                    value=values_batch,
                    reward=rewards_batch,
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

            # Last-value computation for GAE
            def get_last_values(train_states, env_state, obs_batch):
                try:
                    last_avail_actions = jax.vmap(env.get_avail_actions)(
                        env_state.env_state
                    )
                    last_avail_batch = {a: last_avail_actions[a] for a in agents}
                except (AttributeError, NotImplementedError, KeyError, TypeError):
                    last_avail_batch = {
                        a: jnp.ones((config["NUM_ENVS"], env.action_space(a).n))
                        for a in agents
                    }

                last_vals = {}
                for a in agents:
                    _, last_vals[a] = network[a].apply(
                        train_states[a].params, obs_batch[a], last_avail_batch[a]
                    )
                return last_vals

            last_vals = get_last_values(train_states, env_state, obs_batch)

            # Use already-stacked time axes from lax.scan
            done_te = traj_batch.done  # dict[a] -> (T,E)
            reward_te = traj_batch.reward  # dict[a] -> (T,E)
            value_te = traj_batch.value  # dict[a] -> (T,E)
            action_te = traj_batch.action  # dict[a] -> (T,E)
            logp_te = traj_batch.log_prob  # dict[a] -> (T,E)
            obs_te = traj_batch.obs  # dict[a] -> (T,E,obs_dim)
            avail_te = traj_batch.avail_actions  # dict[a] -> (T,E,act_n)

            # GAE per agent
            advantages, targets = {}, {}
            for a in agents:
                dones = done_te[a]
                rews = reward_te[a]
                vals = value_te[a]
                last_v = last_vals[a]  # (E,)

                def _get_advantages(carry, inputs):
                    gae, next_value = carry
                    d_t, r_t, v_t = inputs
                    delta = r_t + config["GAMMA"] * next_value * (1 - d_t) - v_t
                    gae = (
                        delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - d_t) * gae
                    )
                    return (gae, v_t), gae

                (_, _), adv = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_v), last_v),
                    (dones, rews, vals),
                    reverse=True,
                )
                advantages[a] = adv
                targets[a] = adv + vals

            # PPO Update per agent
            new_train_states = {}
            per_agent_loss = {}
            T = config["NUM_STEPS"]
            E = config["NUM_ENVS"]
            B = T * E

            for a in agents:
                st = train_states[a]
                # Flatten buffers
                o_flat = obs_te[a].reshape(B, -1)
                av_flat = avail_te[a].reshape(B, -1)
                act_flat = action_te[a].reshape(
                    B,
                )
                old_lp = logp_te[a].reshape(
                    B,
                )
                adv_flat = advantages[a].reshape(
                    B,
                )
                tgt_flat = targets[a].reshape(
                    B,
                )
                v_hist = value_te[a].reshape(
                    B,
                )

                # Shuffle
                rng, perm_key = jax.random.split(rng)
                idx = jax.random.permutation(perm_key, B)
                o_flat, av_flat, act_flat, old_lp, adv_flat, tgt_flat, v_hist = (
                    jax.tree.map(
                        lambda x: jnp.take(x, idx, axis=0),
                        (o_flat, av_flat, act_flat, old_lp, adv_flat, tgt_flat, v_hist),
                    )
                )

                def _loss_fn(params, o, av, act, oldp, gae, target, vhist):
                    pi, v = network[a].apply(params, o, av)
                    logp = pi.log_prob(act)
                    ratio = jnp.exp(logp - oldp)

                    v_clip = vhist + (v - vhist).clip(
                        -config["CLIP_EPS"], config["CLIP_EPS"]
                    )
                    v_loss = (
                        0.5
                        * jnp.maximum((v - target) ** 2, (v_clip - target) ** 2).mean()
                    )

                    gae_n = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor = -jnp.minimum(
                        ratio * gae_n,
                        jnp.clip(
                            ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
                        )
                        * gae_n,
                    ).mean()
                    entropy = pi.entropy().mean()
                    total = (
                        loss_actor
                        + config["VF_COEF"] * v_loss
                        - config["ENT_COEF"] * entropy
                    )
                    return total, (v_loss, loss_actor, entropy)

                # Epochs x Minibatches
                mb = config["NUM_MINIBATCHES"]
                bs = B // mb

                def one_epoch(state):
                    def take_mb(x, i, bs):
                        start = i * bs
                        # slice [start : start+bs] along axis 0 using JAX primitives
                        return jax.lax.dynamic_slice_in_dim(x, start, bs, axis=0)

                    def mb_body(state, i):
                        o_b = take_mb(o_flat, i, bs)
                        av_b = take_mb(av_flat, i, bs)
                        act_b = take_mb(act_flat, i, bs)
                        oldp_b = take_mb(old_lp, i, bs)
                        adv_b = take_mb(adv_flat, i, bs)
                        tgt_b = take_mb(tgt_flat, i, bs)
                        vh_b = take_mb(v_hist, i, bs)

                        (tot, (vl, al, en)), grads = jax.value_and_grad(
                            _loss_fn, has_aux=True
                        )(state.params, o_b, av_b, act_b, oldp_b, adv_b, tgt_b, vh_b)
                        state = state.apply_gradients(grads=grads)
                        return state, (tot, vl, al, en)

                    state, totals = jax.lax.scan(mb_body, state, jnp.arange(mb))
                    return state, jax.tree.map(lambda x: x.mean(), totals)

                def epoch_body(state, _):
                    st, stats = one_epoch(state)
                    return st, stats

                st, epoch_stats = jax.lax.scan(
                    epoch_body, st, jnp.arange(config["UPDATE_EPOCHS"])
                )
                total_mean, v_mean, a_mean, e_mean = jax.tree.map(
                    lambda x: x.mean(), epoch_stats
                )

                new_train_states[a] = st

                per_agent_loss[a] = {
                    "total_loss": total_mean,
                    "actor_loss": a_mean,
                    "critic_loss": v_mean,
                    "entropy": e_mean,
                }

            train_states = new_train_states
            # Average losses across agents
            loss_info = {
                k: jnp.mean(jnp.array([per_agent_loss[a][k] for a in agents]))
                for k in ["total_loss", "actor_loss", "critic_loss", "entropy"]
            }

            metric = traj_batch.info
            metric = jax.tree.map(lambda x: jnp.mean(x), metric)
            metric = {**metric}
            metric["loss"] = loss_info
            metric["update_steps"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

            # Add reward shaping metrics if configured
            if (
                rew_shaping_anneal is not None
                and isinstance(traj_batch.info, dict)
                and "shaped_reward" in traj_batch.info
            ):
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
            _rng_eval,
            {a: train_states[a] for a in train_states},
            network,
            test_env,
            config,
            collect_data=False,
        )

        # Initial runner state
        runner_state = (train_states, env_state, obs_batch, rng)
        # Run training
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train
