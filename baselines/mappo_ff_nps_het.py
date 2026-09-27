"""MAPPO with separate feed-forward policies for unequal agent spaces.

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

# Networks


class ActorFF(nn.Module):
    """Feed-forward actor with separate parameters for each agent."""

    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, x, return_intermediates=False):
        act_str = self.config.get("ACTIVATION", "relu")
        activation = nn.relu if act_str == "relu" else nn.tanh
        obs, avail_actions = x

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        hidden = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
        )(embedding)
        hidden = activation(hidden)

        pre_softmax_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(hidden)

        if avail_actions is not None:
            unavail = 1 - avail_actions
            action_logits = pre_softmax_logits - (unavail * 1e10)
        else:
            action_logits = pre_softmax_logits

        pi = distrax.Categorical(logits=action_logits)

        if return_intermediates:
            return pi, pre_softmax_logits, hidden
        else:
            return pi


class CriticFF(nn.Module):
    """Centralized critic per agent. Input: world_state (E, ws_dim)."""

    config: dict

    @nn.compact
    def __call__(self, world_state, return_intermediates=False):
        act_str = self.config.get("ACTIVATION", "relu")
        activation = nn.relu if act_str == "relu" else nn.tanh

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        embedding = activation(embedding)

        hidden = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
        )(embedding)
        hidden = activation(hidden)

        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            hidden
        )
        value = jnp.squeeze(value, axis=-1)

        if return_intermediates:
            return value, hidden
        else:
            return value


# Data container


class Transition(NamedTuple):
    global_done: jnp.ndarray  # (E,)
    done: dict[str, jnp.ndarray]  # a: (E,)
    action: dict[str, jnp.ndarray]  # a: (E,)
    value: dict[str, jnp.ndarray]  # a: (E,)
    reward: dict[str, jnp.ndarray]  # a: (E,)
    log_prob: dict[str, jnp.ndarray]  # a: (E,)
    obs: dict[str, jnp.ndarray]  # a: (E, obs_dim_a)
    world_state: jnp.ndarray  # (E, ws_dim)
    info: dict[str, Any]
    avail_actions: dict[str, jnp.ndarray]  # a: (E, act_n_a)


# Evaluation (actor-only)


def run_eval(
    rng, actor_states: dict[str, TrainState], test_env, config, collect_data=False
):
    """Evaluate greedy policies and track episode resets in collected trajectories."""
    return evaluate(
        rng,
        actor_states,
        test_env,
        config,
        recurrent=False,
        independent=False,
        heterogeneous=True,
        collect_data=collect_data,
    )


# Training


def make_train(config):
    env = make_env(config, use_log_wrapper=True, use_state_wrapper=True)
    test_env = make_env(config, use_log_wrapper=True, use_state_wrapper=True)

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"]) // config[
        "NUM_MINIBATCHES"
    ]
    if config.get("SCALE_CLIP_EPS", False):
        config["CLIP_EPS"] = config["CLIP_EPS"] / len(env.agents)

    config.setdefault("TEST_NUM_ENVS", config["NUM_ENVS"])
    config.setdefault("TEST_NUM_STEPS", config["NUM_STEPS"])
    config.setdefault("TEST_INTERVAL", 0.1)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    ws_dim = env.world_state_size()  # Python int

    def train(rng):
        agents = tuple(env.agents)
        A = len(agents)
        E = config["NUM_ENVS"]
        T = config["NUM_STEPS"]

        def obs_shape(a):
            try:
                return env.observation_space(a).shape
            except (TypeError, AttributeError):
                return env.observation_space().shape

        obs_shapes = {a: obs_shape(a) for a in agents}
        act_sizes = {a: env.action_space(a).n for a in agents}

        actor_nets = {a: ActorFF(act_sizes[a], config=config) for a in agents}
        critic_nets = {a: CriticFF(config=config) for a in agents}

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        rng, *ks = jax.random.split(rng, 2 * A + 1)
        a_keys = dict(zip(agents, ks[:A]))
        c_keys = dict(zip(agents, ks[A:]))

        actor_states: dict[str, TrainState] = {}
        critic_states: dict[str, TrainState] = {}

        for a in agents:
            flat_obs = int(np.prod(obs_shapes[a]))
            x_actor = (jnp.zeros((E, flat_obs)), jnp.zeros((E, act_sizes[a])))
            a_params = actor_nets[a].init(a_keys[a], x_actor)

            x_critic = jnp.zeros((E, ws_dim))
            c_params = critic_nets[a].init(c_keys[a], x_critic)

            actor_states[a] = TrainState.create(
                apply_fn=actor_nets[a].apply, params=a_params, tx=actor_tx
            )
            critic_states[a] = TrainState.create(
                apply_fn=critic_nets[a].apply, params=c_params, tx=critic_tx
            )

        # reset
        rng, rkey = jax.random.split(rng)
        rkeys = jax.random.split(rkey, E)
        obsv, env_state = jax.vmap(env.reset)(rkeys)
        obs_data = {
            "obs": {a: obsv[a] for a in agents},
            "world_state": obsv["world_state"][:, 0, :],
        }

        # Metric shapes for updates without evaluation
        rng, er = jax.random.split(rng)
        initial_test_metrics = run_eval(
            er, actor_states, test_env, config, collect_data=False
        )

        def _env_step(state, t):
            actor_states, critic_states, env_state, obs_data, rng = state
            obs = obs_data["obs"]
            world_state = obs_data["world_state"]  # (E, ws_dim)

            # avail
            try:
                all_av = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail = {a: jax.lax.stop_gradient(all_av[a]) for a in agents}
            except (AttributeError, NotImplementedError):
                avail = {a: jnp.ones((E, act_sizes[a])) for a in agents}

            actions, logps, values = {}, {}, {}
            rng, *split = jax.random.split(rng, A + 1)
            akeys = dict(zip(agents, split))

            for a in agents:
                per_env_keys = jax.random.split(akeys[a], E)

                def _act(o_i, av_i, k_i):
                    pi = actor_states[a].apply_fn(actor_states[a].params, (o_i, av_i))
                    act = pi.sample(seed=k_i)
                    return act, pi.log_prob(act)

                acts, lps = jax.vmap(_act)(obs[a], avail[a], per_env_keys)
                actions[a], logps[a] = acts, lps
                values[a] = critic_states[a].apply_fn(
                    critic_states[a].params, world_state
                )

            # step env
            rng, sk = jax.random.split(rng)
            step_keys = jax.random.split(sk, E)
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                step_keys, env_state, actions
            )

            next_obs = {a: obsv[a] for a in agents}
            next_ws = obsv["world_state"][:, 0, :]

            # global done
            if "__all__" in done:
                gdone = jnp.asarray(done["__all__"])
                if gdone.ndim > 1:
                    gdone = jnp.any(gdone, axis=tuple(range(1, gdone.ndim)))
            else:
                stack_d = jnp.stack([done[a] for a in agents], axis=0)
                gdone = jnp.any(stack_d, axis=0)

            info = jax.tree.map(lambda x: x.T if hasattr(x, "T") else x, info)

            tr = Transition(
                global_done=gdone,
                done={a: done[a] for a in agents},
                action={a: actions[a] for a in agents},
                value={a: values[a] for a in agents},
                reward={a: reward[a] for a in agents},
                log_prob={a: logps[a] for a in agents},
                obs={a: obs[a] for a in agents},
                world_state=world_state,
                info=info,
                avail_actions=avail,
            )
            return (
                actor_states,
                critic_states,
                env_state,
                {"obs": next_obs, "world_state": next_ws},
                rng,
            ), tr

        def _update(update_state, update_idx):
            (actor_states, critic_states, env_state, obs_data, rng) = update_state

            # rollout
            (actor_states, critic_states, env_state, obs_data, rng), traj = (
                jax.lax.scan(
                    _env_step,
                    (actor_states, critic_states, env_state, obs_data, rng),
                    jnp.arange(T),
                )
            )

            # last values
            last_ws = obs_data["world_state"]  # (E, ws_dim)
            last_vals = {
                a: critic_states[a].apply_fn(critic_states[a].params, last_ws)
                for a in agents
            }

            # aliases
            gdone_te = traj.global_done
            rew_te = traj.reward
            val_te = traj.value
            act_te = traj.action
            logp_te = traj.log_prob
            obs_te = traj.obs
            avail_te = traj.avail_actions
            ws_te = traj.world_state  # (T,E,ws_dim)

            # GAE
            advantages, targets = {}, {}
            for a in agents:
                dglob = gdone_te
                r = rew_te[a]
                v = val_te[a]
                lv = last_vals[a]

                def scan_gae(carry, x):
                    gae, next_v = carry
                    gd, r_t, v_t = x
                    delta = r_t + config["GAMMA"] * next_v * (1.0 - gd) - v_t
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - gd) * gae
                    )
                    return (gae, v_t), gae

                (_, _), adv = jax.lax.scan(
                    scan_gae, (jnp.zeros_like(lv), lv), (dglob, r, v), reverse=True
                )
                advantages[a] = adv
                targets[a] = adv + v

            # PPO updates per agent
            new_actor_states, new_critic_states = {}, {}
            per_agent_loss = {}

            B = T * E
            MB = config["NUM_MINIBATCHES"]
            bs = B // MB

            for a in agents:
                a_state = actor_states[a]
                c_state = critic_states[a]

                o_flat = obs_te[a].reshape(B, -1)
                av_flat = avail_te[a].reshape(B, -1)
                ac_flat = act_te[a].reshape(
                    B,
                )
                olp_flat = logp_te[a].reshape(
                    B,
                )
                adv_flat = advantages[a].reshape(
                    B,
                )
                tgt_flat = targets[a].reshape(
                    B,
                )
                vh_flat = val_te[a].reshape(
                    B,
                )
                ws_flat = ws_te.reshape(B, -1)

                # shuffle once
                rng, pk = jax.random.split(rng)
                idx = jax.random.permutation(pk, B)
                (
                    o_flat,
                    av_flat,
                    ac_flat,
                    olp_flat,
                    adv_flat,
                    tgt_flat,
                    vh_flat,
                    ws_flat,
                ) = jax.tree.map(
                    lambda x: jnp.take(x, idx, axis=0),
                    (
                        o_flat,
                        av_flat,
                        ac_flat,
                        olp_flat,
                        adv_flat,
                        tgt_flat,
                        vh_flat,
                        ws_flat,
                    ),
                )

                # reshape to (MB, bs, ...)
                def chunk(x):
                    return jnp.reshape(x, (MB, bs) + x.shape[1:])

                o_mb, av_mb = chunk(o_flat), chunk(av_flat)
                ac_mb, olp_mb = chunk(ac_flat), chunk(olp_flat)
                adv_mb, tgt_mb, vh_mb = chunk(adv_flat), chunk(tgt_flat), chunk(vh_flat)
                ws_mb = chunk(ws_flat)

                def actor_loss_fn(params, o, av, act, oldp, gae):
                    pi = actor_nets[a].apply(params, (o, av))
                    logp = pi.log_prob(act)
                    ratio = jnp.exp(logp - oldp)
                    gae_n = (gae - gae.mean()) / (gae.std() + 1e-8)
                    l1 = ratio * gae_n
                    l2 = (
                        jnp.clip(
                            ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
                        )
                        * gae_n
                    )
                    a_loss = -jnp.minimum(l1, l2).mean()
                    ent = pi.entropy().mean()
                    return a_loss - config["ENT_COEF"] * ent, (a_loss, ent)

                def critic_loss_fn(params, ws, tgt, vhist):
                    v = critic_nets[a].apply(params, ws)
                    v_clip = vhist + (v - vhist).clip(
                        -config["CLIP_EPS"], config["CLIP_EPS"]
                    )
                    v_loss = (
                        0.5 * jnp.maximum((v - tgt) ** 2, (v_clip - tgt) ** 2).mean()
                    )
                    return config["VF_COEF"] * v_loss, v_loss

                # SCANNED MINIBATCH
                def _one_mb(carry, mb_i):
                    ts_a, ts_c, acc = carry
                    # take mb_i via dynamic_index_in_dim (no Python slicing)
                    o_b = jax.lax.dynamic_index_in_dim(
                        o_mb, mb_i, axis=0, keepdims=False
                    )
                    av_b = jax.lax.dynamic_index_in_dim(
                        av_mb, mb_i, axis=0, keepdims=False
                    )
                    ac_b = jax.lax.dynamic_index_in_dim(
                        ac_mb, mb_i, axis=0, keepdims=False
                    )
                    olp_b = jax.lax.dynamic_index_in_dim(
                        olp_mb, mb_i, axis=0, keepdims=False
                    )
                    gae_b = jax.lax.dynamic_index_in_dim(
                        adv_mb, mb_i, axis=0, keepdims=False
                    )
                    tgt_b = jax.lax.dynamic_index_in_dim(
                        tgt_mb, mb_i, axis=0, keepdims=False
                    )
                    vh_b = jax.lax.dynamic_index_in_dim(
                        vh_mb, mb_i, axis=0, keepdims=False
                    )
                    ws_b = jax.lax.dynamic_index_in_dim(
                        ws_mb, mb_i, axis=0, keepdims=False
                    )

                    (a_tot, (a_l, ent)), a_grads = jax.value_and_grad(
                        actor_loss_fn, has_aux=True
                    )(ts_a.params, o_b, av_b, ac_b, olp_b, gae_b)
                    ts_a = ts_a.apply_gradients(grads=a_grads)

                    (c_tot, c_l), c_grads = jax.value_and_grad(
                        critic_loss_fn, has_aux=True
                    )(ts_c.params, ws_b, tgt_b, vh_b)
                    ts_c = ts_c.apply_gradients(grads=c_grads)

                    acc = {
                        "total_loss": acc["total_loss"] + (a_tot + c_tot),
                        "actor_loss": acc["actor_loss"] + a_l,
                        "critic_loss": acc["critic_loss"] + c_l,
                        "entropy": acc["entropy"] + ent,
                    }
                    return (ts_a, ts_c, acc), None

                # SCANNED EPOCH
                def _one_epoch(carry, _):
                    return jax.lax.scan(_one_mb, carry, jnp.arange(MB))

                init_carry = (
                    a_state,
                    c_state,
                    {
                        "total_loss": 0.0,
                        "actor_loss": 0.0,
                        "critic_loss": 0.0,
                        "entropy": 0.0,
                    },
                )
                (a_state, c_state, acc_epoch), _ = jax.lax.scan(
                    _one_epoch, init_carry, jnp.arange(config["UPDATE_EPOCHS"])
                )

                new_actor_states[a] = a_state
                new_critic_states[a] = c_state

                per_agent_loss[a] = {
                    "total_loss": acc_epoch["total_loss"],
                    "actor_loss": acc_epoch["actor_loss"],
                    "critic_loss": acc_epoch["critic_loss"],
                    "entropy": acc_epoch["entropy"],
                }

            # Average losses across agents
            loss_info = {
                k: jnp.mean(jnp.array([per_agent_loss[a][k] for a in agents]))
                for k in ["total_loss", "actor_loss", "critic_loss", "entropy"]
            }

            metric = jax.tree.map(lambda x: jnp.mean(x), traj.info)
            metric = {**metric}
            metric["loss"] = loss_info
            metric["update_steps"] = update_idx
            metric["env_step"] = update_idx * T * E

            # periodic eval (actor only)
            if config.get("TEST_DURING_TRAINING", True):
                eval_interval = max(
                    1, int(config["NUM_UPDATES"] * config.get("TEST_INTERVAL", 0.1))
                )
                should_eval = (update_idx == config["NUM_UPDATES"] - 1) | (
                    update_idx % eval_interval == 0
                )

                def do_eval(_):
                    rng_eval, _ = jax.random.split(rng)
                    return run_eval(
                        rng_eval, new_actor_states, test_env, config, collect_data=False
                    )

                def skip_eval(_):
                    return jax.tree.map(
                        lambda x: (
                            jnp.full_like(x, jnp.nan) if hasattr(x, "shape") else None
                        ),
                        initial_test_metrics,
                    )

                test_metrics = jax.lax.cond(should_eval, do_eval, skip_eval, None)
                metric.update({"test_" + k: v for k, v in test_metrics.items()})

            return (
                new_actor_states,
                new_critic_states,
                env_state,
                obs_data,
                rng,
            ), metric

        runner_state = (actor_states, critic_states, env_state, obs_data, rng)

        (actor_states, critic_states, env_state, obs_data, rng), metrics = jax.lax.scan(
            _update, runner_state, jnp.arange(config["NUM_UPDATES"])
        )
        return {"runner_state": (actor_states, critic_states), "metrics": metrics}

    return train
