"""MAPPO with separate recurrent policies for unequal agent spaces.

Use ``python -m baselines.run`` to train and export diagnostic trajectories.
"""

import functools
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


# RNN core
class ScannedRNN(nn.Module):
    """Time-scanned GRU with reset on episode boundaries.

    Contract: __call__(carry, (inputs, resets)) with nn.scan(in_axes=0).
    hidden_size is static and passed in the constructor.
    """

    hidden_size: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,  # scan over time for BOTH leaves of the tuple
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        # x = (inputs, resets)
        state = carry  # (E, H)
        inputs, resets = x  # (E, F), (E,) or scalar True/False per env

        # reset hidden on episode boundaries
        state = jnp.where(
            resets[:, None],
            self.initialize_carry(state.shape[0], self.hidden_size),
            state,
        )
        new_state, y = nn.GRUCell(features=self.hidden_size)(state, inputs)
        return new_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


# Actor / Critic
class ActorRNN(nn.Module):
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, hidden, x, return_intermediates=False):
        # x: (obs, dones, avail)
        act_str = self.config.get("ACTIVATION", "relu")
        activation = nn.relu if act_str == "relu" else nn.tanh
        H = int(self.config["GRU_HIDDEN_DIM"])

        obs, dones, avail = x  # (T,E,Do), (T,E), (T,E,A) or None

        # embed
        emb = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        emb = activation(emb)

        # RNN over time (carry is (E,H))
        hidden, rnn_out = ScannedRNN(H)(hidden, (emb, dones))

        # small head
        h = nn.Dense(
            H,
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
        )(rnn_out)
        h = activation(h)

        pre_softmax_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(h)

        logits = pre_softmax_logits
        if avail is not None:
            logits = pre_softmax_logits - ((1 - avail) * 1e10)

        pi = distrax.Categorical(logits=logits)  # (T,E)
        if return_intermediates:
            return hidden, pi, pre_softmax_logits, rnn_out
        else:
            return hidden, pi


class CriticRNN(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, hidden, x, return_intermediates=False):
        # x: (world_state, dones)
        act_str = self.config.get("ACTIVATION", "relu")
        activation = nn.relu if act_str == "relu" else nn.tanh
        H = int(self.config["GRU_HIDDEN_DIM"])

        ws, dones = x  # (T,E,ws), (T,E)

        emb = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(ws)
        emb = activation(emb)

        hidden, rnn_out = ScannedRNN(H)(hidden, (emb, dones))

        h = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
        )(rnn_out)
        h = activation(h)

        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(h)
        if return_intermediates:
            return hidden, jnp.squeeze(v, axis=-1), rnn_out
        else:
            return hidden, jnp.squeeze(v, axis=-1)


# Transition
class Transition(NamedTuple):
    global_done: jnp.ndarray  # (E,)
    done: dict[str, jnp.ndarray]  # a: (E,)
    action: dict[str, jnp.ndarray]  # a: (E,)
    value: dict[str, jnp.ndarray]  # a: (E,)
    reward: dict[str, jnp.ndarray]  # a: (E,)
    log_prob: dict[str, jnp.ndarray]  # a: (E,)
    obs: dict[str, jnp.ndarray]  # a: (E, Do_a)
    world_state: jnp.ndarray  # (E, A, ws)
    info: dict[str, Any]
    avail_actions: dict[str, jnp.ndarray]  # a: (E, A_a)


# Eval
def run_eval(
    rng,
    train_states: dict[str, dict[str, TrainState]],
    test_env,
    config,
    collect_data: bool = False,
):
    """Evaluate greedy policies and track episode resets in collected trajectories."""
    return evaluate(
        rng,
        train_states["actor"],
        test_env,
        config,
        recurrent=True,
        independent=False,
        heterogeneous=True,
        collect_data=collect_data,
    )


# Training
def make_train(config):
    # ensure wrapper attaches "world_state" to obs as (E, A, ws_dim)
    env = make_env(config, use_log_wrapper=True, use_state_wrapper=True)
    test_env = make_env(config, use_log_wrapper=True, use_state_wrapper=True)

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    if config.get("SCALE_CLIP_EPS", False):
        config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents

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

    def train(rng):
        agents = tuple(env.agents)
        A = len(agents)
        E = config["NUM_ENVS"]
        T = config["NUM_STEPS"]
        H = int(config["GRU_HIDDEN_DIM"])

        # dims per agent
        obs_dims = {}
        act_dims = {}
        for a in agents:
            try:
                oshape = env.observation_space(a).shape
            except (AttributeError, NotImplementedError):
                oshape = env.observation_space().shape
            obs_dims[a] = int(np.prod(oshape))
            act_dims[a] = env.action_space(a).n

        # networks per agent
        actor_nets = {a: ActorRNN(act_dims[a], config=config) for a in agents}
        critic_nets = {a: CriticRNN(config=config) for a in agents}

        # init params
        rng, *ks = jax.random.split(rng, 2 * A + 1)
        a_keys = dict(zip(agents, ks[:A]))
        c_keys = dict(zip(agents, ks[A:]))

        if config["ANNEAL_LR"]:
            tx_actor = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            tx_critic = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx_actor = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            tx_critic = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        actor_states: dict[str, TrainState] = {}
        critic_states: dict[str, TrainState] = {}

        for a in agents:
            # Actor init with (T=1,E,Do), (T=1,E), (T=1,E,A)
            x_actor = (
                jnp.zeros((1, E, obs_dims[a])),
                jnp.zeros((1, E), dtype=bool),
                jnp.zeros((1, E, act_dims[a])),
            )
            hA0 = ScannedRNN.initialize_carry(E, H)
            a_params = actor_nets[a].init(a_keys[a], hA0, x_actor)
            actor_states[a] = TrainState.create(
                apply_fn=actor_nets[a].apply, params=a_params, tx=tx_actor
            )

            # Critic init with (T=1,E,ws), (T=1,E)
            x_critic = (
                jnp.zeros((1, E, env.world_state_size())),
                jnp.zeros((1, E), dtype=bool),
            )
            hC0 = ScannedRNN.initialize_carry(E, H)
            c_params = critic_nets[a].init(c_keys[a], hC0, x_critic)
            critic_states[a] = TrainState.create(
                apply_fn=critic_nets[a].apply, params=c_params, tx=tx_critic
            )

        # reset envs
        rng, rkey = jax.random.split(rng)
        rkeys = jax.random.split(rkey, E)
        obsv, env_state = jax.vmap(env.reset)(rkeys)

        obs = {a: obsv[a] for a in agents}  # a: (E, Do)
        ws_all = obsv["world_state"]  # (E, A, ws)

        actor_h = {a: ScannedRNN.initialize_carry(E, H) for a in agents}
        critic_h = {a: ScannedRNN.initialize_carry(E, H) for a in agents}
        done_buf = {a: jnp.zeros((E,), dtype=bool) for a in agents}

        # Evaluate once to determine metric shapes.
        rng, er = jax.random.split(rng)
        initial_test_metrics = run_eval(
            er, {"actor": actor_states, "critic": critic_states}, test_env, config
        )

        # single env step
        def _env_step(state, _):
            (
                actor_states,
                critic_states,
                env_state,
                obs,
                ws_all,
                done_buf,
                actor_h,
                critic_h,
                rng,
            ) = state

            # avail masks
            try:
                all_av = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail = {a: jax.lax.stop_gradient(all_av[a]) for a in agents}
            except (AttributeError, NotImplementedError):
                avail = {a: jnp.ones((E, act_dims[a])) for a in agents}

            actions, logps, values = {}, {}, {}

            rng, *split = jax.random.split(rng, A + 1)
            akeys = dict(zip(agents, split))

            # per-agent actor (T=1 into net), batched sample
            for a in agents:
                ac_in = (obs[a][None, ...], done_buf[a][None, ...], avail[a][None, ...])
                new_h, pi = actor_states[a].apply_fn(
                    actor_states[a].params, actor_h[a], ac_in
                )
                act = pi.sample(seed=akeys[a]).squeeze(0)  # (E,)
                lp = pi.log_prob(act).squeeze(0)  # (E,)
                actor_h[a] = new_h
                actions[a], logps[a] = act, lp

            # per-agent critic (centralized ws slice)
            for i, a in enumerate(agents):
                ws_i = ws_all[:, i, :]  # (E, ws)
                cr_in = (ws_i[None, ...], done_buf[a][None, ...])
                new_h, v = critic_states[a].apply_fn(
                    critic_states[a].params, critic_h[a], cr_in
                )
                critic_h[a] = new_h
                values[a] = v.squeeze(0)  # (E,)

            # env step
            rng, sk = jax.random.split(rng)
            step_keys = jax.random.split(sk, E)
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                step_keys, env_state, actions
            )

            next_obs = {a: obsv[a] for a in agents}
            next_done = {a: done[a] for a in agents}
            ws_all_next = obsv["world_state"]

            # global done per env
            gdone = done["__all__"]

            # info to (E,...) for averaging
            info = jax.tree.map(lambda x: x.T if hasattr(x, "T") else x, info)

            # log step as (E, ...)
            tr = Transition(
                global_done=gdone,
                done={a: done_buf[a] for a in agents},
                action={a: actions[a] for a in agents},
                value={a: values[a] for a in agents},
                reward={a: reward[a] for a in agents},
                log_prob={a: logps[a] for a in agents},
                obs={a: obs[a] for a in agents},
                world_state=ws_all,
                info=info,
                avail_actions={a: avail[a] for a in agents},
            )

            next_state = (
                actor_states,
                critic_states,
                env_state,
                next_obs,
                ws_all_next,
                next_done,
                actor_h,
                critic_h,
                rng,
            )
            return next_state, tr

        # PPO update
        def _update(update_state, update_idx):
            (
                actor_states,
                critic_states,
                env_state,
                obs,
                ws_all,
                done_buf,
                actor_h,
                critic_h,
                rng,
            ) = update_state

            # Replay from the hidden states that generated the rollout.
            rollout_actor_hidden, rollout_critic_hidden = actor_h, critic_h

            # rollout T steps: each field becomes (T,E,...) automatically
            (
                (
                    actor_states,
                    critic_states,
                    env_state,
                    obs,
                    ws_all,
                    done_buf,
                    actor_h,
                    critic_h,
                    rng,
                ),
                traj,
            ) = jax.lax.scan(
                _env_step,
                (
                    actor_states,
                    critic_states,
                    env_state,
                    obs,
                    ws_all,
                    done_buf,
                    actor_h,
                    critic_h,
                    rng,
                ),
                None,
                length=T,
            )

            # last values per agent (bootstrap)
            last_vals = {}
            for i, a in enumerate(agents):
                ws_i = ws_all[:, i, :]  # (E, ws)
                cr_in = (ws_i[None, ...], done_buf[a][None, ...])
                _, v = critic_states[a].apply_fn(
                    critic_states[a].params, critic_h[a], cr_in
                )
                last_vals[a] = v.squeeze(0)  # (E,)

            # rollout tensors (already T,E,...)
            gdone_TE = traj.global_done  # (T,E)
            rew_TE = traj.reward  # dict[a] -> (T,E)
            val_TE = traj.value  # dict[a] -> (T,E)
            act_TE = traj.action  # dict[a] -> (T,E)
            logp_TE = traj.log_prob  # dict[a] -> (T,E)
            obs_TE = traj.obs  # dict[a] -> (T,E,Do)
            avail_TE = traj.avail_actions  # dict[a] -> (T,E,A)
            ws_TE = traj.world_state  # (T,E,A,ws)
            dn_TE = traj.done  # dict[a] -> (T,E)

            # GAE per agent
            advantages, targets = {}, {}
            for a in agents:
                dglob = gdone_TE  # (T,E)
                r = rew_TE[a]  # (T,E)
                v = val_TE[a]  # (T,E)
                lv = last_vals[a]  # (E,)

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

            # PPO updates per agent (minibatch over envs)
            MB = config["NUM_MINIBATCHES"]
            bs = E // MB

            new_actor_states, new_critic_states = {}, {}
            per_agent_loss = {}

            for i, a in enumerate(agents):
                ts_a = actor_states[a]
                ts_c = critic_states[a]

                o = obs_TE[a]  # (T,E,Do)
                av = avail_TE[a]  # (T,E,A)
                ac = act_TE[a]  # (T,E)
                olp = logp_TE[a]  # (T,E)
                adv = advantages[a]  # (T,E)
                tgt = targets[a]  # (T,E)
                vh = val_TE[a]  # (T,E)
                ws = ws_TE[:, :, i, :]  # (T,E,ws) — centralized slice for agent i
                dn = dn_TE[a]  # (T,E)

                # shuffle envs (axis=1)
                rng, pk = jax.random.split(rng)
                perm = jax.random.permutation(pk, E)

                def take(x):
                    return jnp.take(x, perm, axis=1)

                o, av, ac, olp, adv, tgt, vh, ws, dn = map(
                    take, (o, av, ac, olp, adv, tgt, vh, ws, dn)
                )

                # split envs into MB minibatches, keep full time axis intact
                MB = config["NUM_MINIBATCHES"]
                bs = E // MB

                def split_mb(x):
                    # (T,E,...) -> (MB,T,bs,...) for scan over MB
                    new_shape = (x.shape[0], MB, bs) + x.shape[2:]
                    x_ = jnp.reshape(x, new_shape)  # (T,MB,bs,...)
                    return jnp.swapaxes(x_, 0, 1)  # (MB,T,bs,...)

                o_mb, av_mb, ac_mb, olp_mb, adv_mb, tgt_mb, vh_mb, ws_mb, dn_mb = map(
                    split_mb, (o, av, ac, olp, adv, tgt, vh, ws, dn)
                )

                hA0 = jnp.take(rollout_actor_hidden[a], perm, axis=0).reshape(MB, bs, H)
                hC0 = jnp.take(rollout_critic_hidden[a], perm, axis=0).reshape(
                    MB, bs, H
                )

                # losses
                def actor_loss_fn(params, o_b, dn_b, av_b, ac_b, olp_b, gae_b, h0):
                    _, pi = ActorRNN(act_dims[a], config).apply(
                        params, h0, (o_b, dn_b, av_b)
                    )
                    logp = pi.log_prob(ac_b)
                    ratio = jnp.exp(logp - olp_b)
                    gae_n = (gae_b - gae_b.mean()) / (gae_b.std() + 1e-8)
                    l1 = ratio * gae_n
                    l2 = (
                        jnp.clip(
                            ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
                        )
                        * gae_n
                    )
                    a_loss = -jnp.minimum(l1, l2).mean()
                    ent = pi.entropy().mean()
                    return a_loss - config["ENT_COEF"] * ent, (
                        a_loss,
                        ent,
                        ratio.mean(),
                    )

                def critic_loss_fn(params, ws_b, dn_b, tgt_b, vh_b, h0):
                    _, v = CriticRNN(config).apply(params, h0, (ws_b, dn_b))
                    v_clip = vh_b + (v - vh_b).clip(
                        -config["CLIP_EPS"], config["CLIP_EPS"]
                    )
                    vloss = (
                        0.5
                        * jnp.maximum((v - tgt_b) ** 2, (v_clip - tgt_b) ** 2).mean()
                    )
                    return config["VF_COEF"] * vloss, vloss

                # scan over minibatches
                def _update_one_mb(carry, mb_i):
                    ts_a_c, ts_c_c, loss_acc = carry

                    o_b = o_mb[mb_i]  # (T,bs,Do)
                    av_b = av_mb[mb_i]  # (T,bs,A)
                    ac_b = ac_mb[mb_i]  # (T,bs)
                    olp_b = olp_mb[mb_i]  # (T,bs)
                    gae_b = adv_mb[mb_i]  # (T,bs)
                    tgt_b = tgt_mb[mb_i]  # (T,bs)
                    vh_b = vh_mb[mb_i]  # (T,bs)
                    ws_b = ws_mb[mb_i]  # (T,bs,ws)
                    dn_b = dn_mb[mb_i]  # (T,bs)

                    hA = hA0[mb_i]  # (bs,H)
                    hC = hC0[mb_i]  # (bs,H)

                    # actor
                    (a_tot, (a_l, ent, r_mean)), a_grads = jax.value_and_grad(
                        actor_loss_fn, has_aux=True
                    )(ts_a_c.params, o_b, dn_b, av_b, ac_b, olp_b, gae_b, hA)
                    ts_a_c = ts_a_c.apply_gradients(grads=a_grads)

                    # critic
                    (c_tot, c_l), c_grads = jax.value_and_grad(
                        critic_loss_fn, has_aux=True
                    )(ts_c_c.params, ws_b, dn_b, tgt_b, vh_b, hC)
                    ts_c_c = ts_c_c.apply_gradients(grads=c_grads)

                    # accumulate losses
                    loss_acc = {
                        "total_loss": loss_acc["total_loss"] + (a_tot + c_tot),
                        "actor_loss": loss_acc["actor_loss"] + a_l,
                        "critic_loss": loss_acc["critic_loss"] + c_l,
                        "entropy": loss_acc["entropy"] + ent,
                        "ratio": loss_acc["ratio"] + r_mean,
                    }
                    return (ts_a_c, ts_c_c, loss_acc), None

                def _update_one_epoch(carry, _):
                    carry, _ = jax.lax.scan(_update_one_mb, carry, jnp.arange(MB))
                    return carry, None

                init_carry = (
                    ts_a,
                    ts_c,
                    {
                        "total_loss": 0.0,
                        "actor_loss": 0.0,
                        "critic_loss": 0.0,
                        "entropy": 0.0,
                        "ratio": 0.0,
                    },
                )

                (ts_a, ts_c, loss_acc), _ = jax.lax.scan(
                    _update_one_epoch, init_carry, jnp.arange(config["UPDATE_EPOCHS"])
                )

                new_actor_states[a] = ts_a
                new_critic_states[a] = ts_c

                # add to global accumulators
                per_agent_loss[a] = {
                    "total_loss": loss_acc["total_loss"],
                    "actor_loss": loss_acc["actor_loss"],
                    "critic_loss": loss_acc["critic_loss"],
                    "entropy": loss_acc["entropy"],
                    "ratio": loss_acc["ratio"],
                }

            # average losses
            loss_info = {
                k: jnp.mean(jnp.array([per_agent_loss[a][k] for a in agents]))
                for k in ["total_loss", "actor_loss", "critic_loss", "entropy", "ratio"]
            }

            metric = jax.tree.map(lambda x: jnp.mean(x), traj.info)
            metric["loss"] = loss_info
            metric["update_steps"] = update_idx
            metric["env_step"] = update_idx * T * E

            # Periodic evaluation
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
                        rng_eval,
                        {"actor": new_actor_states, "critic": new_critic_states},
                        test_env,
                        config,
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

            update_state = (
                new_actor_states,
                new_critic_states,
                env_state,
                obs,
                ws_all,
                done_buf,
                actor_h,
                critic_h,
                rng,
            )
            return update_state, metric

        # initial runner state
        runner0 = (
            actor_states,
            critic_states,
            env_state,
            obs,
            ws_all,
            done_buf,
            actor_h,
            critic_h,
            rkey,
        )

        # train
        (
            (
                actor_states,
                critic_states,
                env_state,
                obs,
                ws_all,
                done_buf,
                actor_h,
                critic_h,
                _,
            ),
            metrics,
        ) = jax.lax.scan(_update, runner0, jnp.arange(config["NUM_UPDATES"]))

        return {
            "runner_state": (
                actor_states,
                critic_states,
                env_state,
                obs,
                ws_all,
                done_buf,
                actor_h,
                critic_h,
            ),
            "metrics": metrics,
        }

    return train
