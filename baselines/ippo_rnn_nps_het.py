"""IPPO with separate recurrent policies for unequal agent spaces.

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

# RNN Core


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


# Actor-critic per agent


class ActorCriticRNN(nn.Module):
    action_dim: int
    config: dict

    @nn.compact
    def __call__(self, hidden, x, return_intermediates=False):
        act_str = self.config.get("ACTIVATION", "relu")
        activation = nn.relu if act_str == "relu" else nn.tanh

        obs, dones, avail_actions = x  # obs: (T,E,Do) or (1,E,Do)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_h = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
        )(embedding)
        actor_h = activation(actor_h)

        pre_softmax_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_h)

        if avail_actions is not None:
            unavail = 1 - avail_actions
            action_logits = pre_softmax_logits - (unavail * 1e10)
        else:
            action_logits = pre_softmax_logits
        pi = distrax.Categorical(logits=action_logits)

        critic_h = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2.0),
            bias_init=constant(0.0),
        )(embedding)
        critic_h = nn.relu(critic_h)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic_h
        )
        value = jnp.squeeze(value, axis=-1)  # (T,E)

        if return_intermediates:
            return hidden, pi, value, pre_softmax_logits, embedding
        else:
            return hidden, pi, value


# Trajectory fields per agent


class Transition(NamedTuple):
    global_done: jnp.ndarray  # (E,) per step -> stacks to (T,E)
    done: dict[str, jnp.ndarray]  # a: (E,)
    action: dict[str, jnp.ndarray]  # a: (E,)
    value: dict[str, jnp.ndarray]  # a: (E,)
    reward: dict[str, jnp.ndarray]  # a: (E,)
    log_prob: dict[str, jnp.ndarray]  # a: (E,)
    obs: dict[str, jnp.ndarray]  # a: (E, Do_a)
    info: dict[str, Any]
    avail_actions: dict[str, jnp.ndarray]  # a: (E, A_a)


# Evaluation (actor-only)


def run_eval(
    rng,
    train_states: dict[str, TrainState],
    networks: dict[str, ActorCriticRNN],
    test_env,
    config,
    collect_data=False,
):
    """Evaluate greedy policies and track episode resets in collected trajectories."""
    return evaluate(
        rng,
        train_states,
        test_env,
        config,
        recurrent=True,
        independent=True,
        heterogeneous=True,
        collect_data=collect_data,
    )


# Training


def make_train(config):
    env = make_env(config, use_log_wrapper=True)
    test_env = make_env(config, use_log_wrapper=True)

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
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

    # Optional reward shaping schedule
    if config.get("REW_SHAPING_HORIZON") is not None:
        optax.linear_schedule(
            init_value=1.0,
            end_value=0.0,
            transition_steps=config["REW_SHAPING_HORIZON"],
        )

    def train(rng):
        agents = tuple(env.agents)
        A = len(agents)
        E = config["NUM_ENVS"]
        T = config["NUM_STEPS"]

        # Observation and action sizes per agent
        def _obs_dim(a):
            try:
                shp = env.observation_space(a).shape
            except (AttributeError, NotImplementedError):
                shp = env.observation_space().shape
            return int(np.prod(shp))

        obs_dims = {a: _obs_dim(a) for a in agents}
        act_sizes = {a: env.action_space(a).n for a in agents}

        # Per-agent networks & states
        networks = {a: ActorCriticRNN(act_sizes[a], config=config) for a in agents}

        rng, *rks = jax.random.split(rng, A + 1)
        init_keys = dict(zip(agents, rks))

        train_states: dict[str, TrainState] = {}
        for a in agents:
            # Initialize with a single timestep and NUM_ENVS parallel environments.
            init_x = (
                jnp.zeros((1, E, obs_dims[a])),
                jnp.zeros((1, E), dtype=bool),
                jnp.zeros((1, E, act_sizes[a])),
            )
            init_h = ScannedRNN.initialize_carry(E, config["GRU_HIDDEN_DIM"])
            params = networks[a].init(init_keys[a], init_h, init_x)

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(
                    learning_rate=(
                        linear_schedule if config["ANNEAL_LR"] else config["LR"]
                    ),
                    eps=1e-5,
                ),
            )
            train_states[a] = TrainState.create(
                apply_fn=networks[a].apply, params=params, tx=tx
            )

        # Reset envs
        rng, _r = jax.random.split(rng)
        reset_keys = jax.random.split(_r, E)
        obsv, env_state = jax.vmap(env.reset, in_axes=0)(reset_keys)

        # Observations & hidden states per agent
        obs_batch = {a: obsv[a] for a in agents}  # (E, Do_a)
        done_batch = {a: jnp.zeros((E,), dtype=bool) for a in agents}  # (E,)
        hstates = {
            a: ScannedRNN.initialize_carry(E, config["GRU_HIDDEN_DIM"]) for a in agents
        }

        # Evaluation metric shapes
        rng, _re = jax.random.split(rng)
        initial_test_metrics = run_eval(
            _re, train_states, networks, test_env, config, collect_data=False
        )

        # One environment step
        def _env_step(runner_state, step_idx):
            train_states, env_state, obs_batch, done_batch, hstates, rng = runner_state

            # Avail masks per agent
            try:
                all_avail = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail = {a: jax.lax.stop_gradient(all_avail[a]) for a in agents}
            except (AttributeError, NotImplementedError):
                avail = {a: jnp.ones((E, act_sizes[a])) for a in agents}

            actions, logps, values, new_hs = {}, {}, {}, {}

            rng, *akeys = jax.random.split(rng, A + 1)
            akeys = dict(zip(agents, akeys))

            for a in agents:
                # Build a single-step (T=1) batch.
                ac_in = (
                    obs_batch[a][None, ...],  # (1, E, Do_a)
                    done_batch[a][None, ...],  # (1, E)
                    avail[a][None, ...],  # (1, E, A_a)
                )

                # Forward (returns time-major (1, E, ...))
                new_h, pi, val = networks[a].apply(
                    train_states[a].params, hstates[a], ac_in
                )

                # Sample the whole batch once, then squeeze time axis (no vmap needed)
                act = pi.sample(seed=akeys[a]).squeeze(0)  # (E,)
                lp = pi.log_prob(act).squeeze(0)  # (E,)
                val = val.squeeze(0)  # (E,)

                # Update carry and buffers
                new_hs[a] = new_h
                actions[a] = act
                logps[a] = lp
                values[a] = val

            # Step envs
            rng, sk = jax.random.split(rng)
            step_keys = jax.random.split(sk, E)
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                step_keys, env_state, actions
            )

            # Next obs/done
            next_obs = {a: obsv[a] for a in agents}
            next_done = {a: done[a] for a in agents}

            # Global done (per env) robustly to shapes like (E,2)
            if isinstance(done, dict) and "__all__" in done:
                gdone = jnp.asarray(done["__all__"])
                if gdone.ndim > 1:
                    gdone = jnp.any(gdone, axis=tuple(range(1, gdone.ndim)))
            else:
                stack_d = jnp.stack([done[a] for a in agents], axis=0)  # (A,E)
                gdone = jnp.any(stack_d, axis=0)  # (E,)

            # info → (agents, envs) → average-friendly
            info = jax.tree.map(lambda x: x.T if hasattr(x, "T") else x, info)

            tr = Transition(
                global_done=gdone,
                done={a: done_batch[a] for a in agents},
                action={a: actions[a] for a in agents},
                value={a: values[a] for a in agents},
                reward={a: reward[a] for a in agents},
                log_prob={a: logps[a] for a in agents},
                obs={a: obs_batch[a] for a in agents},
                info=info,
                avail_actions=avail,
            )

            return (train_states, env_state, next_obs, next_done, new_hs, rng), tr

        # One PPO update (T steps)
        def _update(update_state, update_idx):
            train_states, env_state, obs_batch, done_batch, hstates, rng = update_state

            # Preserve pre-rollout memory; a batch can begin mid-episode.
            rollout_hidden = hstates

            # Rollout
            (train_states, env_state, obs_batch, done_batch, hstates, rng), traj = (
                jax.lax.scan(
                    _env_step,
                    (train_states, env_state, obs_batch, done_batch, hstates, rng),
                    jnp.arange(T),
                )
            )

            # Last values per agent
            try:
                all_av_fin = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_fin = {a: jax.lax.stop_gradient(all_av_fin[a]) for a in agents}
            except (AttributeError, NotImplementedError):
                avail_fin = {a: jnp.ones((E, act_sizes[a])) for a in agents}

            last_vals = {}
            for a in agents:
                ac_in = (
                    obs_batch[a][None, ...],
                    done_batch[a][None, ...],
                    avail_fin[a][None, ...],
                )
                _, _, lv = networks[a].apply(train_states[a].params, hstates[a], ac_in)
                lv = lv.squeeze(0)  # (E,)
                last_vals[a] = lv

            # Time-stacked tensors
            gdone_te = traj.global_done  # (T,E) or (T,E,?)
            if gdone_te.ndim == 3:
                gdone_te = jnp.any(gdone_te, axis=-1)
            elif gdone_te.ndim > 3:
                gdone_te = jnp.any(gdone_te, axis=tuple(range(2, gdone_te.ndim)))

            rew_te = traj.reward  # dict[a] -> (T,E)
            val_te = traj.value  # dict[a] -> (T,E)
            act_te = traj.action  # dict[a] -> (T,E)
            logp_te = traj.log_prob  # dict[a] -> (T,E)
            obs_te = traj.obs  # dict[a] -> (T,E,Do_a)
            avail_te = traj.avail_actions  # dict[a] -> (T,E,A_a)

            # Per-agent GAE
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

            # Per-agent PPO update (minibatching over envs; keep sequence)
            new_train_states: dict[str, TrainState] = {}
            per_agent_loss = {}

            mb = config["NUM_MINIBATCHES"]
            bs = E // mb  # minibatch size over envs

            for a in agents:
                ts = train_states[a]
                o = obs_te[a]  # (T,E,Do)
                av = avail_te[a]  # (T,E,Aa)
                ac = act_te[a]  # (T,E)
                olp = logp_te[a]  # (T,E)
                adv = advantages[a]  # (T,E)
                tgt = targets[a]  # (T,E)
                vhist = val_te[a]  # (T,E)

                dn = traj.done[a]

                # Shuffle envs
                rng, pk = jax.random.split(rng)
                perm = jax.random.permutation(pk, E)

                o = jnp.take(o, perm, axis=1)
                av = jnp.take(av, perm, axis=1)
                ac = jnp.take(ac, perm, axis=1)
                olp = jnp.take(olp, perm, axis=1)
                adv = jnp.take(adv, perm, axis=1)
                tgt = jnp.take(tgt, perm, axis=1)
                vhist = jnp.take(vhist, perm, axis=1)
                dn = jnp.take(dn, perm, axis=1)

                # Split into minibatches across envs: (mb, T, bs, ...)
                def _split_env(x):
                    new_shape = (x.shape[0], mb, bs) + x.shape[2:]
                    x = jnp.reshape(x, new_shape)  # (T,mb,bs,...)
                    return jnp.swapaxes(x, 0, 1)  # (mb,T,bs,...)

                o_mb, av_mb, ac_mb, olp_mb, adv_mb, tgt_mb, vh_mb = map(
                    _split_env, (o, av, ac, olp, adv, tgt, vhist)
                )

                dn_mb = _split_env(dn)
                h0_mb = jnp.take(rollout_hidden[a], perm, axis=0).reshape(
                    mb, bs, config["GRU_HIDDEN_DIM"]
                )

                def actor_loss_fn(
                    params, o_seq, av_seq, ac_seq, olp_seq, gae_seq, h0_seq, dn_seq
                ):
                    # o_seq: (T,bs,Do)  av_seq: (T,bs,Aa)  ac/olp/gae: (T,bs)  h0_seq: (bs,H)
                    # Let the module's internal scan handle time:
                    _, pi, _ = networks[a].apply(
                        params,
                        h0_seq,
                        (o_seq, dn_seq, av_seq),
                    )  # pi over (T,bs)

                    logp = pi.log_prob(ac_seq)  # (T,bs)
                    ratio = jnp.exp(logp - olp_seq)  # (T,bs)

                    # Normalize advantages across time and batch.
                    gae_n = (gae_seq - gae_seq.mean()) / (gae_seq.std() + 1e-8)

                    l1 = ratio * gae_n
                    l2 = (
                        jnp.clip(
                            ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
                        )
                        * gae_n
                    )
                    a_loss = -jnp.minimum(l1, l2).mean()  # mean over T and bs
                    ent = pi.entropy().mean()  # mean over T and bs
                    return a_loss - config["ENT_COEF"] * ent, (
                        a_loss,
                        ent,
                        ratio.mean(),
                    )

                def critic_loss_fn(
                    params, o_seq, tgt_seq, vh_seq, av_seq, h0_seq, dn_seq
                ):
                    # o_seq: (T,bs,Do)  tgt/vh: (T,bs)
                    _, _, v = networks[a].apply(
                        params,
                        h0_seq,
                        (o_seq, dn_seq, av_seq),
                    )  # v: (T,bs)

                    v_clip = vh_seq + (v - vh_seq).clip(
                        -config["CLIP_EPS"], config["CLIP_EPS"]
                    )
                    v_loss = (
                        0.5
                        * jnp.maximum(
                            (v - tgt_seq) ** 2, (v_clip - tgt_seq) ** 2
                        ).mean()
                    )
                    return config["VF_COEF"] * v_loss, v_loss

                vg_actor = jax.value_and_grad(actor_loss_fn, has_aux=True)
                vg_critic = jax.value_and_grad(critic_loss_fn, has_aux=True)

                def mb_body(ts, i):
                    # dynamic index per minibatch (mb, T, bs, ...) -> (T,bs,...)
                    o_slice = jax.lax.dynamic_index_in_dim(o_mb, i, keepdims=False)
                    av_slice = jax.lax.dynamic_index_in_dim(av_mb, i, keepdims=False)
                    ac_slice = jax.lax.dynamic_index_in_dim(ac_mb, i, keepdims=False)
                    olp_slice = jax.lax.dynamic_index_in_dim(olp_mb, i, keepdims=False)
                    adv_slice = jax.lax.dynamic_index_in_dim(adv_mb, i, keepdims=False)
                    tgt_slice = jax.lax.dynamic_index_in_dim(tgt_mb, i, keepdims=False)
                    vh_slice = jax.lax.dynamic_index_in_dim(vh_mb, i, keepdims=False)
                    h0_slice = jax.lax.dynamic_index_in_dim(
                        h0_mb, i, keepdims=False
                    )  # (bs,H)

                    (a_tot, (a_l, ent, ratio)), a_grads = vg_actor(
                        ts.params,
                        o_slice,
                        av_slice,
                        ac_slice,
                        olp_slice,
                        adv_slice,
                        h0_slice,
                        dn_mb[i],
                    )
                    (c_tot, c_l), c_grads = vg_critic(
                        ts.params,
                        o_slice,
                        tgt_slice,
                        vh_slice,
                        av_slice,
                        h0_slice,
                        dn_mb[i],
                    )

                    grads_sum = jax.tree.map(lambda x, y: x + y, a_grads, c_grads)
                    ts = ts.apply_gradients(grads=grads_sum)
                    return ts, (a_tot + c_tot, a_l, c_l, ent, ratio)

                def epoch_body(ts, _):
                    ts, stats_mb = jax.lax.scan(mb_body, ts, jnp.arange(mb))
                    # Sum losses over minibatches.
                    return ts, jax.tree.map(lambda x: x.sum(), stats_mb)

                ts, stats_ep = jax.lax.scan(
                    epoch_body, ts, jnp.arange(config["UPDATE_EPOCHS"])
                )
                # sum over epochs as well
                tot_all, act_all, crit_all, ent_all, ratio_all = jax.tree.map(
                    lambda x: x.sum(), stats_ep
                )

                per_agent_loss[a] = {
                    "total_loss": tot_all,
                    "actor_loss": act_all,
                    "critic_loss": crit_all,
                    "entropy": ent_all,
                    "ratio": ratio_all / (mb * config["UPDATE_EPOCHS"]),
                }

                new_train_states[a] = ts

            # Average losses across agents
            loss_info = {
                k: jnp.mean(jnp.array([per_agent_loss[a][k] for a in agents]))
                for k in ["total_loss", "actor_loss", "critic_loss", "entropy", "ratio"]
            }

            metric = traj.info
            metric = jax.tree.map(lambda x: jnp.mean(x), metric)
            metric = {**metric}
            metric["loss"] = loss_info
            metric["update_steps"] = update_idx
            metric["env_step"] = update_idx * T * E

            # Periodic evaluation (actor-only)
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
                        new_train_states,
                        networks,
                        test_env,
                        config,
                        collect_data=False,
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
                new_train_states,
                env_state,
                obs_batch,
                done_batch,
                hstates,
                rng,
            ), metric

        # Initial runner state
        runner_state = (train_states, env_state, obs_batch, done_batch, hstates, _r)

        # Train
        (train_states, env_state, obs_batch, done_batch, hstates, _), metrics = (
            jax.lax.scan(_update, runner_state, jnp.arange(config["NUM_UPDATES"]))
        )
        return {
            "metrics": metrics,
            "runner_state": (train_states,),
        }

    return train
