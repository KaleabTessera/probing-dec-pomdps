"""Greedy evaluation with observations, actions and memory from one forward pass.

Arrays remain (time, parallel environment, feature) until export. Episode IDs
are unique across resets and environments, so metric histories cannot cross an
episode boundary. Recurrent ``hidden`` is the post-observation GRU output used
to choose the recorded action.
"""

import jax
import jax.numpy as jnp

from baselines.env_utils import action_dim


def agent_states(states, agents, heterogeneous):
    """Split a vmapped TrainState into independent agent states, if necessary."""
    if heterogeneous:
        return states
    return {a: jax.tree.map(lambda x: x[i], states) for i, a in enumerate(agents)}


def evaluate(
    rng,
    states,
    env,
    config,
    *,
    recurrent,
    independent,
    heterogeneous,
    collect_data=False,
):
    """Run fixed-length evaluation; include completed-episode returns only.

    ``independent`` selects IPPO's joint actor/critic network signature. MAPPO
    evaluates only its actor, with local observations, never the global state.
    Each environment auto-resets through JaxMARL's step implementation.
    """
    agents = tuple(env.agents)
    states = agent_states(states, agents, heterogeneous)
    count, steps = int(config["TEST_NUM_ENVS"]), int(config["TEST_NUM_STEPS"])
    rng, reset_key = jax.random.split(rng)
    obs, env_state = jax.vmap(env.reset)(jax.random.split(reset_key, count))
    hidden = {a: jnp.zeros((count, config.get("GRU_HIDDEN_DIM", 1))) for a in agents}
    dones = {a: jnp.zeros(count, dtype=bool) for a in agents}
    time = jnp.zeros(count, dtype=jnp.int32)
    episodes = jnp.arange(count)
    returns = jnp.zeros((count, len(agents)))

    def step(carry, _):
        obs, env_state, hidden, dones, time, episodes, returns, rng = carry
        try:
            available = jax.vmap(env.get_avail_actions)(env_state.env_state)
        except (AttributeError, NotImplementedError):
            available = {a: jnp.ones((count, action_dim(env, a))) for a in agents}
        actions, next_hidden, records = {}, {}, {}
        for a in agents:
            local_obs = obs[a].reshape(count, -1)
            if recurrent:
                result = states[a].apply_fn(
                    states[a].params,
                    hidden[a],
                    (local_obs[None], dones[a][None], available[a][None]),
                    return_intermediates=True,
                )
                next_hidden[a], pi = result[:2]
                action = pi.mode()[0]
                logits, memory = result[-2][0], result[-1][0]
            elif independent:
                pi, _, logits = states[a].apply_fn(
                    states[a].params, local_obs, available[a], return_intermediates=True
                )
                action, memory = pi.mode(), jnp.zeros((count, 1))
                next_hidden[a] = hidden[a]
            else:
                pi, logits, _ = states[a].apply_fn(
                    states[a].params,
                    (local_obs, available[a]),
                    return_intermediates=True,
                )
                action, memory = pi.mode(), jnp.zeros((count, 1))
                next_hidden[a] = hidden[a]
            actions[a] = action
            if collect_data:
                records[a] = dict(
                    states=local_obs,
                    actions=action,
                    hidden=memory,
                    timesteps=time,
                    episode_ids=episodes,
                )
                # Preserve action coordinates for continuous information estimators.
                policy_key = (
                    "action_mean"
                    if config.get("CONTINUOUS_ACTIONS", False)
                    else "pre_softmax_logits"
                )
                records[a][policy_key] = logits
                if not config.get("CONTINUOUS_ACTIONS", False):
                    records[a]["available_actions"] = available[a]
                elif hasattr(env, "agent_action_mapping"):
                    # Export only the actuator coordinates this agent controls.
                    indices = env.agent_action_mapping[a]
                    method = env.homogenisation_method
                    if method == "max":
                        indices = jnp.arange(indices.size)
                    elif method not in ("pad", "concat"):
                        indices = jnp.arange(action.shape[-1])
                    records[a]["actions"] = action[..., indices]
                    records[a][policy_key] = logits[..., indices]
        rng, key = jax.random.split(rng)
        next_obs, env_state, reward, done, _ = jax.vmap(env.step)(
            jax.random.split(key, count), env_state, actions
        )
        returns = returns + jnp.stack([reward[a] for a in agents], axis=-1)
        ended = done["__all__"]
        completed = jnp.where(ended[:, None], returns, jnp.nan)
        if collect_data:
            for a in agents:
                records[a]["rewards"] = reward[a]
                records[a]["dones"] = ended
        carry = (
            next_obs,
            env_state,
            next_hidden,
            {a: done[a] for a in agents},
            jnp.where(ended, 0, time + 1),
            episodes + ended * count,
            jnp.where(ended[:, None], 0.0, returns),
            rng,
        )
        return carry, (completed, records)

    initial = (obs, env_state, hidden, dones, time, episodes, returns, rng)
    _, (completed, records) = jax.lax.scan(step, initial, None, length=steps)
    metrics = {
        "returned_episode_returns": jnp.nanmean(completed),
        "completed_episodes": jnp.sum(jnp.isfinite(completed[..., 0])).astype(
            jnp.float32
        ),
    }
    if collect_data:
        metrics["eval_data"] = records
    return metrics
