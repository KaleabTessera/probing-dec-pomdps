"""Environment factories and centralized observations for the paper baselines."""

from functools import partial

import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
from jaxmarl.environments.spaces import Box
from jaxmarl.wrappers.baselines import JaxMARLWrapper


def observation_space(env, agent):
    """Handle both per-agent and shared observation-space APIs in JaxMARL."""
    try:
        return env.observation_space(agent)
    except TypeError:
        return env.observation_space()


def action_dim(env, agent):
    """Number of discrete choices or coordinates of a continuous action."""
    space = env.action_space(agent)
    return space.n if hasattr(space, "n") else int(np.prod(space.shape))


class FlatObservationWrapper(JaxMARLWrapper):
    """Expose vector observations consistently to every actor and collector.

    Flatten before vectorizing environments, so image axes never become batch
    axes. The shared-space API in Overcooked v2 is also exposed per agent here.
    Extra observations such as SMAX's world state retain their original shape.
    """

    def observation_space(self, agent):
        space = observation_space(self._env, agent)
        # Hanabi exposes its vector length as an integer instead of a tuple.
        original_shape = (space.shape,) if np.isscalar(space.shape) else space.shape
        shape = (int(np.prod(original_shape)),)
        return Box(
            np.broadcast_to(space.low, original_shape).reshape(shape),
            np.broadcast_to(space.high, original_shape).reshape(shape),
            shape=shape,
            dtype=space.dtype,
        )

    def _flatten(self, obs):
        return {**obs, **{a: jnp.ravel(obs[a]) for a in self.agents}}

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, state = self._env.reset(key)
        return self._flatten(obs), state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, actions):
        obs, state, rewards, dones, info = self._env.step(key, state, actions)
        return self._flatten(obs), state, rewards, dones, info


class WorldStateWrapper(JaxMARLWrapper):
    """Build centralized critic inputs before vectorizing environments.

    Use SMAX's global state with agent identities, MaBrax's full physical
    observation, or concatenated local observations for other environments.
    """

    def __init__(self, env, smax=False):
        super().__init__(env)
        self.smax = smax

    def _augment(self, obs):
        obs = dict(obs)
        if self.smax:
            state = jnp.broadcast_to(
                obs["world_state"], (self.num_agents, self._env.state_size)
            )
            obs["world_state"] = jnp.concatenate(
                (state, jnp.eye(self.num_agents)), axis=-1
            )
        elif "global" in obs:
            # The critic needs the full physical observation, including coordinates
            # absent from the agents' local views.
            obs["world_state"] = jnp.broadcast_to(
                obs["global"], (self.num_agents, obs["global"].size)
            )
        else:
            state = jnp.concatenate([jnp.ravel(obs[a]) for a in self.agents])
            obs["world_state"] = jnp.broadcast_to(state, (self.num_agents, state.size))
        return obs

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, state = self._env.reset(key)
        return self._augment(obs), state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, actions):
        obs, state, rewards, dones, info = self._env.step(key, state, actions)
        return self._augment(obs), state, rewards, dones, info

    def world_state_size(self):
        if self.smax:
            return self._env.state_size + self.num_agents
        if hasattr(self._env, "world_state_size"):
            return self._env.world_state_size()
        return sum(
            int(np.prod(observation_space(self._env, a).shape)) for a in self.agents
        )


def make_env(config, use_log_wrapper=False, use_state_wrapper=False):
    """Build the paper's MPE, SMAX, Overcooked, Hanabi or MaBrax tasks."""
    from jaxmarl.wrappers.baselines import LogWrapper, MPELogWrapper, SMAXLogWrapper

    name = config["ENV_NAME"]
    kwargs = dict(config.get("ENV_KWARGS", {}))
    smax = config.get("MAP_NAME") is not None
    if smax:
        from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario

        env = HeuristicEnemySMAX(
            scenario=map_name_to_scenario(config["MAP_NAME"]), **kwargs
        )
        wrapper = SMAXLogWrapper
    elif name in (
        "ant_4x2",
        "halfcheetah_6x1",
        "hopper_3x1",
        "humanoid_9|8",
        "walker2d_2x3",
    ):
        from baselines.mabrax.env import MABraxEnv

        env = MABraxEnv(name, **kwargs)
        wrapper = LogWrapper
    elif name.lower().startswith("mpe") or name in (
        "overcooked",
        "overcooked_v2",
        "hanabi",
    ):
        if name == "overcooked" and isinstance(kwargs.get("layout"), str):
            from jaxmarl.environments.overcooked import overcooked_layouts

            kwargs["layout"] = overcooked_layouts[kwargs["layout"]]
        env = jaxmarl.make(name, **kwargs)
        wrapper = MPELogWrapper if name.lower().startswith("mpe") else LogWrapper
    else:
        raise ValueError(f"Unsupported environment: {name}")
    env = FlatObservationWrapper(env)
    if use_state_wrapper:
        env = WorldStateWrapper(env, smax=smax)
    if use_log_wrapper:
        env = wrapper(env, replace_info=True) if smax else wrapper(env)
    return env
