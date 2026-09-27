"""Multi-agent factorisations of the five Brax tasks used in the paper.

The mappings preserve each agent's local joint observations and actuator subset.
JaxMARL handles episode resets; the underlying Brax environment does not reset a
second time. MAPPO receives the complete physical observation under ``global``.
"""

from functools import partial

import jax
import jax.numpy as jnp
from brax import envs
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Box

from baselines.mabrax.mappings import _agent_action_mapping, _agent_observation_mapping


class MABraxEnv(MultiAgentEnv):
    """Local observations and continuous joint actions, optionally padded.

    ``max`` prepends a one-hot agent identity and pads local observations/actions
    to their respective maximum sizes. ``pad`` omits the identity and embeds
    actions in the global action vector. ``concat`` embeds both in global-sized
    vectors. ``None`` preserves unequal spaces.
    """

    def __init__(
        self,
        env_name,
        episode_length=1000,
        action_repeat=1,
        homogenisation_method="max",
        backend="positional",
        **kwargs,
    ):
        if env_name not in _agent_action_mapping:
            raise ValueError(f"Unsupported MaBrax task: {env_name}")
        if homogenisation_method not in (None, "max", "pad", "concat"):
            raise ValueError("homogenisation_method must be null, max, pad or concat")
        # These HalfCheetah indices refer to all nine positions and velocities.
        # Brax otherwise drops root x, silently clipping the final index and
        # shifting every joint observation by one coordinate.
        if env_name == "halfcheetah_6x1":
            kwargs.setdefault("exclude_current_positions_from_observation", False)
        self.env = envs.create(
            env_name.split("_")[0],
            episode_length=episode_length,
            action_repeat=action_repeat,
            auto_reset=False,
            backend=backend,
            **kwargs,
        )
        self.agent_obs_mapping = _agent_observation_mapping[env_name]
        self.agent_action_mapping = _agent_action_mapping[env_name]
        if any(
            int(jnp.max(indices)) >= self.env.observation_size
            for indices in self.agent_obs_mapping.values()
        ):
            raise ValueError(
                "Brax observation size does not match the task's joint mapping"
            )
        self.agents = list(self.agent_action_mapping)
        self.num_agents = len(self.agents)
        self.homogenisation_method = homogenisation_method
        self.max_obs_size = max(self.agent_obs_mapping[a].size for a in self.agents)
        self.max_act_size = max(self.agent_action_mapping[a].size for a in self.agents)
        self.observation_spaces, self.action_spaces = {}, {}
        for agent in self.agents:
            obs_size = self.agent_obs_mapping[agent].size
            act_size = self.agent_action_mapping[agent].size
            if homogenisation_method == "max":
                obs_size = self.num_agents + self.max_obs_size
                act_size = self.max_act_size
            elif homogenisation_method in ("pad", "concat"):
                obs_size = (
                    self.max_obs_size
                    if homogenisation_method == "pad"
                    else self.env.observation_size
                )
                act_size = self.env.action_size
            self.observation_spaces[agent] = Box(-jnp.inf, jnp.inf, (obs_size,))
            # Brax's positional backend receives raw Gaussian torques. These
            # bounds describe actuator controls; policies are not squashed.
            limit = 0.4 if env_name == "humanoid_9|8" else 1.0
            self.action_spaces[agent] = Box(-limit, limit, (act_size,))

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        state = self.env.reset(key)
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=0)
    def step_env(self, key, state, actions):
        state = self.env.step(state, self.map_agents_to_global_action(actions))
        rewards = {a: state.reward for a in self.agents}
        done = state.done.astype(bool)
        dones = {a: done for a in (*self.agents, "__all__")}
        return self.get_obs(state), state, rewards, dones, {}

    def map_agents_to_global_action(self, actions):
        """Discard padding and assemble each agent's controlled actuators."""
        result = jnp.zeros(self.env.action_size)
        for agent, indices in self.agent_action_mapping.items():
            local = actions[agent]
            if self.homogenisation_method == "max":
                local = local[: indices.size]
            elif self.homogenisation_method in ("pad", "concat"):
                local = local[indices]
            result = result.at[indices].set(local)
        return result

    def get_obs(self, state):
        """Extract physical observations without overwriting agent identities."""
        result = {"global": state.obs[self.agent_obs_mapping["global"]]}
        for index, agent in enumerate(self.agents):
            indices = self.agent_obs_mapping[agent]
            local = state.obs[indices]
            if self.homogenisation_method == "max":
                local = jnp.concatenate(
                    (
                        jax.nn.one_hot(index, self.num_agents),
                        jnp.pad(local, (0, self.max_obs_size - local.size)),
                    )
                )
            elif self.homogenisation_method == "pad":
                local = jnp.pad(local, (0, self.max_obs_size - local.size))
            elif self.homogenisation_method == "concat":
                local = jnp.zeros_like(state.obs).at[indices].set(local)
            result[agent] = local
        return result

    def get_avail_actions(self, state):
        """Continuous actors ignore masks; preserve the common trainer contract."""
        return {a: jnp.ones(self.action_spaces[a].shape) for a in self.agents}

    def world_state_size(self):
        return self.agent_obs_mapping["global"].size

    @property
    def sys(self):
        return self.env.sys
