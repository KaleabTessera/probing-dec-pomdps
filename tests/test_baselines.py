"""Regression tests for rollout semantics; optional JAX dependencies required."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("jaxmarl")
jnp = pytest.importorskip("jax.numpy")
from flax import struct

from baselines.collect import as_dataset
from baselines.rollout import evaluate
from baselines.run import ALGORITHMS, load_config
from dec_pomdp_diagnostics.data import unpack_run


@struct.dataclass
class State:
    time: object


class ResetEnv:
    """Two agents sharing a three-step episode, with vectorized resets."""

    agents = ("agent_0", "agent_1")

    def action_space(self, agent):
        return SimpleNamespace(n=2)

    def reset(self, key):
        state = State(jnp.int32(0))
        return {a: jnp.zeros(1) for a in self.agents}, state

    def step(self, key, state, actions):
        ended = state.time == 2
        time = jnp.where(ended, 0, state.time + 1)
        obs = {a: jnp.array([time], dtype=jnp.float32) for a in self.agents}
        done = {a: ended for a in (*self.agents, "__all__")}
        return obs, State(time), {a: jnp.float32(1) for a in self.agents}, done, {}


class Distribution:
    def __init__(self, logits):
        self.logits = logits

    def mode(self):
        return self.logits.argmax(axis=-1)


def recurrent_policy(params, hidden, inputs, return_intermediates=False):
    """A simple stateful policy exposes any extra forward pass or missed reset."""
    obs, resets, available = inputs
    hidden = jnp.where(resets[0, :, None], 0, hidden)
    next_hidden = 2 * hidden + obs[0] + 1
    logits = jnp.concatenate((-next_hidden, next_hidden), axis=-1)[None]
    return (
        next_hidden,
        Distribution(logits),
        jnp.zeros(obs.shape[:2]),
        logits,
        next_hidden[None],
    )


def test_collection_resets_memory_and_preserves_episode_alignment():
    config = dict(
        TEST_NUM_ENVS=2, TEST_NUM_STEPS=8, GRU_HIDDEN_DIM=1, SEED=0, ENV_NAME="test"
    )
    states = {
        a: SimpleNamespace(params={}, apply_fn=recurrent_policy)
        for a in ResetEnv.agents
    }
    result = jax.device_get(
        jax.jit(
            lambda key: evaluate(
                key,
                states,
                ResetEnv(),
                config,
                recurrent=True,
                independent=True,
                heterogeneous=True,
                collect_data=True,
            )
        )(jax.random.PRNGKey(0))
    )
    assert result["completed_episodes"] == 4
    assert result["returned_episode_returns"] == 3
    expected = np.array([1, 4, 11, 1, 4, 11, 1, 4])
    for fields in result["eval_data"].values():
        np.testing.assert_array_equal(fields["hidden"][:, 0, 0], expected)
        np.testing.assert_array_equal(
            fields["timesteps"][:, 0], [0, 1, 2, 0, 1, 2, 0, 1]
        )
        np.testing.assert_array_equal(
            fields["episode_ids"][:, 0], [0, 0, 0, 2, 2, 2, 4, 4]
        )
        np.testing.assert_array_equal(
            fields["episode_ids"][:, 1], [1, 1, 1, 3, 3, 3, 5, 5]
        )
    dataset = as_dataset(result, config, "ippo_rnn_nps_het")
    obs, actions, hidden, times, episodes, has_hidden = unpack_run(dataset["runs"][0])
    assert has_hidden
    assert obs["agent_0"].shape == (16, 1)
    assert hidden["agent_0"].shape == (16, 1)
    np.testing.assert_array_equal(times["agent_0"], times["agent_1"])
    np.testing.assert_array_equal(episodes["agent_0"], episodes["agent_1"])
    np.testing.assert_array_equal(actions["agent_0"], np.ones(16))


@pytest.mark.parametrize(
    "config_path",
    sorted((Path(__file__).parents[1] / "baselines/config").glob("*.yaml")),
)
def test_bundled_configs_load(config_path):
    config = load_config(str(config_path))
    assert config["TOTAL_TIMESTEPS"] >= config["NUM_STEPS"] * config["NUM_ENVS"]
    assert config["TEST_NUM_ENVS"] > 0
    assert config["ENV_NAME"]


@pytest.mark.parametrize(
    "override",
    ["NUM_ENVS=0", "NUM_STEPS=1.5", "TOTAL_TIMESTEPS=1", "NUM_MINIBATCHES=3"],
)
def test_invalid_batches_are_rejected(override):
    with pytest.raises(ValueError):
        load_config("ippo_ff_mpe", [override])


@pytest.mark.parametrize(
    "algorithm",
    ["ippo_rnn_nps", "ippo_rnn_nps_het", "mappo_rnn_nps", "mappo_rnn_nps_het"],
)
def test_recurrent_ppo_replays_behavior_policy_across_batch_and_episode_boundaries(
    algorithm,
):
    """With LR=0, PPO likelihood ratios must stay one even after the first batch.

    Sixteen-step rollouts cut across the 25-step MPE horizon. This catches zero
    replay carries, wrong reset masks, and broadcasting with one env/minibatch.
    """
    import importlib

    name = "_".join(algorithm.split("_")[:2]) + "_mpe"
    config = load_config(
        name,
        [
            "LR=0",
            "NUM_ENVS=2",
            "NUM_STEPS=16",
            "TOTAL_TIMESTEPS=128",
            "NUM_MINIBATCHES=2",
            "UPDATE_EPOCHS=1",
            "FC_DIM_SIZE=8",
            "GRU_HIDDEN_DIM=8",
            "TEST_NUM_ENVS=1",
            "TEST_NUM_STEPS=26",
            "ENV_NAME=MPE_simple_speaker_listener_v4"
            if algorithm.endswith("_het")
            else "ENV_NAME=MPE_simple_spread_v3",
        ],
    )
    module = importlib.import_module(f"baselines.{algorithm}")
    result = jax.device_get(jax.jit(module.make_train(config))(jax.random.PRNGKey(2)))
    ratio = result["metrics"]["loss"]["ratio"]
    # The heterogeneous MAPPO logger sums its per-minibatch loss statistics.
    if algorithm == "mappo_rnn_nps_het":
        ratio = ratio / config["NUM_MINIBATCHES"]
    np.testing.assert_allclose(ratio, 1.0, atol=1e-6)
    for value in jax.tree.leaves(result["metrics"]["loss"]):
        assert np.isfinite(value).all()
    jax.clear_caches()


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_all_trainers_accept_overcooked_observations(algorithm, version):
    """Trace rollout and optimization with both image observation-space APIs."""
    import importlib

    config = load_config(
        f"ippo_rnn_overcooked_{version}",
        [
            "NUM_ENVS=2",
            "NUM_STEPS=2",
            "TOTAL_TIMESTEPS=4",
            "NUM_MINIBATCHES=1",
            "UPDATE_EPOCHS=1",
            "FC_DIM_SIZE=8",
            "GRU_HIDDEN_DIM=8",
            "TEST_NUM_ENVS=1",
            "TEST_NUM_STEPS=2",
        ],
    )
    module = importlib.import_module(f"baselines.{algorithm}")
    result = jax.eval_shape(module.make_train(config), jax.random.PRNGKey(0))
    assert result["metrics"]["loss"]
    jax.clear_caches()


def test_experiment_recipes_preserve_distinct_map_settings():
    """A recipe must not silently overwrite another experiment's hyperparameters."""
    import json

    path = Path(__file__).parents[1] / "baselines/experiments.json"
    recipes = json.loads(path.read_text())
    assert len({r["name"] for r in recipes}) == len(recipes)
    for recipe in recipes:
        overrides = recipe["overrides"]
        keys = [item.split("=", 1)[0] for item in overrides]
        assert len(keys) == len(set(keys)), recipe["name"]
        assert recipe["algorithm"] in ALGORITHMS
        # Resolve one sweep member to exercise the documented loading path.
        config = load_config(recipe["config"], [x.split(",", 1)[0] for x in overrides])
        assert config["ENV_NAME"]


def test_recurrent_mappo_accepts_unmasked_actions():
    from baselines.mappo_rnn_nps_het import ActorRNN

    network = ActorRNN(3, {"FC_DIM_SIZE": 8, "GRU_HIDDEN_DIM": 8})
    hidden = jnp.zeros((2, 8))
    inputs = (jnp.zeros((1, 2, 4)), jnp.zeros((1, 2), dtype=bool), None)
    params = network.init(jax.random.PRNGKey(0), hidden, inputs)
    _, policy = network.apply(params, hidden, inputs)
    assert policy.logits.shape == (1, 2, 3)
    assert np.isfinite(policy.logits).all()


@pytest.mark.parametrize("method", ["ippo", "mappo"])
@pytest.mark.parametrize("arch", ["ff", "rnn"])
def test_gaussian_policy_preserves_singleton_batch_and_action_axes(method, arch):
    """One actuator still has a vector action and a scalar joint log density."""
    import importlib

    module = importlib.import_module(f"baselines.{method}_{arch}_nps")
    cls = getattr(
        module,
        ("ActorCritic" if method == "ippo" else "Actor")
        + ("RNN" if arch == "rnn" else ("" if method == "ippo" else "FF")),
    )
    config = dict(FC_DIM_SIZE=8, GRU_HIDDEN_DIM=8, CONTINUOUS_ACTIONS=True)
    network = cls(1, config=config)
    if arch == "rnn":
        inputs = (
            jnp.zeros((1, 8)),
            (jnp.zeros((2, 1, 4)), jnp.zeros((2, 1), dtype=bool), jnp.ones((2, 1, 1))),
        )
        expected_shape = (2, 1, 1)
    else:
        obs, mask = jnp.zeros((1, 4)), jnp.ones((1, 1))
        inputs = (obs, mask) if method == "ippo" else ((obs, mask),)
        expected_shape = (1, 1)
    params = network.init(jax.random.PRNGKey(0), *inputs)
    result = network.apply(params, *inputs)
    policy = result[1] if arch == "rnn" else result[0] if method == "ippo" else result
    actions = policy.sample(seed=jax.random.PRNGKey(1))
    assert actions.shape == expected_shape
    assert policy.log_prob(actions).shape == expected_shape[:-1]
    assert np.isfinite(policy.log_prob(actions)).all()


def test_mabrax_critic_receives_complete_physical_observation():
    from baselines.env_utils import WorldStateWrapper

    env = SimpleNamespace(num_agents=2, agents=("agent_0", "agent_1"))
    obs = {
        "agent_0": jnp.array([1.0]),
        "agent_1": jnp.array([2.0]),
        "global": jnp.array([1.0, 2.0, 3.0]),
    }
    result = WorldStateWrapper(env)._augment(obs)
    np.testing.assert_array_equal(result["world_state"], [[1, 2, 3], [1, 2, 3]])


def test_halfcheetah_mapping_uses_full_observation_and_preserves_agent_ids():
    from baselines.mabrax.env import MABraxEnv

    env = MABraxEnv("halfcheetah_6x1", homogenisation_method="max")
    assert env.env.observation_size == 18
    state = SimpleNamespace(obs=jnp.arange(18, dtype=jnp.float32))
    observations = env.get_obs(state)
    for index, agent in enumerate(env.agents):
        np.testing.assert_array_equal(
            observations[agent][: env.num_agents], jax.nn.one_hot(index, env.num_agents)
        )
        indices = env.agent_obs_mapping[agent]
        np.testing.assert_array_equal(
            observations[agent][env.num_agents : env.num_agents + indices.size],
            state.obs[indices],
        )
