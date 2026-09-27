"""Train a baseline and save its policy and rollouts."""

import argparse
import importlib
import json
import pickle
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np
from flax import serialization
from omegaconf import OmegaConf

from baselines.collect import as_dataset
from baselines.env_utils import action_dim, make_env, observation_space
from baselines.rollout import agent_states, evaluate

ALGORITHMS = tuple(
    f"{method}_{arch}_nps{suffix}"
    for method in ("ippo", "mappo")
    for arch in ("ff", "rnn")
    for suffix in ("", "_het")
)


def load_config(name, overrides=()):
    """Read a bundled YAML or explicit path and apply KEY=VALUE overrides."""
    path = Path(name)
    if not path.is_file():
        path = Path(__file__).parent / "config" / f"{name}.yaml"
    config = OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(overrides))
    config = OmegaConf.to_container(config, resolve=True)
    config.setdefault("TEST_NUM_ENVS", config["NUM_ENVS"])
    config.setdefault("TEST_NUM_STEPS", config["NUM_STEPS"])
    config.setdefault("SCALE_CLIP_EPS", False)
    for key in (
        "NUM_ENVS",
        "NUM_STEPS",
        "TOTAL_TIMESTEPS",
        "NUM_MINIBATCHES",
        "UPDATE_EPOCHS",
        "TEST_NUM_ENVS",
        "TEST_NUM_STEPS",
    ):
        value = config[key]
        if (
            isinstance(value, bool)
            or int(float(value)) != float(value)
            or int(float(value)) <= 0
        ):
            raise ValueError(f"{key} must be a positive integer, got {value!r}")
        config[key] = int(float(value))
    batch = config["NUM_ENVS"] * config["NUM_STEPS"]
    if config["TOTAL_TIMESTEPS"] < batch:
        raise ValueError(
            "TOTAL_TIMESTEPS must cover at least one NUM_ENVS * NUM_STEPS batch"
        )
    if batch % config["NUM_MINIBATCHES"]:
        raise ValueError("The rollout batch must be divisible by NUM_MINIBATCHES")
    if config.get("ACTIVATION", "relu") not in ("relu", "tanh"):
        raise ValueError("ACTIVATION must be relu or tanh")
    return config


def policy_states(algorithm, output):
    """Extract actor TrainStates from each trainer's returned runner state."""
    runner = output["runner_state"]
    if algorithm == "mappo_ff_nps":
        return runner[0][0][0]
    if algorithm == "mappo_rnn_nps":
        return runner[0][0]
    return runner[0]


def train_and_collect(algorithm, config, output_dir):
    """Train one seed and save its configuration, policy, and rollout dataset.

    Require a new output directory to protect existing runs. Wait for JAX
    computations to finish before writing results.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    recurrent, hetero = "_rnn_" in algorithm, algorithm.endswith("_het")
    if recurrent and config["NUM_ENVS"] % config["NUM_MINIBATCHES"]:
        raise ValueError(
            "Recurrent minibatches require NUM_ENVS divisible by NUM_MINIBATCHES"
        )
    if recurrent and config["FC_DIM_SIZE"] != config["GRU_HIDDEN_DIM"]:
        raise ValueError("These GRUs require FC_DIM_SIZE == GRU_HIDDEN_DIM")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config = dict(config, ALG=algorithm.upper(), NUM_SEEDS=1)
    # Capture requested settings before make_train adds derived batch sizes.
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    module = importlib.import_module(f"baselines.{algorithm}")
    env = make_env(config, use_log_wrapper=True)
    continuous = config.get("CONTINUOUS_ACTIONS", False)
    if continuous and hetero:
        raise ValueError("MaBrax uses the standard trainers with padded agent spaces")
    if any(hasattr(env.action_space(a), "n") == continuous for a in env.agents):
        raise ValueError(
            "CONTINUOUS_ACTIONS must match the environment's action spaces"
        )
    if not hetero:
        shapes = {
            (observation_space(env, a).shape, action_dim(env, a)) for a in env.agents
        }
        if len(shapes) != 1:
            raise ValueError(
                "Unequal agent spaces require the corresponding _het baseline"
            )
    seed = int(config["SEED"])
    train_key = jax.random.PRNGKey(seed)
    eval_key = jax.random.fold_in(train_key, 1)
    print(
        f"Training {algorithm}: seed={seed}, requested environment steps={config['TOTAL_TIMESTEPS']}",
        flush=True,
    )
    with jax.disable_jit(config.get("DISABLE_JIT", False)):
        train = jax.jit(module.make_train(config))
        output = jax.block_until_ready(train(train_key))
        actors = policy_states(algorithm, output)
        evaluate_jit = jax.jit(
            lambda key: evaluate(
                key,
                actors,
                env,
                config,
                recurrent=recurrent,
                independent=algorithm.startswith("ippo"),
                heterogeneous=hetero,
                collect_data=True,
            )
        )
        evaluation = jax.device_get(evaluate_jit(eval_key))
    metrics = jax.device_get(output["metrics"])
    losses = metrics["loss"]
    if not all(np.isfinite(x).all() for x in jax.tree.leaves(losses)):
        raise RuntimeError("Training produced non-finite losses")
    if int(evaluation["completed_episodes"]) == 0:
        raise ValueError("No evaluation episode completed; increase TEST_NUM_STEPS")
    dataset = as_dataset(evaluation, config, algorithm)
    per_agent = dataset["runs"][0]["per_agent"]
    evaluation.pop("eval_data")
    with (output_dir / "rollouts.pkl").open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Plain Flax state dictionaries keep checkpoints independent of optimizer objects.
    expected_updates = (
        config["NUM_UPDATES"] * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
    )
    for state in agent_states(actors, env.agents, hetero).values():
        if int(state.step) != expected_updates:
            raise RuntimeError("Unexpected optimizer update count")
    params = {
        a: state.params for a, state in agent_states(actors, env.agents, hetero).items()
    }
    (output_dir / "policy.msgpack").write_bytes(
        serialization.to_bytes(jax.device_get(params))
    )
    flat_metrics = {}

    def flatten(tree, prefix=""):
        for key, value in tree.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                flatten(value, name + "/")
            else:
                flat_metrics[name] = value

    flatten(metrics)
    np.savez_compressed(output_dir / "training_metrics.npz", **flat_metrics)
    report = dict(
        algorithm=algorithm,
        seed=seed,
        environment_steps=config["NUM_UPDATES"]
        * config["NUM_ENVS"]
        * config["NUM_STEPS"],
        requested_environment_steps=config["TOTAL_TIMESTEPS"],
        optimizer_steps_per_agent=int(
            config["NUM_UPDATES"] * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
        ),
        samples_per_agent={a: len(d["actions"]) for a, d in per_agent.items()},
        evaluation={k: float(v) for k, v in evaluation.items()},
        backend=jax.default_backend(),
        versions={
            name: version(name)
            for name in ("jax", "jaxlib", "jaxmarl", "flax", "optax", "numpy")
            + (("brax", "mujoco", "mujoco-mjx") if continuous else ())
        },
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument(
        "--config", required=True, help="Bundled config name or YAML path"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument(
        "overrides", nargs="*", help="Configuration overrides, e.g. SEED=0"
    )
    args = parser.parse_args()
    train_and_collect(
        args.algorithm, load_config(args.config, args.overrides), args.output
    )


if __name__ == "__main__":
    main()
