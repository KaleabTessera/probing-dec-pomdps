"""Restore a saved policy and collect fresh rollouts."""

import argparse
import importlib
import json
import pickle
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import jax
from flax import serialization

from baselines.env_utils import action_dim, make_env
from baselines.rollout import evaluate


def as_dataset(evaluation, config, algorithm):
    """Flatten time/env axes without padding heterogeneous observations."""
    recurrent = "_rnn_" in algorithm
    per_agent = {}
    for agent, fields in evaluation["eval_data"].items():
        per_agent[agent] = {
            key: value.reshape((-1,) + value.shape[2:]) for key, value in fields.items()
        }
        per_agent[agent]["has_hidden"] = recurrent
    seed = config["SEED"]
    name = f"{algorithm}_seed{seed}"
    scenario = (
        config.get("MAP_NAME")
        or config.get("ENV_KWARGS", {}).get("layout")
        or config["ENV_NAME"]
    )
    run = dict(
        run_id=name,
        run_name=name,
        map_name=scenario,
        alg_name=algorithm.upper(),
        config=config,
        per_agent=per_agent,
    )
    return {"runs": [run]}


def restore_and_collect(run_dir, *, seed=None, num_envs=None, num_steps=None):
    """Load actor parameters and evaluate with the recorded settings.

    Defaults reproduce training-time evaluation. ``seed`` selects a fresh
    evaluation stream without changing the recorded training seed.
    """
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    algorithm = config["ALG"].lower()
    if num_envs is not None:
        config["TEST_NUM_ENVS"] = num_envs
    if num_steps is not None:
        config["TEST_NUM_STEPS"] = num_steps
    if min(config["TEST_NUM_ENVS"], config["TEST_NUM_STEPS"]) <= 0:
        raise ValueError("Evaluation environment and step counts must be positive")
    eval_seed = config["SEED"] if seed is None else seed
    config["EVAL_SEED"] = eval_seed
    module = importlib.import_module(f"baselines.{algorithm}")
    recurrent, independent = "_rnn_" in algorithm, algorithm.startswith("ippo")
    cls = getattr(
        module,
        ("ActorCriticRNN" if recurrent else "ActorCritic")
        if independent
        else ("ActorRNN" if recurrent else "ActorFF"),
    )
    env = make_env(config, use_log_wrapper=True)
    params = serialization.msgpack_restore((run_dir / "policy.msgpack").read_bytes())
    states = {
        a: SimpleNamespace(
            params=jax.device_put(params[a]),
            apply_fn=cls(action_dim(env, a), config=config).apply,
        )
        for a in env.agents
    }
    eval_fn = jax.jit(
        partial(
            evaluate,
            states=states,
            env=env,
            config=config,
            recurrent=recurrent,
            independent=independent,
            heterogeneous=True,
            collect_data=True,
        )
    )
    key = jax.random.fold_in(jax.random.PRNGKey(eval_seed), 1)
    result = jax.device_get(eval_fn(key))
    return as_dataset(result, config, algorithm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, required=True, help="Directory from baselines.run"
    )
    parser.add_argument("--output", type=Path, required=True, help="New dataset pickle")
    parser.add_argument(
        "--seed", type=int, help="Evaluation seed (default: training seed)"
    )
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--num-steps", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    dataset = restore_and_collect(
        args.run, seed=args.seed, num_envs=args.num_envs, num_steps=args.num_steps
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
