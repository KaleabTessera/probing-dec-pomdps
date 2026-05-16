"""Collect rollout artifacts from Weights & Biases for diagnostic analysis.

Downloads artifact bundles produced during evaluation, parses the various
on-disk layouts (stacked-array, per-agent dict, packed npz), and packages
them into a single pickle file consumable by ``compute_metrics``.
"""

from __future__ import annotations

import glob
import logging
import os
import pickle
import tempfile
from collections import defaultdict
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Eval bundle loading
# ----------------------------------------------------------------------


def _unwrap_0d(x):
    if isinstance(x, np.ndarray) and x.ndim == 0:
        inner = x.item()
        if isinstance(inner, dict):
            return inner
    return x


def _load_any_eval_file(path: str) -> Any:
    """Load .npy / .npz / .pkl, decoding the packed ``data/shape/dtype/mask`` form."""
    if path.endswith(".npy"):
        return _unwrap_0d(np.load(path, allow_pickle=True))
    if path.endswith(".npz"):
        f = np.load(path, allow_pickle=True)
        if {"data", "shape", "dtype", "mask"}.issubset(set(f.files)):
            data = f["data"]
            shape = tuple(f["shape"])
            dtype = np.dtype(str(f["dtype"]))
            mask = np.unpackbits(f["mask"]).astype(bool)[: np.prod(shape)].reshape(shape)
            data = data.astype(dtype, copy=False)
            return np.where(mask, np.nan, data)
        agent_keys = [k for k in f.files if k not in ("data", "shape", "dtype", "mask")]
        if agent_keys:
            first_val = _unwrap_0d(f[agent_keys[0]])
            if isinstance(first_val, dict):
                return first_val
            if len(agent_keys) > 1:
                return {k: _unwrap_0d(f[k]) for k in agent_keys}
            return _unwrap_0d(f[agent_keys[0]])
        return _unwrap_0d(f[f.files[0]])
    if path.endswith(".pkl") or path.endswith(".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unknown file extension: {path}")


def _find_and_load(root: str, names: List[str]) -> Optional[Any]:
    cands = []
    for name in names:
        cands += glob.glob(os.path.join(root, f"**/{name}.npz"), recursive=True)
        cands += glob.glob(os.path.join(root, f"**/{name}.npy"), recursive=True)
    if cands:
        return _load_any_eval_file(sorted(cands)[0])
    return None


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    return arr


def _as_1d(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).ravel()


def _is_valid_array(x):
    if x is None or isinstance(x, dict):
        return False
    a = np.asarray(x)
    if a.ndim == 0 or a.dtype == object:
        return False
    return a.size > 0


def _shape_or_keys(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: getattr(v, "shape", type(v).__name__) for k, v in x.items()}
    return getattr(x, "shape", type(x).__name__)


def _group_by_agent(states, actions, hidden, agent_ids, timesteps=None, episode_ids=None):
    """Group stacked-array data into per-agent dicts keyed by ``agent_id``."""
    has_hidden = hidden is not None

    def _flat(arr):
        a = np.asarray(arr)
        if a.ndim == 0:
            return a.reshape(1)
        if a.ndim == 1:
            return a
        return a.reshape(-1, *a.shape[2:]) if a.ndim >= 2 else a

    _ref_arr = np.asarray(agent_ids if agent_ids is not None else states)
    _ref_ax1 = _ref_arr.shape[1] if _ref_arr.ndim >= 2 else None

    # Continuous-action spaces sometimes arrive as (T, n_agents*E*act_dim)
    # rather than (T, n_agents*E, act_dim). Detect and reshape.
    act_arr = np.asarray(actions)
    if (_ref_ax1 is not None and act_arr.ndim == 2
            and act_arr.shape[1] != _ref_ax1
            and act_arr.shape[1] % _ref_ax1 == 0):
        act_dim = act_arr.shape[1] // _ref_ax1
        logger.info(
            f"  [_group_by_agent] reshaping actions {act_arr.shape} -> "
            f"({act_arr.shape[0]}, {_ref_ax1}, {act_dim}) (continuous, act_dim={act_dim})"
        )
        act_arr = act_arr.reshape(act_arr.shape[0], _ref_ax1, act_dim)

    s_flat = _flat(states)
    a_flat = _flat(act_arr)
    h_flat = _flat(hidden) if has_hidden else None
    aid_flat = _flat(agent_ids).ravel() if agent_ids is not None else None
    ts_flat  = _flat(timesteps).ravel() if timesteps is not None else None
    ep_flat  = _flat(episode_ids).ravel() if episode_ids is not None else None

    if aid_flat is not None:
        unique_agents = sorted(np.unique(aid_flat).tolist())
    else:
        unique_agents = [0]
        aid_flat = np.zeros(s_flat.shape[0], dtype=np.int32)

    continuous_actions = a_flat.ndim >= 2 and a_flat.shape[-1] > 1

    result = {}
    for ag in unique_agents:
        mask = (aid_flat == ag)
        s_ag = _as_2d(s_flat[mask])
        a_ag = _as_2d(a_flat[mask]) if continuous_actions else _as_1d(a_flat[mask])
        h_ag = _as_2d(h_flat[mask]) if (has_hidden and h_flat is not None) else \
            np.zeros((s_ag.shape[0], 1), dtype=np.float32)
        ts_ag = ts_flat[mask] if ts_flat is not None else np.arange(s_ag.shape[0])
        ep_ag = ep_flat[mask] if ep_flat is not None else None

        result[str(ag)] = {
            "states": s_ag,
            "actions": a_ag,
            "hidden": h_ag,
            "has_hidden": has_hidden,
            "agent_ids": np.full(s_ag.shape[0], ag),
            "timesteps": ts_ag,
            "episode_ids": ep_ag,
        }
    return result


def _build_per_agent_from_dicts(states, actions, hidden, timesteps, episode_ids):
    """Build per-agent dicts from per-agent dict-format data (different obs spaces per agent)."""
    agents = sorted(states.keys())
    per_agent = {}

    for ag in agents:
        s = np.asarray(states[ag])
        a = np.asarray(actions[ag])

        T_steps = s.shape[0]
        E_envs = s.shape[1] if s.ndim >= 2 else 1
        s_flat = s.reshape(-1, s.shape[-1]) if s.ndim >= 2 else s.reshape(-1, 1)
        a_flat = a.reshape(-1, a.shape[-1]) if a.ndim >= 3 else a.reshape(-1)

        if hidden is not None and isinstance(hidden, dict) and ag in hidden:
            h = np.asarray(hidden[ag])
            h_flat = h.reshape(-1, h.shape[-1]) if h.ndim >= 2 else h.reshape(-1, 1)
            has_hid = True
        else:
            h_flat = np.zeros((s_flat.shape[0], 1), dtype=np.float32)
            has_hid = False

        if timesteps is not None and isinstance(timesteps, dict) and ag in timesteps:
            t = np.asarray(timesteps[ag]).reshape(-1).astype(np.int64)
        else:
            t = np.repeat(np.arange(T_steps, dtype=np.int64), E_envs)

        if episode_ids is not None and isinstance(episode_ids, dict) and ag in episode_ids:
            e = np.asarray(episode_ids[ag]).reshape(-1).astype(np.int64)
        else:
            e = np.tile(np.arange(E_envs, dtype=np.int64), T_steps)

        n = s_flat.shape[0]
        per_agent[ag] = {
            "states": s_flat,
            "actions": a_flat[:n],
            "hidden": h_flat[:n],
            "has_hidden": has_hid,
            "agent_ids": np.full(n, agents.index(ag), dtype=np.int64),
            "timesteps": t[:n],
            "episode_ids": e[:n],
        }

    return per_agent


def load_eval_bundle(root: str) -> Optional[dict]:
    """Load an eval-bundle directory; auto-detect stacked vs per-agent-dict layouts."""
    states = _find_and_load(root, ["states"])
    actions = _find_and_load(root, ["actions"])
    hidden = _find_and_load(root, ["hidden_states", "rnn_hidden", "rnn_hidden_states"])
    agent_ids = _find_and_load(root, ["agent_ids"])
    timesteps = _find_and_load(root, ["timesteps"])
    episode_ids = _find_and_load(root, ["episode_ids"])

    if states is None or actions is None:
        logger.warning(f"  Missing states or actions in {root}")
        return None
    if hidden is None:
        logger.info(f"  No hidden states in {root} (feed-forward model)")

    is_dict_format = isinstance(states, dict) and isinstance(actions, dict)
    logger.info(
        f"  Raw shapes (dict_format={is_dict_format}): "
        f"states={_shape_or_keys(states)}, actions={_shape_or_keys(actions)}, "
        f"hidden={_shape_or_keys(hidden)}, agent_ids={_shape_or_keys(agent_ids)}, "
        f"timesteps={_shape_or_keys(timesteps)}, episode_ids={_shape_or_keys(episode_ids)}"
    )

    if is_dict_format:
        per_agent = _build_per_agent_from_dicts(states, actions, hidden, timesteps, episode_ids)
    else:
        if not _is_valid_array(states) or not _is_valid_array(actions):
            logger.warning(f"  Skipping degenerate artifact in {root} (states/actions empty)")
            return None
        per_agent = _group_by_agent(states, actions, hidden, agent_ids, timesteps, episode_ids)

    return {
        "states": states, "actions": actions, "hidden": hidden,
        "agent_ids": agent_ids, "timesteps": timesteps, "episode_ids": episode_ids,
        "per_agent": per_agent,
    }


# ----------------------------------------------------------------------
# W&B download pipeline
# ----------------------------------------------------------------------


def fetch_matching_runs(api, entity: str, project: str, tag: str, max_runs: Optional[int] = None) -> list:
    project_path = f"{entity}/{project}"
    filters = {"tags": tag} if tag and tag.lower() != "none" else {}
    runs = list(api.runs(project_path, filters=filters))
    logger.info(f"Found {len(runs)} runs with tag='{tag}'")
    if max_runs is not None:
        runs = runs[:max_runs]
        logger.info(f"Limited to first {max_runs} runs")
    return runs


def find_matching_artifact(run, artifact_type: str, artifact_name_substr: str):
    for art in run.logged_artifacts():
        if art.type == artifact_type and artifact_name_substr in art.name:
            return art
    return None


def collect_single_run(run, *, map_key, alg_key, artifact_type, artifact_name,
                       tmp_root: str) -> Optional[dict]:
    run_id = run.id
    run_name = run.name
    config = run.config

    if "ENV_KWARGS" in map_key:
        env_kwargs = config.get("ENV_KWARGS")
        env_key = map_key.split(".")[-1]
        map_name = env_kwargs.get(env_key)
    else:
        map_name = config.get(map_key, "unknown_map")
    alg_name = config.get(alg_key, "unknown_alg")

    logger.info(f"  Run {run_id} ({run_name}): map={map_name}, alg={alg_name}")

    artifact = find_matching_artifact(run, artifact_type, artifact_name)
    if artifact is None:
        logger.warning(f"  No matching artifact for run {run_id}, skipping")
        return None
    try:
        art_dir = artifact.download(root=os.path.join(tmp_root, artifact.name))
    except Exception as e:
        logger.error(f"  Failed to download artifact {artifact.name}: {e}")
        return None

    bundle = load_eval_bundle(art_dir)
    if bundle is None:
        logger.warning(f"  Could not load eval bundle for run {run_id}, skipping")
        return None

    return {
        "run_id": run_id,
        "run_name": run_name,
        "map_name": map_name,
        "alg_name": alg_name,
        "config": dict(config),
        "per_agent": bundle["per_agent"],
    }


def collect_all(*, entity, project, tag, artifact_type, artifact_name,
                map_key, alg_key, max_runs=None, dry_run=False) -> dict:
    """Collect every matching run into the standard pickle dataset format."""
    import wandb

    api = wandb.Api()
    runs = fetch_matching_runs(api, entity, project, tag, max_runs)

    if dry_run:
        logger.info("=== DRY RUN: listing matching runs ===")
        for i, run in enumerate(runs):
            cfg = run.config
            logger.info(
                f"  [{i+1}/{len(runs)}] {run.id} | {run.name} | "
                f"map={cfg.get(map_key, '?')} | alg={cfg.get(alg_key, '?')}"
            )
        return {}

    dataset = {
        "metadata": {
            "entity": entity, "project": project, "tag": tag,
            "artifact_type": artifact_type, "artifact_name": artifact_name,
            "map_key": map_key, "alg_key": alg_key,
            "total_runs_found": len(runs),
        },
        "runs": [],
        "index": defaultdict(list),
    }

    with tempfile.TemporaryDirectory(prefix="wandb_collect_") as tmp_root:
        for i, run in enumerate(runs):
            logger.info(f"Processing run [{i+1}/{len(runs)}]...")
            result = collect_single_run(
                run, map_key=map_key, alg_key=alg_key,
                artifact_type=artifact_type, artifact_name=artifact_name,
                tmp_root=tmp_root,
            )
            if result is not None:
                idx = len(dataset["runs"])
                dataset["runs"].append(result)
                dataset["index"][(result["map_name"], result["alg_name"])].append(idx)

    dataset["index"] = dict(dataset["index"])
    logger.info(f"Collected {len(dataset['runs'])} runs successfully")
    logger.info(f"Index keys (map, alg): {sorted(dataset['index'].keys())}")
    return dataset


def save_dataset(dataset: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Saved dataset to {output_path} ({size_mb:.1f} MB)")
