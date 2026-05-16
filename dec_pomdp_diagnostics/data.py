"""Dataset loading, packed-format unpacking, and per-agent alignment.

The collected pickle files store rollouts produced by JAX-based MARL training.
For storage efficiency they may be in a packed ``{data, dtype, mask, shape}``
form. This module reconstructs them into the per-agent dicts that the metric
functions consume.

Shape conventions for the unpacked per-agent dicts:

* ``Sd[agent_id]``: ``(N, obs_dim)`` observations
* ``Ad[agent_id]``: ``(N,)`` int actions or ``(N, act_dim)`` continuous actions
* ``Hd[agent_id]``: ``(N, hidden_dim)`` RNN hidden states (or zeros if FF)
* ``Td[agent_id]``: ``(N,)`` timestep within episode
* ``Ed[agent_id]``: ``(N,)`` parallel-env / episode index
"""

from __future__ import annotations

import logging
import pickle

import numpy as np

from .estimators import _ensure_1d_int, _ensure_2d, _is_discrete

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Action-shape detection / packing helpers
# ----------------------------------------------------------------------


def _actions_are_continuous(A):
    A = np.asarray(A)
    if A.ndim >= 2 and A.shape[-1] > 1:
        return True
    if A.ndim >= 2 and A.shape[-1] == 1:
        A = A.ravel()
    if not np.issubdtype(A.dtype, np.integer):
        return not _is_discrete(A)
    return False


def _collect_action(A, row):
    A = np.asarray(A)
    if A.ndim >= 2:
        return A[row]
    return int(A[row])


def _stack_actions(A_list):
    if len(A_list) == 0:
        return np.zeros(0), False
    first = A_list[0]
    if isinstance(first, np.ndarray) and first.ndim >= 1:
        return np.stack(A_list), True
    return np.asarray(A_list, dtype=np.int64), False


def _action_window(indices, actions, n_classes=None):
    """Flat feature vector for a window of actions: one-hot if discrete, raw if continuous.

    Pass ``n_classes`` from a per-agent precompute when calling inside a hot
    loop — otherwise this function rescans ``actions`` (O(N)) on every call.
    """
    if not indices:
        return np.zeros((0,), dtype=float)
    idx_arr = np.asarray(indices, dtype=int)
    a_seq = actions[idx_arr]
    if _actions_are_continuous(actions):
        return np.asarray(a_seq, dtype=float).reshape(-1)
    a_int = _ensure_1d_int(a_seq)
    if n_classes is None:
        n_classes = max(int(np.max(_ensure_1d_int(actions))) + 1, 2)
    return np.eye(n_classes, dtype=float)[a_int].reshape(-1)


# ----------------------------------------------------------------------
# Episode / timestep alignment
# ----------------------------------------------------------------------


def _derive_episode_ids(T, E):
    """Infer episode IDs by detecting timestep non-increases per env."""
    T = np.asarray(T, dtype=np.int64)
    E = np.asarray(E, dtype=np.int64)
    keys = list(zip(E.tolist(), T.tolist()))
    if len(keys) == len(set(keys)):
        return None
    last_t, ep_ctr = {}, {}
    ep = np.empty_like(T)
    for i, (ti, ei) in enumerate(zip(T, E)):
        ei, ti = int(ei), int(ti)
        if ei not in last_t:
            last_t[ei] = ti
            ep_ctr[ei] = 0
        elif ti <= last_t[ei]:
            ep_ctr[ei] += 1
            last_t[ei] = ti
        else:
            last_t[ei] = ti
        ep[i] = ep_ctr[ei]
    return ep


def _build_index(T, E):
    """Map ``(episode?, env, time) -> row index`` for fast cross-agent lookup."""
    T = np.asarray(T, dtype=np.int64)
    E = np.asarray(E, dtype=np.int64)
    ep = _derive_episode_ids(T, E)
    if ep is None:
        return {(int(E[i]), int(T[i])): i for i in range(len(T))}, None
    return {(int(ep[i]), int(E[i]), int(T[i])): i for i in range(len(T))}, ep


def _align_pair(Ti, Ei, Tj, Ej, *, map_i=None, epj=None):
    """Return matched index arrays for two agents that share (episode, env, time) keys.

    Pass precomputed ``map_i`` and ``epj`` (from ``_build_index``) to avoid
    rebuilding the per-agent indexes inside a double agent loop.
    """
    if map_i is None:
        map_i, _ = _build_index(Ti, Ei)
    if epj is None:
        _, epj = _build_index(Tj, Ej)

    Tj = np.asarray(Tj, dtype=np.int64)
    Ej = np.asarray(Ej, dtype=np.int64)

    keys_j = []
    for idx in range(len(Tj)):
        t, e = int(Tj[idx]), int(Ej[idx])
        keys_j.append((e, t) if epj is None else (int(epj[idx]), e, t))

    idx_i, idx_j = [], []
    for jj, kj in enumerate(keys_j):
        if kj in map_i:
            idx_i.append(map_i[kj])
            idx_j.append(jj)
    return np.asarray(idx_i, dtype=int), np.asarray(idx_j, dtype=int)


def subsample_agents(Sd, Ad, Hd, Td, Ed, max_samples, rng_seed=0):
    """Subsample to at most ``max_samples`` per agent, jointly across agents.

    Uses the same (episode, timestep) indices for every agent so that
    cross-agent alignment (needed for PIF/DAI/AA) is preserved. Stratified by
    episode so temporal structure is maintained for history-window metrics.
    """
    rng = np.random.default_rng(rng_seed if rng_seed else 0)

    ref_ak = next(iter(Sd))
    n = len(Sd[ref_ak])
    if n <= max_samples:
        logger.info(f"    Subsampling skipped: {n} <= {max_samples}")
        return Sd, Ad, Hd, Td, Ed

    eps_arr = Ed[ref_ak]
    unique_eps = np.unique(eps_arr)
    n_eps = len(unique_eps)
    per_ep = max(1, max_samples // n_eps)

    indices = []
    for ep in unique_eps:
        ep_idx = np.where(eps_arr == ep)[0]
        if len(ep_idx) <= per_ep:
            indices.append(ep_idx)
        else:
            chosen = rng.choice(ep_idx, size=per_ep, replace=False)
            indices.append(np.sort(chosen))
    indices = np.concatenate(indices)

    if len(indices) > max_samples:
        indices = rng.choice(indices, size=max_samples, replace=False)
        indices = np.sort(indices)

    ref_E = Ed[ref_ak]
    ref_T = Td[ref_ak]

    Sd2, Ad2, Hd2, Td2, Ed2 = {}, {}, {}, {}, {}
    for ak in Sd:
        if ak == ref_ak:
            ak_idx = indices
        elif (len(Ed[ak]) == len(ref_E)
              and np.array_equal(Ed[ak], ref_E)
              and np.array_equal(Td[ak], ref_T)):
            ak_idx = indices
        else:
            ref_keys = set(zip(ref_E[indices].tolist(), ref_T[indices].tolist()))
            ak_mask = np.array(
                [(int(Ed[ak][r]), int(Td[ak][r])) in ref_keys for r in range(len(Sd[ak]))],
                dtype=bool,
            )
            ak_idx = np.where(ak_mask)[0]

        Sd2[ak] = Sd[ak][ak_idx]
        Ad2[ak] = Ad[ak][ak_idx]
        Hd2[ak] = Hd[ak][ak_idx]
        Td2[ak] = Td[ak][ak_idx]
        Ed2[ak] = Ed[ak][ak_idx]

    first_ak = next(iter(Sd))
    logger.info(f"    Subsampled: {len(Sd[first_ak])} -> {len(Sd2[first_ak])} samples/agent (joint)")
    return Sd2, Ad2, Hd2, Td2, Ed2


def permute_actions(Ad, rng):
    """Permute each agent's action sequence independently (preserves marginals)."""
    Ad_null = {}
    for a, A in Ad.items():
        A = np.asarray(A)
        perm = rng.permutation(len(A))
        Ad_null[a] = A[perm]
    return Ad_null


# ----------------------------------------------------------------------
# Packed-format loading
# ----------------------------------------------------------------------


def _extract_array(field):
    """Reconstruct an ndarray from a possibly-packed ``{data, dtype, mask, shape}`` dict."""
    if isinstance(field, dict) and "data" in field:
        data = np.asarray(field["data"])
        if {"shape", "dtype", "mask"}.issubset(field.keys()):
            shape = tuple(field["shape"])
            dtype = np.dtype(str(field["dtype"]))
            mask_raw = np.asarray(field["mask"])
            mask = np.unpackbits(mask_raw).astype(bool)[: np.prod(shape)].reshape(shape)
            data = data.astype(dtype, copy=False)
            return np.where(mask, np.nan, data)
        return data
    if hasattr(field, "files"):
        if {"data", "shape", "dtype", "mask"}.issubset(set(field.files)):
            data = field["data"]
            shape = tuple(field["shape"])
            dtype = np.dtype(str(field["dtype"]))
            mask = np.unpackbits(field["mask"]).astype(bool)[: np.prod(shape)].reshape(shape)
            data = data.astype(dtype, copy=False)
            return np.where(mask, np.nan, data)
        for k in field.files:
            if k not in ("data", "shape", "dtype", "mask"):
                return field[k]
        return field[field.files[0]]
    return np.asarray(field)


def _is_packed_format(per_agent):
    if not isinstance(per_agent, dict):
        return False
    return ({"data", "shape", "dtype", "mask"}.issubset(per_agent.keys())
            and isinstance(per_agent.get("data"), dict))


def _reconstruct_packed_field(data, dtype_info, mask_info, shape_info):
    data = np.asarray(data)

    target_dtype = data.dtype
    if dtype_info is not None and not isinstance(dtype_info, (bool, int, float)):
        dtype_arr = np.asarray(dtype_info)
        if dtype_arr.dtype.kind in ("U", "S", "O"):
            try:
                target_dtype = np.dtype(str(dtype_arr.flat[0]))
            except (TypeError, IndexError):
                pass

    target_shape = data.shape
    if shape_info is not None and not isinstance(shape_info, (bool, int, float)):
        shape_arr = np.asarray(shape_info)
        if shape_arr.dtype.kind in ("i", "u"):
            target_shape = tuple(int(x) for x in shape_arr.flatten())

    try:
        data = data.reshape(target_shape)
    except ValueError:
        pass
    try:
        data = data.astype(target_dtype, copy=False)
    except (TypeError, ValueError):
        pass

    if mask_info is not None and not isinstance(mask_info, (bool, int, float)):
        mask_arr = np.asarray(mask_info)
        if mask_arr.dtype == np.uint8:
            total = int(np.prod(target_shape))
            unpacked = np.unpackbits(mask_arr.flatten()).astype(bool)[:total]
            try:
                mask = unpacked.reshape(target_shape)
                if np.any(mask) and np.issubdtype(target_dtype, np.floating):
                    data = np.where(mask, np.nan, data)
            except ValueError:
                pass
    return data


def _unpack_run_packed(per_agent):
    """Unpack ``per_agent`` from packed ``{data, dtype, mask, shape}`` dicts."""
    raw_data = per_agent["data"]
    raw_dtype = per_agent.get("dtype", {})
    raw_mask = per_agent.get("mask", {})
    raw_shape = per_agent.get("shape", {})

    fields = {}
    has_hidden = False
    for field_name, val in raw_data.items():
        if isinstance(val, (bool, int, float, str, type(None))):
            if field_name == "has_hidden":
                has_hidden = bool(val)
            fields[field_name] = val
            continue
        arr = np.asarray(val)
        if arr.ndim == 0:
            if field_name == "has_hidden":
                has_hidden = bool(arr.item())
            fields[field_name] = arr.item()
            continue

        d_info = raw_dtype.get(field_name) if isinstance(raw_dtype, dict) else None
        m_info = raw_mask.get(field_name) if isinstance(raw_mask, dict) else None
        s_info = raw_shape.get(field_name) if isinstance(raw_shape, dict) else None

        if isinstance(d_info, (bool, int, float)):
            d_info = None
        if isinstance(m_info, (bool, int, float)):
            m_info = None
        if isinstance(s_info, (bool, int, float)):
            s_info = None

        fields[field_name] = _reconstruct_packed_field(arr, d_info, m_info, s_info)

    states = fields.get("states")
    actions = fields.get("actions")
    hidden = fields.get("hidden")
    agent_ids = fields.get("agent_ids")
    timesteps = fields.get("timesteps")
    episode_ids = fields.get("episode_ids")

    # Flatten leading dimensions (jax.lax.scan creates (T, agents*envs, ...) shape)
    if states is not None and states.ndim > 2:
        states = states.reshape(-1, states.shape[-1])
    if actions is not None:
        actions = np.asarray(actions)
        if actions.ndim > 2:
            actions = actions.reshape(-1, actions.shape[-1])
        elif actions.ndim == 2:
            if agent_ids is not None:
                aid_flat_count = np.asarray(agent_ids).flatten().size
                if actions.shape[0] * actions.shape[1] == aid_flat_count:
                    actions = actions.flatten()
            else:
                actions = actions.flatten()
        else:
            actions = actions.flatten()
    if hidden is not None and hidden.ndim > 2:
        hidden = hidden.reshape(-1, hidden.shape[-1])
    if agent_ids is not None:
        agent_ids = np.asarray(agent_ids).flatten().astype(np.int64)
    if timesteps is not None:
        timesteps = np.asarray(timesteps).flatten().astype(np.int64)
    if episode_ids is not None:
        episode_ids = np.asarray(episode_ids).flatten().astype(np.int64)

    _cont_a = (actions is not None and actions.ndim >= 2 and actions.shape[-1] > 1
               and not np.issubdtype(actions.dtype, np.integer))

    Sd, Ad, Hd, Td, Ed = {}, {}, {}, {}, {}
    if agent_ids is None:
        raise ValueError("Packed format requires 'agent_ids' field to split data by agent.")

    for uid in np.unique(agent_ids):
        m = agent_ids == uid
        ag = str(int(uid))
        n = int(m.sum())
        Sd[ag] = _ensure_2d(states[m]) if states is not None else np.zeros((n, 1))
        if actions is not None:
            Ad[ag] = _ensure_2d(actions[m]) if _cont_a else _ensure_1d_int(actions[m])
        else:
            Ad[ag] = np.zeros(n, dtype=np.int64)
        if hidden is not None and has_hidden:
            Hd[ag] = _ensure_2d(hidden[m])
        else:
            Hd[ag] = np.zeros((n, 1), dtype=np.float32)
        Td[ag] = timesteps[m] if timesteps is not None else np.arange(n, dtype=np.int64)
        Ed[ag] = episode_ids[m] if episode_ids is not None else np.zeros(n, dtype=np.int64)

    return Sd, Ad, Hd, Td, Ed, has_hidden


# ----------------------------------------------------------------------
# Top-level unpack: dispatches between packed and per-agent dict formats
# ----------------------------------------------------------------------


def unpack_run(run_data):
    """Convert a single collected run into ``(Sd, Ad, Hd, Td, Ed, has_hidden)``."""
    per_agent = run_data["per_agent"]

    if _is_packed_format(per_agent):
        return _unpack_run_packed(per_agent)

    Sd, Ad, Hd, Td, Ed = {}, {}, {}, {}, {}
    any_real_hidden = False

    for ag, d in per_agent.items():
        aid_raw = d.get("agent_ids")
        if aid_raw is not None:
            try:
                aids = _extract_array(aid_raw).flatten().astype(np.int64)
            except (ValueError, TypeError):
                aids = None
        else:
            aids = None

        unique_aids = np.unique(aids) if aids is not None else None
        need_split = unique_aids is not None and len(unique_aids) > 1

        def _extract_ts_ep(d_sub, n_sub):
            ts_raw = d_sub.get("timesteps")
            if ts_raw is not None:
                try:
                    ts_arr = _extract_array(ts_raw).flatten().astype(np.int64)
                    ts_out = ts_arr if ts_arr.size == n_sub else np.arange(n_sub, dtype=np.int64)
                except (ValueError, TypeError):
                    ts_out = np.arange(n_sub, dtype=np.int64)
            else:
                ts_out = np.arange(n_sub, dtype=np.int64)

            ep_raw = d_sub.get("episode_ids")
            if ep_raw is not None:
                try:
                    ep_arr = _extract_array(ep_raw).flatten().astype(np.int64)
                    ep_out = ep_arr if ep_arr.size == n_sub else np.zeros(n_sub, dtype=np.int64)
                except (ValueError, TypeError):
                    ep_out = np.zeros(n_sub, dtype=np.int64)
            else:
                ep_out = np.zeros(n_sub, dtype=np.int64)
            return ts_out, ep_out

        if need_split:
            all_states = _ensure_2d(_extract_array(d["states"]))
            raw_acts = _extract_array(d["actions"])
            _cont = _actions_are_continuous(raw_acts)
            all_actions = _ensure_2d(raw_acts) if _cont else _ensure_1d_int(raw_acts)
            all_hidden = _ensure_2d(_extract_array(d["hidden"]))
            has_hid = d.get("has_hidden", False)
            n_all = len(all_states)
            ts_all, ep_all = _extract_ts_ep(d, n_all)

            if has_hid:
                any_real_hidden = True

            for uid in unique_aids:
                m = aids == uid
                ag_key = str(int(uid))
                Sd[ag_key] = all_states[m]
                Ad[ag_key] = all_actions[m]
                Hd[ag_key] = all_hidden[m] if has_hid else np.zeros((int(m.sum()), 1), dtype=np.float32)
                Td[ag_key] = ts_all[m]
                Ed[ag_key] = ep_all[m]
        else:
            Sd[ag] = _ensure_2d(_extract_array(d["states"]))
            raw_acts = _extract_array(d["actions"])
            Ad[ag] = _ensure_2d(raw_acts) if _actions_are_continuous(raw_acts) else _ensure_1d_int(raw_acts)
            Hd[ag] = _ensure_2d(_extract_array(d["hidden"]))
            if d.get("has_hidden", True):
                any_real_hidden = True

            n = len(Sd[ag])
            Td[ag], Ed[ag] = _extract_ts_ep(d, n)

    return Sd, Ad, Hd, Td, Ed, any_real_hidden


def load_dataset(path):
    """Load a pickle dataset produced by ``collect_data``."""
    with open(path, "rb") as f:
        return pickle.load(f)
