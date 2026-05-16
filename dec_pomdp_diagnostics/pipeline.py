"""Per-run orchestration: compute every diagnostic plus its permutation null.

The permutation null (Sec. 6.1, Decision Rules) is computed by independently
permuting each agent's action sequence within a run, which destroys temporal
and cross-agent dependencies while preserving each agent's marginal action
distribution. A real value is treated as significant only when it exceeds
the mean of the null replicates.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from .data import permute_actions, subsample_agents, unpack_run
from .metrics import (
    compute_aa,
    compute_dai_hidden,
    compute_dai_oa_hist,
    compute_har_hidden,
    compute_har_ohist,
    compute_oar,
    compute_pif_hidden,
    compute_pif_oa_hist,
    compute_pif_ohist,
)

logger = logging.getLogger(__name__)


AVAILABLE_METRICS = {"oar", "har", "pif", "aa", "dai"}

# All metric columns produced per run. Kept in module scope so summary code
# and CLI listings agree on the schema without re-deriving it.
METRIC_COLUMNS = [
    "oar_max", "oar_min", "oarR_max", "oarR_min",
    "har_ohist_max", "har_ohist_min", "harRcond_ohist_max", "harRcond_ohist_min",
    "har_hidden_max", "har_hidden_min", "harRcond_hidden_max", "harRcond_hidden_min",
    "pif_ohist_max", "pif_ohist_min", "pifRcond_ohist_max", "pifRcond_ohist_min", "pif_ohist_pairs",
    "pif_hidden_max", "pif_hidden_min", "pifRcond_hidden_max", "pifRcond_hidden_min", "pif_hidden_pairs",
    "pifOA_ohist_max", "pifOA_ohist_min", "pifOARcond_ohist_max", "pifOARcond_ohist_min",
    "aa_max", "aa_min", "aaRcond_max", "aaRcond_min", "aa_pairs",
    "daiOA_ohist_max", "daiOA_ohist_min", "daiOARcond_ohist_max", "daiOARcond_ohist_min", "daiOA_ohist_pairs",
    "dai_hidden_max", "dai_hidden_min", "daiRcond_hidden_max", "daiRcond_hidden_min", "dai_hidden_pairs",
]


def _agg(d):
    """Aggregate dict values to (max, min) ignoring NaN.

    Per Sec. 6.1: "we compute the *maximum* diagnostic value across agents,
    asking whether *any* agent exhibits the property."
    """
    arr = np.asarray(list(d.values()), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.max(arr)), float(np.min(arr))


def _log_dataset_shape(agent_keys, Sd, Hd, Td, Ed):
    for ak in agent_keys:
        n_samples = len(Sd[ak])
        unique_eps = np.unique(Ed[ak])
        unique_ts = np.unique(Td[ak])
        n_eps, n_ts = len(unique_eps), len(unique_ts)
        max_t = int(unique_ts.max()) if n_ts > 0 else -1
        expected = n_eps * n_ts
        logger.info(
            f"    agent {ak}: samples={n_samples}, "
            f"episodes={n_eps}, "
            f"steps_per_episode={n_ts} (max_t={max_t}), "
            f"expected={n_eps}×{n_ts}={expected}, "
            f"obs_dim={Sd[ak].shape[1] if Sd[ak].ndim == 2 else '?'}, "
            f"hidden_dim={Hd[ak].shape[1] if Hd[ak].ndim == 2 else '?'}"
        )
        if expected != n_samples:
            logger.warning(
                f"    ⚠ agent {ak}: sample count mismatch: expected {expected} got {n_samples}"
            )


def _compute_metrics_block(Sd, Ad, Hd, Td, Ed, has_hidden, *, metrics_to_run,
                           history_k, cmi_k, bits, force_continuous_A, kd_workers,
                           posterior_alpha, run_id, label="real"):
    """Compute every requested metric on the given (possibly null-permuted) data.

    Returns a dict of column -> value, suitable for merging into the per-run
    row. Catches per-metric exceptions so one failure doesn't kill the run.
    """
    out = {}

    if "oar" in metrics_to_run:
        try:
            v, r = compute_oar(Sd, Ad, k=cmi_k, bits=bits,
                               force_continuous_A=force_continuous_A, kd_workers=kd_workers,
                               posterior_alpha=posterior_alpha)
            out["oar_max"], out["oar_min"] = _agg(v)
            out["oarR_max"], out["oarR_min"] = _agg(r)
        except Exception as e:
            logger.warning(f"  OAR ({label}) failed ({run_id}): {e}")

    if "har" in metrics_to_run:
        try:
            v, r = compute_har_ohist(Sd, Ad, Td, Ed, k_window=history_k,
                                     k=cmi_k, bits=bits,
                                     force_continuous_A=force_continuous_A, kd_workers=kd_workers,
                                     posterior_alpha=posterior_alpha)
            out["har_ohist_max"], out["har_ohist_min"] = _agg(v)
            out["harRcond_ohist_max"], out["harRcond_ohist_min"] = _agg(r)
        except Exception as e:
            logger.warning(f"  HAR_ohist ({label}) failed ({run_id}): {e}")

        if has_hidden:
            try:
                v, r = compute_har_hidden(Sd, Hd, Ad, k=cmi_k, bits=bits,
                                          force_continuous_A=force_continuous_A, kd_workers=kd_workers,
                                          posterior_alpha=posterior_alpha)
                out["har_hidden_max"], out["har_hidden_min"] = _agg(v)
                out["harRcond_hidden_max"], out["harRcond_hidden_min"] = _agg(r)
            except Exception as e:
                logger.warning(f"  HAR_hidden ({label}) failed ({run_id}): {e}")

    if "pif" in metrics_to_run:
        try:
            v, r, n = compute_pif_ohist(Sd, Ad, Td, Ed, k_window=history_k,
                                        k=cmi_k, bits=bits,
                                        force_continuous_A=force_continuous_A, kd_workers=kd_workers,
                                        posterior_alpha=posterior_alpha)
            out["pif_ohist_max"], out["pif_ohist_min"] = _agg(v)
            out["pifRcond_ohist_max"], out["pifRcond_ohist_min"] = _agg(r)
            out["pif_ohist_pairs"] = n
        except Exception as e:
            logger.warning(f"  PIF_ohist ({label}) failed ({run_id}): {e}")

        if has_hidden:
            try:
                v, r, n = compute_pif_hidden(Hd, Sd, Ad, Td, Ed, k=cmi_k, bits=bits,
                                             force_continuous_A=force_continuous_A,
                                             kd_workers=kd_workers, posterior_alpha=posterior_alpha)
                out["pif_hidden_max"], out["pif_hidden_min"] = _agg(v)
                out["pifRcond_hidden_max"], out["pifRcond_hidden_min"] = _agg(r)
                out["pif_hidden_pairs"] = n
            except Exception as e:
                logger.warning(f"  PIF_hidden ({label}) failed ({run_id}): {e}")

        try:
            v, r, n = compute_pif_oa_hist(Sd, Ad, Td, Ed, k_window=history_k,
                                          k=cmi_k, bits=bits,
                                          force_continuous_A=force_continuous_A, kd_workers=kd_workers,
                                          posterior_alpha=posterior_alpha)
            out["pifOA_ohist_max"], out["pifOA_ohist_min"] = _agg(v)
            out["pifOARcond_ohist_max"], out["pifOARcond_ohist_min"] = _agg(r)
        except Exception as e:
            logger.warning(f"  PIF_OA ({label}) failed ({run_id}): {e}")

    if "aa" in metrics_to_run:
        try:
            v, r, n = compute_aa(Sd, Ad, Td, Ed, k=cmi_k, bits=bits,
                                 force_continuous_A=force_continuous_A, kd_workers=kd_workers,
                                 posterior_alpha=posterior_alpha)
            out["aa_max"], out["aa_min"] = _agg(v)
            out["aaRcond_max"], out["aaRcond_min"] = _agg(r)
            out["aa_pairs"] = n
        except Exception as e:
            logger.warning(f"  AA ({label}) failed ({run_id}): {e}")

    if "dai" in metrics_to_run:
        try:
            v, r, n = compute_dai_oa_hist(Sd, Ad, Td, Ed, k_window=history_k,
                                          k=cmi_k, bits=bits,
                                          force_continuous_A=force_continuous_A,
                                          kd_workers=kd_workers, posterior_alpha=posterior_alpha)
            out["daiOA_ohist_max"], out["daiOA_ohist_min"] = _agg(v)
            out["daiOARcond_ohist_max"], out["daiOARcond_ohist_min"] = _agg(r)
            out["daiOA_ohist_pairs"] = n
        except Exception as e:
            logger.warning(f"  DAI_OA ({label}) failed ({run_id}): {e}")

        if has_hidden:
            try:
                v, r, n = compute_dai_hidden(Hd, Sd, Ad, Td, Ed, k=cmi_k, bits=bits,
                                             force_continuous_A=force_continuous_A,
                                             kd_workers=kd_workers, posterior_alpha=posterior_alpha)
                out["dai_hidden_max"], out["dai_hidden_min"] = _agg(v)
                out["daiRcond_hidden_max"], out["daiRcond_hidden_min"] = _agg(r)
                out["dai_hidden_pairs"] = n
            except Exception as e:
                logger.warning(f"  DAI_hidden ({label}) failed ({run_id}): {e}")

    return out


def process_run(run_data, metrics_to_run, history_k=3, cmi_k=25, bits=True,
                force_continuous_A=False, null_reps=1, kd_workers=1, max_samples=None,
                posterior_alpha=0.5):
    """Compute every diagnostic for a single run (real values + permutation null)."""
    run_id = run_data["run_id"]
    run_name = run_data["run_name"]
    map_name = run_data["map_name"]
    alg_name = run_data["alg_name"]
    seed = run_data.get("config", {}).get("SEED", None)

    logger.info(f"  Computing metrics: {run_id} ({map_name} / {alg_name})")

    try:
        Sd, Ad, Hd, Td, Ed, has_hidden = unpack_run(run_data)
    except Exception as e:
        logger.warning(f"  Failed to unpack run {run_id}: {e}")
        return None

    agent_keys = sorted(
        Sd.keys(),
        key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else (x if isinstance(x, str) else str(x)),
    )
    logger.info(f"    agents={agent_keys}, has_hidden={has_hidden}")
    _log_dataset_shape(agent_keys, Sd, Hd, Td, Ed)

    if max_samples is not None:
        Sd, Ad, Hd, Td, Ed = subsample_agents(Sd, Ad, Hd, Td, Ed, max_samples,
                                              rng_seed=seed or 0)

    row = {
        "run_id": run_id,
        "run_name": run_name,
        "env_name": map_name,
        "alg": alg_name,
        "seed": seed,
        "HISTORY_K": history_k,
        "CMI_K": cmi_k,
        "POSTERIOR_ALPHA": posterior_alpha,
    }
    for c in METRIC_COLUMNS:
        row[c] = np.nan

    real = _compute_metrics_block(
        Sd, Ad, Hd, Td, Ed, has_hidden,
        metrics_to_run=metrics_to_run, history_k=history_k, cmi_k=cmi_k,
        bits=bits, force_continuous_A=force_continuous_A, kd_workers=kd_workers,
        posterior_alpha=posterior_alpha, run_id=run_id, label="real",
    )
    row.update(real)

    if null_reps > 0:
        logger.info(f"    Computing null baselines ({null_reps} reps)...")
        null_accum = defaultdict(list)
        for rep in range(null_reps):
            rng = np.random.default_rng(rep)
            Ad_null = permute_actions(Ad, rng)
            null_row = _compute_metrics_block(
                Sd, Ad_null, Hd, Td, Ed, has_hidden,
                metrics_to_run=metrics_to_run, history_k=history_k, cmi_k=cmi_k,
                bits=bits, force_continuous_A=force_continuous_A, kd_workers=kd_workers,
                posterior_alpha=posterior_alpha, run_id=run_id, label=f"null[{rep}]",
            )
            # Only the *_max columns participate in the null comparison.
            for col, val in null_row.items():
                if col.endswith("_max"):
                    null_accum[col].append(val)

        for nk, vals in null_accum.items():
            finite = [v for v in vals if np.isfinite(v)]
            if not finite:
                if vals:
                    logger.warning(f"  Null {nk} ({run_id}): all {len(vals)} reps NaN")
                row[f"{nk}_null"] = np.nan
            else:
                if len(finite) < len(vals):
                    logger.debug(
                        f"  Null {nk} ({run_id}): {len(vals)-len(finite)}/{len(vals)} NaN, "
                        f"averaging {len(finite)} finite values"
                    )
                row[f"{nk}_null"] = float(np.mean(finite))

    return row
