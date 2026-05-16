"""The five Dec-POMDP behavioural diagnostics from Tessera et al. (2026).

Each diagnostic is provided in two variants:

* For **feed-forward (FF)** policies, ``H^i_t`` is approximated by an
  observation history window ``O^i_{t-k:t-1}`` (and ``τ^i_t`` by an
  observation-action history window).
* For **recurrent (RNN)** policies, ``H^i_t`` is the RNN hidden state.

The paper's normalised reports use the FF variant for FF runs and the hidden
variant for RNN runs (see ``_NORM_COLS`` in ``summary.py``).

Diagnostic-to-function map
--------------------------

* ``OAR`` (Diag 3, Eq. 4): :func:`compute_oar`
* ``HAR`` (Diag 2, Eq. 3): :func:`compute_har_ohist` (FF) /
  :func:`compute_har_hidden` (RNN)
* ``PIF`` (Diag 4, Eq. 5): :func:`compute_pif_oa_hist` (FF, paper τ=[O,A]) /
  :func:`compute_pif_hidden` (RNN). :func:`compute_pif_ohist` is an
  observation-only ablation.
* ``AA`` (Diag 5, Eq. 6): :func:`compute_aa`
* ``DAI`` (Diag 6, Eq. 7): :func:`compute_dai_oa_hist` (FF) /
  :func:`compute_dai_hidden` (RNN)
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .data import (
    _action_window,
    _actions_are_continuous,
    _align_pair,
    _build_index,
    _collect_action,
    _stack_actions,
)
from .estimators import (
    _ensure_1d_int,
    _ensure_2d,
    _entropy_from_counts,
    _warn_discrete_posterior_risk,
    _zscore,
    cmi_mixed,
    mi_cc,
    mi_cd,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Diagnostic 3 (Eq. 4): OAR — Observation-Action Relevance
# ----------------------------------------------------------------------


def compute_oar(Sd, Ad, k=25, bits=True, force_continuous_A=False,
                mi_cd_max_dim=50, kd_workers=1, posterior_alpha=0.5):
    """OAR: ``I(O^i_t ; A^i_t)`` per agent.

    Picks an estimator based on action type and observation dimensionality:

    * Continuous actions: KSG-1 :func:`mi_cc` (PCA-reduced if ``obs_dim > mi_cd_max_dim``).
    * Discrete actions, low-D observations: Ross 2014 :func:`mi_cd`.
    * Discrete actions, high-D observations: kNN-posterior I(O;A) = H(A) - H(A|O).

    Returns
    -------
    oar : dict[agent -> float]
        OAR in bits.
    r_uncond : dict[agent -> float]
        Normalised ``OAR^norm = OAR / H(A)`` in [0, 1].
    """
    oar, r_uncond = {}, {}
    agents = sorted(set(Sd.keys()) & set(Ad.keys()))
    for a in agents:
        S, A = Sd[a], Ad[a]
        obs_dim = S.shape[1]

        if force_continuous_A or _actions_are_continuous(A):
            if obs_dim > mi_cd_max_dim:
                logger.info(f"Agent {a}: reducing obs dim {obs_dim}->{mi_cd_max_dim} via PCA for OAR.")
                n_components = min(mi_cd_max_dim, S.shape[0] - 1)
                S = PCA(n_components=n_components, random_state=42).fit_transform(S)
            k_ksg = min(k, 3)
            logger.info(f"Agent {a}: continuous actions, KSG with k={k_ksg}")
            v, ex = mi_cc(S, A, k=k_ksg, bits=bits)
        elif obs_dim <= mi_cd_max_dim:
            k_ross = min(k, 3)
            logger.info(f"Agent {a}: Ross estimator (low-D), k={k_ross}")
            v, ex = mi_cd(S, A, k=k_ross, bits=bits)
        else:
            logger.info(f"Agent {a}: reducing obs dim {obs_dim}->{mi_cd_max_dim} via PCA for OAR.")
            n_components = min(mi_cd_max_dim, S.shape[0] - 1)
            S = PCA(n_components=n_components, random_state=42).fit_transform(S)
            logger.info(f"Agent {a}: posterior-KL estimator (high-D)")
            v, ex = _oar_posterior_kl(S, A, k=k, bits=bits, alpha=posterior_alpha)

        logger.info(f"Agent {a}: OAR = {v:.4f} bits, R_uncond = {ex.get('R_uncond', np.nan):.4f}")
        oar[a] = float(v) if np.isfinite(v) else np.nan
        R = ex.get("R_uncond", np.nan)
        r_uncond[a] = float(np.clip(R, 0, 1)) if np.isfinite(R) else np.nan
    return oar, r_uncond


def _oar_posterior_kl(S, A, k=25, bits=True, alpha=0.5):
    """High-D discrete-action OAR: I(O;A) = H(A) - H(A|O) via kNN posterior voting."""
    obs_arr = _ensure_2d(S).astype(float)
    y = _ensure_1d_int(A)
    n = len(y)
    n_classes = int(np.max(y) + 1)
    observed_classes = int(np.unique(y).size)

    obs_z = _zscore(obs_arr)
    knn = min(k, n - 1)
    _warn_discrete_posterior_risk(
        "OAR",
        n_classes=n_classes,
        observed_classes=observed_classes,
        k=knn,
        alpha=alpha,
        n_samples=n,
        obs_dim=obs_arr.shape[1],
    )
    nbrs = NearestNeighbors(n_neighbors=knn + 1, algorithm="auto").fit(obs_z)
    _, idx = nbrs.kneighbors(obs_z)
    idx = idx[:, 1:]

    counts = np.zeros((n, n_classes), dtype=float)
    rows_idx = np.arange(n)[:, None].repeat(knn, axis=1)
    np.add.at(counts, (rows_idx, y[idx]), 1.0)
    p_local = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * n_classes)
    p_local = np.clip(p_local, 1e-12, 1.0)
    H_A_given_O_nats = float(-np.mean(np.sum(p_local * np.log(p_local), axis=1)))

    gc = np.bincount(y, minlength=n_classes).astype(float)
    H_A_nats = _entropy_from_counts(gc, bits=False)

    mi_nats = max(0.0, H_A_nats - H_A_given_O_nats)
    if mi_nats == 0.0 and H_A_nats > 1e-12 and H_A_given_O_nats > H_A_nats:
        inv = 1.0 / np.log(2.0) if bits else 1.0
        logger.warning(
            "OAR posterior estimate clipped to 0 because estimated H(A|O)=%.4f "
            "exceeds H(A)=%.4f (bits=%s, k=%d, posterior_alpha=%g, "
            "observed_actions=%d, encoded_actions=%d, obs_dim=%d, n=%d). "
            "This often means local posteriors are over-smoothed or too sparse "
            "for a high-cardinality discrete action space. Try lowering "
            "posterior_alpha (for example 0.01), increasing --cmi-k and/or "
            "--max-samples, and checking action-ID encoding.",
            H_A_given_O_nats * inv,
            H_A_nats * inv,
            bits,
            int(knn),
            float(alpha),
            int(observed_classes),
            int(n_classes),
            int(obs_arr.shape[1]),
            int(n),
        )
    R_val = mi_nats / max(H_A_nats, 1e-12) if H_A_nats > 1e-12 else np.nan

    inv = 1.0 / np.log(2.0) if bits else 1.0
    v = mi_nats * inv
    return v, {
        "mi": v,
        "H_A": H_A_nats * inv,
        "H_A_given_O": H_A_given_O_nats * inv,
        "R_uncond": R_val,
        "posterior_alpha": float(alpha),
        "n_classes": int(n_classes),
        "observed_classes": int(observed_classes),
        "posterior_pseudocount_mass": float(alpha) * int(n_classes),
    }


# ----------------------------------------------------------------------
# Diagnostic 2 (Eq. 3): HAR — History-Action Relevance
# ----------------------------------------------------------------------


def compute_har_hidden(Sd, Hd, Ad, k=25, bits=True, force_continuous_A=False,
                       kd_workers=1, posterior_alpha=0.5):
    """HAR (RNN): ``I(H^i_t ; A^i_t | O^i_t)`` using the recurrent hidden state.

    Returns ``(har, R_cond)`` per agent. ``R_cond = HAR / H(A|O)`` in [0, 1].
    """
    har, rcond = {}, {}
    agents = sorted(set(Sd.keys()) & set(Hd.keys()) & set(Ad.keys()))
    for a in agents:
        v, ex = cmi_mixed(Sd[a], Hd[a], Ad[a], k=k, bits=bits,
                          alpha=posterior_alpha,
                          force_continuous_A=force_continuous_A, kd_workers=kd_workers)
        har[a] = float(v) if np.isfinite(v) else np.nan
        R = ex.get("R_cond_from_kl", np.nan)
        rcond[a] = float(np.clip(R, 0, 1)) if np.isfinite(R) else np.nan
    return har, rcond


def compute_har_ohist(Sd, Ad, Td, Ed, k_window=3, k=25, bits=True,
                      force_continuous_A=False, kd_workers=1, posterior_alpha=0.5):
    """HAR (FF): ``I(O^i_{t-k:t-1} ; A^i_t | O^i_t)``.

    Builds the history window using episode/env/time keys to avoid leakage
    across episodes. Falls back to a smaller window if the rollout is shorter
    than ``k_window``.
    """
    har, rcond = {}, {}
    agents = sorted(set(Sd.keys()) & set(Ad.keys()))
    for a in agents:
        S, A = Sd.get(a), Ad.get(a)
        T, E = Td.get(a), Ed.get(a)
        if any(x is None for x in [S, A, T, E]):
            har[a] = rcond[a] = np.nan
            continue

        idx_map, ep = _build_index(T, E)
        max_t = int(np.max(T))
        eff_k = min(k_window, max_t) if max_t > 0 else k_window
        if eff_k < 1:
            har[a] = rcond[a] = np.nan
            continue

        O_list, H_list, A_list = [], [], []
        for row in range(len(T)):
            t, e = int(T[row]), int(E[row])
            if ep is None:
                keys_hist = [(e, tau) for tau in range(t - eff_k, t)]
                key_curr = (e, t)
            else:
                epi = int(ep[row])
                keys_hist = [(epi, e, tau) for tau in range(t - eff_k, t)]
                key_curr = (epi, e, t)

            idxs = [idx_map.get(kx, -1) for kx in keys_hist]
            if any(j < 0 for j in idxs):
                continue

            H_hist = np.concatenate([S[j] for j in idxs], axis=0)
            O_curr = S[idx_map[key_curr]]
            O_list.append(O_curr)
            H_list.append(H_hist)
            A_list.append(_collect_action(A, row))

        if len(A_list) <= 1:
            har[a] = rcond[a] = np.nan
        else:
            O_mat, H_mat = np.stack(O_list), np.stack(H_list)
            A_vec, _ = _stack_actions(A_list)
            v, ex = cmi_mixed(O_mat, H_mat, A_vec, k=k, bits=bits,
                              alpha=posterior_alpha,
                              force_continuous_A=force_continuous_A, kd_workers=kd_workers)
            har[a] = float(v) if np.isfinite(v) else np.nan
            R = ex.get("R_cond_from_kl", np.nan)
            rcond[a] = float(np.clip(R, 0, 1)) if np.isfinite(R) else np.nan
    return har, rcond


# ----------------------------------------------------------------------
# Diagnostic 4 (Eq. 5): PIF — Private Information Flow
# ----------------------------------------------------------------------


def compute_pif_hidden(Hd, Sd, Ad, Td, Ed, k=25, bits=True,
                       force_continuous_A=False, kd_workers=1, posterior_alpha=0.5):
    """PIF (RNN): ``I(H^i_t ; A^j_t | H^j_t)`` per ``(i -> j)`` pair.

    Aggregated by destination ``j`` (mean over sources ``i ≠ j``).
    Returns ``(pif_per_dst, rcond_per_dst, n_pairs)``.
    """
    agents = sorted(set(Hd.keys()) & set(Ad.keys()) & set(Sd.keys()))
    dst_vals = {j: [] for j in agents}
    dst_rcond = {j: [] for j in agents}
    total = 0

    agent_idx = {a: _build_index(Td[a], Ed[a])
                 for a in agents if Td.get(a) is not None and Ed.get(a) is not None}

    for i in agents:
        for j in agents:
            if i == j:
                continue
            H_i, H_j, A_j = Hd.get(i), Hd.get(j), Ad.get(j)
            T_i, E_i = Td.get(i), Ed.get(i)
            T_j, E_j = Td.get(j), Ed.get(j)
            if any(x is None for x in [H_i, H_j, A_j, T_i, E_i, T_j, E_j]):
                continue
            ii, jj = _align_pair(T_i, E_i, T_j, E_j,
                                 map_i=agent_idx[i][0], epj=agent_idx[j][1])
            if ii.size <= 1:
                continue
            v, ex = cmi_mixed(H_j[jj], H_i[ii], A_j[jj], k=k, bits=bits,
                              alpha=posterior_alpha,
                              force_continuous_A=force_continuous_A, kd_workers=kd_workers)
            if np.isfinite(v):
                dst_vals[j].append(float(v))
                R = ex.get("R_cond_from_kl", np.nan)
                if np.isfinite(R):
                    dst_rcond[j].append(float(np.clip(R, 0, 1)))
                total += 1

    pif = {j: float(np.mean(v)) if v else np.nan for j, v in dst_vals.items()}
    rc = {j: float(np.mean(v)) if v else np.nan for j, v in dst_rcond.items()}
    return pif, rc, total


def compute_pif_ohist(Sd, Ad, Td, Ed, k_window=3, k=25, bits=True,
                      force_continuous_A=False, kd_workers=1, posterior_alpha=0.5):
    """PIF (observation-only ablation): ``I(O^i_{t-k:t} ; A^j_t | O^j_{t-k:t})``.

    Uses the observation window only — useful for isolating the effect of
    excluding past actions from ``τ``.
    """
    agents = sorted(set(Sd.keys()) & set(Ad.keys()))
    dst_vals = {j: [] for j in agents}
    dst_rcond = {j: [] for j in agents}
    total = 0

    all_max_t = [int(np.max(Td[a])) for a in agents if Td.get(a) is not None]
    eff_k = min(k_window, min(all_max_t)) if all_max_t and min(all_max_t) > 0 else k_window
    if eff_k < 1:
        return ({j: np.nan for j in agents}, {j: np.nan for j in agents}, 0)

    agent_idx = {a: _build_index(Td[a], Ed[a])
                 for a in agents if Td.get(a) is not None and Ed.get(a) is not None}

    for i in agents:
        for j in agents:
            if i == j:
                continue
            Oi, Oj, Aj = Sd.get(i), Sd.get(j), Ad.get(j)
            Ti, Ei = Td.get(i), Ed.get(i)
            Tj, Ej = Td.get(j), Ed.get(j)
            if any(x is None for x in [Oi, Oj, Aj, Ti, Ei, Tj, Ej]):
                continue

            map_i, _ = agent_idx[i]
            map_j, epj = agent_idx[j]

            Hsrc, Hdst, A_list = [], [], []
            for row in range(len(Tj)):
                t, e = int(Tj[row]), int(Ej[row])
                if epj is None:
                    keys_j = [(e, tau) for tau in range(t - eff_k, t)]
                    keys_i = [(e, tau) for tau in range(t - eff_k, t)]
                    cur_key = (e, t)
                else:
                    ep_cur = int(epj[row])
                    keys_j = [(ep_cur, e, tau) for tau in range(t - eff_k, t)]
                    keys_i = [(ep_cur, e, tau) for tau in range(t - eff_k, t)]
                    cur_key = (ep_cur, e, t)

                jj = [map_j.get(kx, -1) for kx in keys_j]
                ii = [map_i.get(kx, -1) for kx in keys_i]
                if any(u < 0 for u in jj) or any(v < 0 for v in ii):
                    continue
                cur_idx_j = map_j.get(cur_key, -1)
                cur_idx_i = map_i.get(cur_key, -1)
                if cur_idx_j < 0 or cur_idx_i < 0:
                    continue

                Hdst.append(np.concatenate([Oj[u] for u in jj] + [Oj[cur_idx_j]], axis=0))
                Hsrc.append(np.concatenate([Oi[v] for v in ii] + [Oi[cur_idx_i]], axis=0))
                A_list.append(_collect_action(Aj, cur_idx_j))

            if len(A_list) > 1:
                O_mat, H_mat = np.stack(Hdst), np.stack(Hsrc)
                A_vec, _ = _stack_actions(A_list)
                v, ex = cmi_mixed(O_mat, H_mat, A_vec, k=k, bits=bits,
                                  alpha=posterior_alpha,
                                  force_continuous_A=force_continuous_A, kd_workers=kd_workers)
                if np.isfinite(v):
                    dst_vals[j].append(float(v))
                    R = ex.get("R_cond_from_kl", np.nan)
                    if np.isfinite(R):
                        dst_rcond[j].append(float(np.clip(R, 0, 1)))
                    total += 1

    pif = {j: float(np.mean(v)) if v else np.nan for j, v in dst_vals.items()}
    rc = {j: float(np.mean(v)) if v else np.nan for j, v in dst_rcond.items()}
    return pif, rc, total


def compute_pif_oa_hist(Sd, Ad, Td, Ed, k_window=3, k=25, bits=True,
                        force_continuous_A=False, kd_workers=1, posterior_alpha=0.5):
    """PIF (FF, paper definition with ``τ = [O, A]``).

    ``I([O^i_{t-k:t}, A^i_{t-k:t-1}] ; A^j_t | [O^j_{t-k:t}, A^j_{t-k:t-1}])``.

    ``τ_t`` includes the current observation ``O_t`` and past actions
    ``A_{t-k:t-1}`` (the to-be-predicted ``A_t`` is the target, not part of
    ``τ_t``).
    """
    agents = sorted(set(Sd.keys()) & set(Ad.keys()))
    dst_vals = {j: [] for j in agents}
    dst_rcond = {j: [] for j in agents}
    total = 0

    all_max_t = [int(np.max(Td[a])) for a in agents if Td.get(a) is not None]
    eff_k = min(k_window, min(all_max_t)) if all_max_t and min(all_max_t) > 0 else k_window
    if eff_k < 1:
        return ({j: np.nan for j in agents}, {j: np.nan for j in agents}, 0)

    agent_idx = {a: _build_index(Td[a], Ed[a])
                 for a in agents if Td.get(a) is not None and Ed.get(a) is not None}
    nclasses = {a: (None if _actions_are_continuous(Ad[a])
                    else max(int(np.max(_ensure_1d_int(Ad[a]))) + 1, 2))
                for a in agents if Ad.get(a) is not None}

    for i in agents:
        for j in agents:
            if i == j:
                continue
            Oi, Oj = Sd.get(i), Sd.get(j)
            Ai, Aj = Ad.get(i), Ad.get(j)
            Ti, Ei = Td.get(i), Ed.get(i)
            Tj, Ej = Td.get(j), Ed.get(j)
            if any(x is None for x in [Oi, Oj, Ai, Aj, Ti, Ei, Tj, Ej]):
                continue

            map_i, _ = agent_idx[i]
            map_j, epj = agent_idx[j]
            nc_i, nc_j = nclasses[i], nclasses[j]

            Hsrc, Hdst, A_list = [], [], []
            for row in range(len(Tj)):
                t, e = int(Tj[row]), int(Ej[row])
                if epj is None:
                    keys_j = [(e, tau) for tau in range(t - eff_k, t)]
                    keys_i = [(e, tau) for tau in range(t - eff_k, t)]
                    cur_key = (e, t)
                else:
                    ep_cur = int(epj[row])
                    keys_j = [(ep_cur, e, tau) for tau in range(t - eff_k, t)]
                    keys_i = [(ep_cur, e, tau) for tau in range(t - eff_k, t)]
                    cur_key = (ep_cur, e, t)

                jj = [map_j.get(kx, -1) for kx in keys_j]
                ii = [map_i.get(kx, -1) for kx in keys_i]
                if any(u < 0 for u in jj) or any(v < 0 for v in ii):
                    continue
                cur_idx_j = map_j.get(cur_key, -1)
                cur_idx_i = map_i.get(cur_key, -1)
                if cur_idx_j < 0 or cur_idx_i < 0:
                    continue

                O_hist_j = np.concatenate([Oj[u] for u in jj] + [Oj[cur_idx_j]], axis=0)
                O_hist_i = np.concatenate([Oi[v] for v in ii] + [Oi[cur_idx_i]], axis=0)
                A_hist_j = _action_window(jj, Aj, n_classes=nc_j)
                A_hist_i = _action_window(ii, Ai, n_classes=nc_i)

                Hdst.append(np.concatenate([O_hist_j, A_hist_j], axis=0))
                Hsrc.append(np.concatenate([O_hist_i, A_hist_i], axis=0))
                A_list.append(_collect_action(Aj, cur_idx_j))

            if len(A_list) > 1:
                O_mat, H_mat = np.stack(Hdst), np.stack(Hsrc)
                A_vec, _ = _stack_actions(A_list)
                v, ex = cmi_mixed(O_mat, H_mat, A_vec, k=k, bits=bits,
                                  alpha=posterior_alpha,
                                  force_continuous_A=force_continuous_A, kd_workers=kd_workers)
                if np.isfinite(v):
                    dst_vals[j].append(float(v))
                    R = ex.get("R_cond_from_kl", np.nan)
                    if np.isfinite(R):
                        dst_rcond[j].append(float(np.clip(R, 0, 1)))
                    total += 1

    pif = {j: float(np.mean(v)) if v else np.nan for j, v in dst_vals.items()}
    rc = {j: float(np.mean(v)) if v else np.nan for j, v in dst_rcond.items()}
    return pif, rc, total


# ----------------------------------------------------------------------
# Diagnostic 5 (Eq. 6): AA — Action–Action Coupling
# ----------------------------------------------------------------------


def compute_aa(Sd, Ad, Td, Ed, k=25, bits=True, force_continuous_A=False,
               kd_workers=1, posterior_alpha=0.5):
    """AA: ``I(A^i_t ; A^j_t | O^i_t, O^j_t)``.

    Aggregated by destination ``j`` (mean over sources ``i ≠ j``). Captures
    instantaneous, observation-conditioned action coupling — e.g. agents
    coordinating on roles or symmetry-breaking.
    """
    agents = sorted(set(Sd.keys()) & set(Ad.keys()))
    dst_vals = {j: [] for j in agents}
    dst_rcond = {j: [] for j in agents}
    total = 0

    C = 0
    for a in agents:
        if Ad.get(a) is not None and not _actions_are_continuous(Ad[a]):
            C = max(C, int(np.max(Ad[a])) + 1)
    C = max(C, 2)

    agent_idx = {a: _build_index(Td[a], Ed[a])
                 for a in agents if Td.get(a) is not None and Ed.get(a) is not None}

    for i in agents:
        for j in agents:
            if i == j:
                continue
            Si, Sj = Sd.get(i), Sd.get(j)
            Ai, Aj = Ad.get(i), Ad.get(j)
            Ti, Ei = Td.get(i), Ed.get(i)
            Tj, Ej = Td.get(j), Ed.get(j)
            if any(x is None for x in [Si, Sj, Ai, Aj, Ti, Ei, Tj, Ej]):
                continue

            ii, jj = _align_pair(Ti, Ei, Tj, Ej,
                                 map_i=agent_idx[i][0], epj=agent_idx[j][1])
            if ii.size <= 1:
                continue

            Si_2d, Sj_2d = _ensure_2d(Si), _ensure_2d(Sj)
            O_cond = np.concatenate([Si_2d[ii], Sj_2d[jj]], axis=1)

            if force_continuous_A or _actions_are_continuous(Ai):
                H_src = _ensure_2d(Ai)[ii]
            else:
                Ai_int = _ensure_1d_int(Ai)
                H_src = np.eye(C, dtype=float)[Ai_int[ii]]

            if force_continuous_A or _actions_are_continuous(Aj):
                A_tgt = _ensure_2d(Aj)[jj]
            else:
                A_tgt = _ensure_1d_int(Aj)[jj]

            v, ex = cmi_mixed(O_cond, H_src, A_tgt, k=k, bits=bits,
                              alpha=posterior_alpha,
                              force_continuous_A=force_continuous_A, kd_workers=kd_workers)
            if np.isfinite(v):
                dst_vals[j].append(float(v))
                R = ex.get("R_cond_from_kl", np.nan)
                if np.isfinite(R):
                    dst_rcond[j].append(float(np.clip(R, 0, 1)))
                total += 1

    aa = {j: float(np.mean(v)) if v else np.nan for j, v in dst_vals.items()}
    rc = {j: float(np.mean(v)) if v else np.nan for j, v in dst_rcond.items()}
    return aa, rc, total


# ----------------------------------------------------------------------
# Diagnostic 6 (Eq. 7): DAI — Directed Action Information
# ----------------------------------------------------------------------


def compute_dai_hidden(Hd, Sd, Ad, Td, Ed, k=25, bits=True,
                       force_continuous_A=False, kd_workers=1, posterior_alpha=0.5):
    """DAI (RNN): ``Σ_t I(H^i_{t-1} ; A^j_t | H^j_{t-1})``.

    Uses hidden states from ``t-1`` to predict ``A^j`` at ``t``. Estimated via
    pooled samples and reported as the average per-timestep value.
    """
    agents = sorted(set(Hd.keys()) & set(Ad.keys()) & set(Sd.keys()))
    dst_vals = {j: [] for j in agents}
    dst_rcond = {j: [] for j in agents}
    total = 0

    agent_idx = {a: _build_index(Td[a], Ed[a])
                 for a in agents if Td.get(a) is not None and Ed.get(a) is not None}

    for i in agents:
        for j in agents:
            if i == j:
                continue
            H_i, H_j, A_j = Hd.get(i), Hd.get(j), Ad.get(j)
            T_i, E_i = Td.get(i), Ed.get(i)
            T_j, E_j = Td.get(j), Ed.get(j)
            if any(x is None for x in [H_i, H_j, A_j, T_i, E_i, T_j, E_j]):
                continue

            map_i, _ = agent_idx[i]
            map_j, epj = agent_idx[j]

            Hsrc, Hdst, A_list = [], [], []
            for row in range(len(T_j)):
                t, e = int(T_j[row]), int(E_j[row])
                if t < 1:
                    continue
                if epj is None:
                    prev_key = (e, t - 1)
                else:
                    ep_cur = int(epj[row])
                    prev_key = (ep_cur, e, t - 1)

                idx_j_prev = map_j.get(prev_key, -1)
                idx_i_prev = map_i.get(prev_key, -1)
                if idx_j_prev < 0 or idx_i_prev < 0:
                    continue

                Hdst.append(H_j[idx_j_prev])
                Hsrc.append(H_i[idx_i_prev])
                A_list.append(_collect_action(A_j, row))

            if len(A_list) <= 1:
                continue
            O_mat, H_mat = np.stack(Hdst), np.stack(Hsrc)
            A_vec, _ = _stack_actions(A_list)
            v, ex = cmi_mixed(O_mat, H_mat, A_vec, k=k, bits=bits,
                              alpha=posterior_alpha,
                              force_continuous_A=force_continuous_A, kd_workers=kd_workers)
            if np.isfinite(v):
                dst_vals[j].append(float(v))
                R = ex.get("R_cond_from_kl", np.nan)
                if np.isfinite(R):
                    dst_rcond[j].append(float(np.clip(R, 0, 1)))
                total += 1

    dai = {j: float(np.mean(v)) if v else np.nan for j, v in dst_vals.items()}
    rc = {j: float(np.mean(v)) if v else np.nan for j, v in dst_rcond.items()}
    return dai, rc, total


def compute_dai_oa_hist(Sd, Ad, Td, Ed, k_window=3, k=25, bits=True,
                        force_continuous_A=False, kd_workers=1, posterior_alpha=0.5):
    """DAI (FF, paper definition): ``Σ_t I(τ^i_{t-1} ; A^j_t | τ^j_{t-1})``.

    ``τ_{t-1} = [O, A]_{t-k:t-1}`` — history up to and including ``t-1``,
    excluding the current observation ``O_t`` (this is the key contrast with
    PIF, which conditions on ``τ_t`` including ``O_t``).
    """
    agents = sorted(set(Sd.keys()) & set(Ad.keys()))
    dst_vals = {j: [] for j in agents}
    dst_rcond = {j: [] for j in agents}
    total = 0

    all_max_t = [int(np.max(Td[a])) for a in agents if Td.get(a) is not None]
    eff_k = min(k_window, min(all_max_t)) if all_max_t and min(all_max_t) > 0 else k_window
    if eff_k < 1:
        return ({j: np.nan for j in agents}, {j: np.nan for j in agents}, 0)

    agent_idx = {a: _build_index(Td[a], Ed[a])
                 for a in agents if Td.get(a) is not None and Ed.get(a) is not None}
    nclasses = {a: (None if _actions_are_continuous(Ad[a])
                    else max(int(np.max(_ensure_1d_int(Ad[a]))) + 1, 2))
                for a in agents if Ad.get(a) is not None}

    for i in agents:
        for j in agents:
            if i == j:
                continue
            Oi, Oj = Sd.get(i), Sd.get(j)
            Ai, Aj = Ad.get(i), Ad.get(j)
            Ti, Ei = Td.get(i), Ed.get(i)
            Tj, Ej = Td.get(j), Ed.get(j)
            if any(x is None for x in [Oi, Oj, Ai, Aj, Ti, Ei, Tj, Ej]):
                continue

            map_i, _ = agent_idx[i]
            map_j, epj = agent_idx[j]
            nc_i, nc_j = nclasses[i], nclasses[j]

            Hsrc, Hdst, A_list = [], [], []
            for row in range(len(Tj)):
                t, e = int(Tj[row]), int(Ej[row])
                if epj is None:
                    keys_j = [(e, tau) for tau in range(t - eff_k, t)]
                    keys_i = [(e, tau) for tau in range(t - eff_k, t)]
                    cur_key = (e, t)
                else:
                    ep_cur = int(epj[row])
                    keys_j = [(ep_cur, e, tau) for tau in range(t - eff_k, t)]
                    keys_i = [(ep_cur, e, tau) for tau in range(t - eff_k, t)]
                    cur_key = (ep_cur, e, t)

                jj = [map_j.get(kx, -1) for kx in keys_j]
                ii = [map_i.get(kx, -1) for kx in keys_i]
                if any(u < 0 for u in jj) or any(v < 0 for v in ii):
                    continue
                cur_idx_j = map_j.get(cur_key, -1)
                if cur_idx_j < 0:
                    continue

                # τ_{t-1}: [O, A]_{t-k:t-1}, NO current O_t (key contrast with PIF).
                O_hist_j = np.concatenate([Oj[u] for u in jj], axis=0)
                O_hist_i = np.concatenate([Oi[v] for v in ii], axis=0)
                A_hist_j = _action_window(jj, Aj, n_classes=nc_j)
                A_hist_i = _action_window(ii, Ai, n_classes=nc_i)

                Hdst.append(np.concatenate([O_hist_j, A_hist_j], axis=0))
                Hsrc.append(np.concatenate([O_hist_i, A_hist_i], axis=0))
                A_list.append(_collect_action(Aj, cur_idx_j))

            if len(A_list) > 1:
                O_mat, H_mat = np.stack(Hdst), np.stack(Hsrc)
                A_vec, _ = _stack_actions(A_list)
                v, ex = cmi_mixed(O_mat, H_mat, A_vec, k=k, bits=bits,
                                  alpha=posterior_alpha,
                                  force_continuous_A=force_continuous_A, kd_workers=kd_workers)
                if np.isfinite(v):
                    dst_vals[j].append(float(v))
                    R = ex.get("R_cond_from_kl", np.nan)
                    if np.isfinite(R):
                        dst_rcond[j].append(float(np.clip(R, 0, 1)))
                    total += 1

    dai = {j: float(np.mean(v)) if v else np.nan for j, v in dst_vals.items()}
    rc = {j: float(np.mean(v)) if v else np.nan for j, v in dst_rcond.items()}
    return dai, rc, total
