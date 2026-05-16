"""kNN-based mutual information and conditional mutual information estimators.

Three families of estimators are provided:

* ``cmi_knn`` / ``cmi_mixed``: posterior-KL CMI for **discrete** actions, with
  a continuous-actions branch that dispatches to the Frenzel-Pompe estimator.
* ``mi_cd`` / ``mi_cc``: marginal MI estimators (Ross 2014 for mixed
  continuous/discrete, KSG-1 for continuous-continuous). Used by ``OAR``.
* ``_kl_entropy`` / ``_mi_ksg`` / ``_cmi_frenzel_pompe``: low-level KSG
  building blocks, used by ``_cmi_continuous_actions``.

All estimators return values in **bits** (``bits=True``, default) or **nats**.
CMI/MI are clipped to be non-negative (the kNN estimators can otherwise
produce small negatives from finite-sample noise).
"""

from __future__ import annotations

import logging

import numpy as np
import scipy.special
from scipy.spatial import cKDTree
from scipy.special import digamma
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.preprocessing import scale

logger = logging.getLogger(__name__)

_DISCRETE_POSTERIOR_WARNED = set()


# ----------------------------------------------------------------------
# Array helpers
# ----------------------------------------------------------------------


def _ensure_1d_int(x):
    x = np.asarray(x)
    if x.ndim > 1 and x.shape[-1] == 1:
        x = x[..., 0]
    x = x.reshape(-1)
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.int64, copy=False)
    _, inv = np.unique(x, return_inverse=True)
    return inv.astype(np.int64)


def _ensure_2d(X):
    X = np.asarray(X)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def _zscore(X, eps=1e-12):
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where(sd < eps, eps, sd)
    Z = (X - mu) / sd
    return np.where(np.isnan(Z), 0.0, Z)


def _is_discrete(arr, threshold=20):
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return True
    if arr.ndim == 1 and len(arr) > 0:
        finite = arr[np.isfinite(arr)]
        if len(finite) > 0 and np.all(finite == np.floor(finite)):
            return True
        return len(np.unique(arr)) <= threshold
    if arr.ndim >= 2:
        return False
    return False


def _entropy_from_counts(counts, bits=True, alpha=0.0):
    counts = np.asarray(counts, dtype=float)
    if alpha > 0:
        counts = counts + alpha
    Z = counts.sum()
    if Z <= 0:
        return 0.0
    p = np.clip(counts / Z, 1e-12, 1.0)
    H = -float(np.sum(p * np.log(p)))
    return H / np.log(2.0) if bits else H


def _warn_discrete_posterior_risk(context, *, n_classes, observed_classes,
                                  k, alpha, n_samples, obs_dim, hidden_dim=None):
    """Warn once when posterior smoothing can swamp local neighbour evidence."""
    pseudo_mass = float(alpha) * int(n_classes)
    if n_classes < 20:
        return
    if pseudo_mass < max(1.0, 0.25 * max(int(k), 1)):
        return

    key = (
        context,
        int(n_classes),
        int(observed_classes),
        int(k),
        round(float(alpha), 8),
        int(obs_dim),
        None if hidden_dim is None else int(hidden_dim),
    )
    if key in _DISCRETE_POSTERIOR_WARNED:
        return
    _DISCRETE_POSTERIOR_WARNED.add(key)

    dims = f"obs_dim={obs_dim}"
    if hidden_dim is not None:
        dims += f", extra_dim={hidden_dim}"
    logger.warning(
        "%s: discrete-action posterior estimator has many action classes "
        "(observed=%d, encoded=%d) with k=%d and posterior_alpha=%g "
        "(alpha * classes = %.2f pseudo-counts; n=%d, %s). "
        "This can over-smooth local kNN posteriors and make MI/CMI diagnostics "
        "appear as exact zeros. If zeros look suspicious, try lowering "
        "posterior_alpha (for example 0.01), increasing --cmi-k and/or "
        "--max-samples, and checking that action IDs are compact/intentional.",
        context,
        int(observed_classes),
        int(n_classes),
        int(k),
        float(alpha),
        pseudo_mass,
        int(n_samples),
        dims,
    )


# ----------------------------------------------------------------------
# CMI for discrete actions: I(H ; A | O) via nested-kNN posterior KL
# ----------------------------------------------------------------------


def cmi_knn(obs, hidden, actions, k=25, bits=True, alpha=0.5,
            cand_mult=5, use_nested=True, nonneg=True):
    """Estimate I(H; A | O) when A is discrete.

    Approximates p(A|O) and p(A|O,H) via Laplace-smoothed kNN posteriors,
    then takes the per-sample KL divergence and averages.

    Returns
    -------
    cmi : float
        I(H; A | O) in bits (or nats if ``bits=False``).
    extras : dict
        Includes ``H_A``, ``H_A_given_O``, and ``R_cond_from_kl`` (the
        normalised value used as ``HAR^norm`` etc.).
    """
    obs_arr = _ensure_2d(obs)
    hist_arr = _ensure_2d(hidden)
    y = _ensure_1d_int(actions)

    n = len(obs_arr)
    obs_z = _zscore(obs_arr)
    hist_z = _zscore(hist_arr)
    obs_hist = np.concatenate([obs_z, hist_z], axis=1)

    def _knn(X, k_nn):
        m = min(max(1, k_nn), n - 1)
        nbrs = NearestNeighbors(n_neighbors=m + 1, algorithm="auto").fit(X)
        _, idx = nbrs.kneighbors(X)
        return idx[:, 1:]

    idx_obs = _knn(obs_z, max(k, 2))
    kcand = min(n - 1, max(k + 5, cand_mult * k))

    if use_nested:
        idx_obs_cand = _knn(obs_z, kcand)
        idx_obs_hist = np.empty((n, min(k, kcand)), dtype=int)
        for i in range(n):
            cand = idx_obs_cand[i]
            d2 = np.einsum(
                "ij,ij->i",
                obs_hist[cand] - obs_hist[i],
                obs_hist[cand] - obs_hist[i],
            )
            order = np.argsort(d2)
            take = min(k, order.size)
            idx_obs_hist[i, :take] = cand[order[:take]]
            if take < k:
                idx_obs_hist[i, take:k] = idx_obs[i, : k - take]
    else:
        idx_obs_hist = _knn(obs_hist, k)

    n_classes = int(np.max(y) + 1)
    observed_classes = int(np.unique(y).size)
    k_eff = min(max(1, int(k)), max(n - 1, 1))
    _warn_discrete_posterior_risk(
        "CMI",
        n_classes=n_classes,
        observed_classes=observed_classes,
        k=k_eff,
        alpha=alpha,
        n_samples=n,
        obs_dim=obs_arr.shape[1],
        hidden_dim=hist_arr.shape[1],
    )

    def _posterior(indices):
        counts = np.zeros((indices.shape[0], n_classes), dtype=float)
        rows, _ = np.indices(indices.shape)
        np.add.at(counts, (rows, y[indices]), 1.0)
        return (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * n_classes)

    p_a_o = _posterior(idx_obs[:, :k])
    p_a_oh = _posterior(idx_obs_hist[:, :k])

    def _row_entropy(P):
        P = np.clip(P, 1e-12, 1.0)
        return -np.sum(P * np.log(P), axis=1)

    def _row_kl(P, Q):
        P, Q = np.clip(P, 1e-12, 1.0), np.clip(Q, 1e-12, 1.0)
        return np.sum(P * (np.log(P) - np.log(Q)), axis=1)

    H_o = _row_entropy(p_a_o)
    H_oh = _row_entropy(p_a_oh)

    cmi_kl = float(np.mean(_row_kl(p_a_oh, p_a_o)))
    cmi_ent = float(np.mean(np.maximum(H_o - H_oh, 0.0) if nonneg else H_o - H_oh))

    H_A_given_O = float(np.mean(H_o))
    gc = np.bincount(y, minlength=n_classes).astype(float)
    pA = np.clip(gc / max(gc.sum(), 1e-12), 1e-12, 1.0)
    H_A = float(-np.sum(pA * np.log(pA)))

    eps = 1e-12
    R_kl = cmi_kl / max(H_A_given_O, eps) if H_A_given_O > eps else np.nan

    inv = 1.0 / np.log(2.0) if bits else 1.0
    return cmi_kl * inv, {
        "cmi_bits": cmi_kl * inv if bits else cmi_kl,
        "cmi_ent_bits": cmi_ent * inv if bits else cmi_ent,
        "H_A_given_O": H_A_given_O * inv,
        "H_A": H_A * inv,
        "R_cond_from_kl": R_kl,
        "posterior_alpha": float(alpha),
        "n_classes": int(n_classes),
        "observed_classes": int(observed_classes),
        "posterior_pseudocount_mass": float(alpha) * int(n_classes),
    }


# ----------------------------------------------------------------------
# Continuous-action CMI: Frenzel-Pompe KSG with rank-transform pre-processing
# ----------------------------------------------------------------------


def _kl_entropy(X, k, workers=-1):
    """Kozachenko-Leonenko differential entropy estimator (in nats).

    Reliable only for low-dimensional X (d <= ~10); higher dimensions
    require KSG MI differences.
    """
    n, d = X.shape
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return 0.0

    tree = cKDTree(X)
    eps = tree.query(X, k=[k_eff + 1], p=np.inf, workers=workers)[0][:, 0]
    eps = np.maximum(eps, 1e-30)
    return -digamma(k_eff) + digamma(n) + d * np.mean(np.log(2.0 * eps))


def _mi_ksg(X, Y, k, workers=-1):
    """KSG-1 mutual information estimator (Kraskov et al. 2004), in nats."""
    n = X.shape[0]
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return 0.0

    XY = np.concatenate([X, Y], axis=1)
    tree_xy = cKDTree(XY)
    eps = tree_xy.query(XY, k=[k_eff + 1], p=np.inf, workers=workers)[0][:, 0]
    eps *= 0.999999999

    tree_x = cKDTree(X)
    tree_y = cKDTree(Y)
    n_x = tree_x.query_ball_point(X, r=eps, p=np.inf, workers=workers, return_length=True)
    n_y = tree_y.query_ball_point(Y, r=eps, p=np.inf, workers=workers, return_length=True)

    return digamma(k_eff) + digamma(n) - np.mean(digamma(n_x) + digamma(n_y))


def _cmi_frenzel_pompe(X, Y, Z, k, workers=-1):
    """Frenzel-Pompe conditional MI I(X; Y | Z), in nats.

    Uses a single eps from the joint (X,Y,Z) space then counts neighbours
    in each marginal subspace. The shared eps makes dimension-dependent
    biases cancel by construction.
    """
    n = X.shape[0]
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return 0.0, np.zeros(n), np.zeros(n), np.zeros(n)

    XYZ = np.concatenate([X, Y, Z], axis=1)
    XZ = np.concatenate([X, Z], axis=1)
    YZ = np.concatenate([Y, Z], axis=1)

    tree_xyz = cKDTree(XYZ)
    eps = tree_xyz.query(XYZ, k=[k_eff + 1], p=np.inf, workers=workers)[0][:, 0]
    eps *= 0.999999999

    tree_xz = cKDTree(XZ)
    tree_yz = cKDTree(YZ)
    tree_z = cKDTree(Z)

    k_xz = np.asarray(
        tree_xz.query_ball_point(XZ, r=eps, p=np.inf, workers=workers, return_length=True),
        dtype=np.float64)
    k_yz = np.asarray(
        tree_yz.query_ball_point(YZ, r=eps, p=np.inf, workers=workers, return_length=True),
        dtype=np.float64)
    k_z = np.asarray(
        tree_z.query_ball_point(Z, r=eps, p=np.inf, workers=workers, return_length=True),
        dtype=np.float64)

    cmi = digamma(k_eff) + np.mean(digamma(k_z) - digamma(k_xz) - digamma(k_yz))
    return cmi, k_xz, k_yz, k_z


def _cmi_continuous_actions(obs, hidden, actions, k=25, bits=True, nonneg=True,
                            add_jitter=1e-6, max_dim=50, transform="ranks",
                            workers=-1):
    """Estimate I(H; A | O) for continuous actions via Frenzel-Pompe KSG.

    Pre-processes inputs by per-block PCA (when above ``max_dim``), rank
    transformation, and small scale-aware jitter, all of which leave the true
    CMI invariant but stabilise the estimator. ``H(A|O)`` is recovered from
    ``H(A) - I(A;O)`` to support ``R_cond = I(H;A|O) / H(A|O)``.
    """
    obs_arr = _ensure_2d(obs).astype(np.float64, copy=True)
    hist_arr = _ensure_2d(hidden).astype(np.float64, copy=True)
    action_arr = _ensure_2d(actions).astype(np.float64, copy=True)
    n = obs_arr.shape[0]

    if obs_arr.shape[1] > max_dim:
        obs_arr = PCA(n_components=min(max_dim, n - 1), random_state=42).fit_transform(obs_arr)
    if hist_arr.shape[1] > max_dim:
        hist_arr = PCA(n_components=min(max_dim, n - 1), random_state=42).fit_transform(hist_arr)
    if action_arr.shape[1] > max_dim:
        action_arr = PCA(n_components=min(max_dim, n - 1), random_state=42).fit_transform(action_arr)

    d_obs, d_hist, d_action = obs_arr.shape[1], hist_arr.shape[1], action_arr.shape[1]

    # H(A) needs un-ranked data: differential entropy is NOT invariant under
    # rank transforms (on ranks H(A) ~ d_A * log(n), which is meaningless).
    A_for_entropy = action_arr.copy()
    A_for_entropy -= A_for_entropy.mean(axis=0, keepdims=True)
    a_std = A_for_entropy.std(axis=0, keepdims=True)
    a_std = np.where(a_std < 1e-12, 1.0, a_std)
    A_for_entropy /= a_std
    rng_ent = np.random.default_rng(43)
    A_for_entropy += add_jitter * rng_ent.random(A_for_entropy.shape)

    array = np.concatenate([obs_arr, hist_arr, action_arr], axis=1).T

    if transform == "ranks":
        array = array.argsort(axis=1).astype(np.float64)
    elif transform == "standardize":
        array = array.astype(np.float64)
        array -= array.mean(axis=1, keepdims=True)
        std = array.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        array /= std

    rng = np.random.default_rng(42)
    feat_std = array.std(axis=1, keepdims=True)
    feat_std = np.maximum(feat_std, 1.0)
    array += add_jitter * feat_std * rng.random(array.shape)

    array = array.T
    Z_data = array[:, :d_obs]
    X_data = array[:, d_obs:d_obs + d_hist]
    Y_data = array[:, d_obs + d_hist:d_obs + d_hist + d_action]

    k_eff = min(k, n - 1)
    ln2 = np.log(2.0)

    cmi_nats, k_xz, k_yz, k_z = _cmi_frenzel_pompe(
        X_data, Y_data, Z_data, k_eff, workers=workers
    )
    cmi_raw = cmi_nats / ln2 if bits else cmi_nats
    cmi_val = max(0.0, cmi_raw) if nonneg else cmi_raw

    h_A_nats = _kl_entropy(A_for_entropy, k_eff, workers=workers)
    mi_AO_nats = _mi_ksg(Y_data, Z_data, k_eff, workers=workers)
    h_A_given_O_nats = h_A_nats - mi_AO_nats

    h_A = h_A_nats / ln2 if bits else h_A_nats
    mi_AO = mi_AO_nats / ln2 if bits else mi_AO_nats
    H_A_given_O = h_A_given_O_nats / ln2 if bits else h_A_given_O_nats

    H_A_given_O_clean = max(0.0, H_A_given_O)
    denom_tol = max(0.1, 1e-3 * abs(h_A))
    if H_A_given_O_clean > denom_tol:
        R_cond = float(np.clip(cmi_val / H_A_given_O_clean, 0.0, 1.0))
    else:
        logger.info(
            f"  H(A|O) = {H_A_given_O:.4f} bits < tol={denom_tol:.4f} — "
            f"policy near-deterministic given O, R_cond undefined "
            f"(CMI={cmi_val:.4f}, H(A)={h_A:.4f}, n={n}, d_A={d_action})"
        )
        R_cond = np.nan

    if mi_AO < -0.1:
        logger.warning(
            f"  I(A;O) = {mi_AO:.4f} is notably negative — "
            f"possible estimation issue (n={n}, d_O={d_obs}, d_A={d_action})"
        )

    return cmi_val, {
        "cmi_bits": cmi_val,
        "H_A_given_O": H_A_given_O,
        "R_cond_from_kl": R_cond,
        "path": "continuous_actions_ksg",
        "cmi_raw": cmi_raw,
        "H_A": h_A,
        "I_A_O": mi_AO,
        "n_samples": n,
        "k_eff": k_eff,
        "joint_dim": d_obs + d_hist + d_action,
    }


def cmi_mixed(obs, hidden, actions, *, k=25, bits=True, alpha=0.5,
              cand_mult=5, use_nested=True, nonneg=True,
              force_continuous_A=False, add_jitter=1e-9, kd_workers=1):
    """Dispatch I(H; A | O) to the appropriate estimator based on action type.

    Continuous actions use the Frenzel-Pompe KSG path; discrete actions use
    the nested-kNN posterior KL path.
    """
    obs_arr = np.asarray(obs)
    hist_arr = np.asarray(hidden)
    A_raw = np.asarray(actions)

    if force_continuous_A or not _is_discrete(A_raw):
        return _cmi_continuous_actions(obs_arr, hist_arr, A_raw, k=k, bits=bits, nonneg=nonneg,
                                       add_jitter=add_jitter, workers=kd_workers)

    if add_jitter:
        rng = np.random.default_rng(42)
        obs_arr = obs_arr.astype(float, copy=True) + rng.standard_normal(obs_arr.shape) * add_jitter
        hist_arr = hist_arr.astype(float, copy=True) + rng.standard_normal(hist_arr.shape) * add_jitter
    return cmi_knn(obs_arr, hist_arr, A_raw, k=k, bits=bits, alpha=alpha,
                   cand_mult=cand_mult, use_nested=use_nested, nonneg=nonneg)


# ----------------------------------------------------------------------
# Marginal MI: I(X ; A) via Ross 2014 (continuous X, discrete A) and KSG-1
# (continuous X, continuous Y). Used by OAR.
# ----------------------------------------------------------------------


def mi_cd(X, A, k=3, bits=True, metric="euclidean"):
    """MI(X; A) for continuous X and discrete A via Ross (2014).

    Returns ``(mi, extras)`` with ``H_A`` and ``R_uncond = mi / H(A)``.
    """
    X = _ensure_2d(X).astype(float, copy=True)
    y = _ensure_1d_int(A)
    n_samples = X.shape[0]

    X = scale(X, with_mean=False, copy=False)
    means = np.maximum(1, np.mean(np.abs(X), axis=0))
    rng = np.random.default_rng(42)
    X += 1e-10 * means * rng.standard_normal(X.shape)

    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    nn = NearestNeighbors(metric=metric)

    for label in np.unique(y):
        mask = y == label
        count = np.sum(mask)
        if count > 1:
            k_eff = min(k, count - 1)
            nn.set_params(n_neighbors=k_eff)
            nn.fit(X[mask])
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k_eff
        label_counts[mask] = count

    mask = label_counts > 1
    n_valid = int(np.sum(mask))
    if n_valid <= 1:
        logger.warning("Not enough valid samples for MI estimation. Returning 0.")
        return 0.0, {"mi": 0.0, "H_A": np.nan, "R_uncond": 0.0, "n_used": n_valid}

    label_counts = label_counts[mask]
    k_all = k_all[mask]
    radius = radius[mask]
    X_valid = X[mask]

    kd = KDTree(X_valid, metric=metric)
    m_all = np.array(
        kd.query_radius(X_valid, radius, count_only=True, return_distance=False),
        dtype=float,
    )

    mi_nats = float(
        digamma(n_valid)
        + np.mean(digamma(k_all))
        - np.mean(digamma(label_counts))
        - np.mean(digamma(m_all))
    )
    mi_nats = max(0.0, mi_nats)

    y_valid = y[mask]
    gc_used = np.bincount(y_valid, minlength=int(np.max(y_valid) + 1)).astype(float)
    H_A_used_nats = _entropy_from_counts(gc_used, bits=False)

    inv = 1.0 / np.log(2.0) if bits else 1.0
    return mi_nats * inv, {
        "mi": mi_nats * inv,
        "H_A": H_A_used_nats * inv,
        "R_uncond": mi_nats / max(H_A_used_nats, 1e-12) if H_A_used_nats > 1e-12 else np.nan,
    }


def mi_cc(X, Y, k=3, bits=True, metric="euclidean"):
    """MI(X; Y) for continuous X and Y via KSG-1 (Kraskov et al. 2004).

    Returns ``(mi, extras)`` with ``H_Y`` (KL estimate) and
    ``R_uncond = mi / H(Y)``.
    """
    X = _ensure_2d(X).astype(float, copy=True)
    Y = _ensure_2d(Y).astype(float, copy=True)
    n_samples = X.shape[0]

    X = scale(X, with_mean=False, copy=False)
    Y = scale(Y, with_mean=False, copy=False)
    rng = np.random.default_rng(42)
    X += 1e-10 * np.maximum(1, np.mean(np.abs(X), axis=0)) * rng.standard_normal(X.shape)
    Y += 1e-10 * np.maximum(1, np.mean(np.abs(Y), axis=0)) * rng.standard_normal(Y.shape)

    XY = np.hstack((X, Y))

    nn = NearestNeighbors(metric=metric, n_neighbors=k)
    nn.fit(XY)
    radius = np.nextafter(nn.kneighbors()[0][:, -1], 0)

    kd_x = KDTree(X, metric=metric)
    nx = np.array(kd_x.query_radius(X, radius, count_only=True, return_distance=False)) - 1.0
    kd_y = KDTree(Y, metric=metric)
    ny = np.array(kd_y.query_radius(Y, radius, count_only=True, return_distance=False)) - 1.0

    mi_nats = float(
        digamma(n_samples)
        + digamma(k)
        - np.mean(digamma(nx + 1))
        - np.mean(digamma(ny + 1))
    )
    mi_nats = max(0.0, mi_nats)

    inv = 1.0 / np.log(2.0) if bits else 1.0
    mi_out = mi_nats * inv

    d = Y.shape[1]
    nn_y = NearestNeighbors(metric=metric, n_neighbors=k)
    nn_y.fit(Y)
    rho = np.maximum(nn_y.kneighbors()[0][:, -1], 1e-10)
    log_Vd = (d / 2.0) * np.log(np.pi) - float(scipy.special.loggamma(d / 2.0 + 1))
    H_Y_nats = float(d * np.mean(np.log(rho)) + log_Vd - digamma(k) + digamma(n_samples))
    H_Y = H_Y_nats * inv

    return mi_out, {
        "mi": mi_out,
        "H_Y": H_Y,
        "R_uncond": mi_nats / max(H_Y_nats, 1e-12) if H_Y_nats > 1e-12 else np.nan,
    }
