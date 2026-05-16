"""Summary statistics (rliable bootstrap CIs) and LaTeX table export.

The paper reports the min-max normalised IQM with 95% stratified bootstrap
CIs (Agarwal et al. 2021); rliable is the canonical implementation. The
LaTeX table produced by :func:`generate_latex_table` mirrors Tbl. 1 in the
paper: one row per ``(env, alg)``, FF runs use the observation/OA-history
variants, RNN runs use the hidden-state variants, and values exceeding the
permutation null are bolded.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _import_rliable():
    """Lazy import — keeps the package usable if rliable's deps are broken."""
    from rliable import library as rly
    from rliable import metrics as rly_metrics
    return rly, rly_metrics


# Each tuple: (LaTeX header, FF column, RNN column, FF null column, RNN null column).
# Used both for the LaTeX export and the columns considered for the
# unified normalised report.
NORM_COLS = [
    (r"$\mathrm{OAR}^{\mathrm{norm}}$",  "oarR_max",             "oarR_max",              "oarR_max_null",             "oarR_max_null"),
    (r"$\mathrm{HAR}^{\mathrm{norm}}$",   "harRcond_ohist_max",   "harRcond_hidden_max",   "harRcond_ohist_max_null",   "harRcond_hidden_max_null"),
    (r"$\mathrm{PIF}^{\mathrm{norm}}$",   "pifOARcond_ohist_max", "pifRcond_hidden_max",   "pifOARcond_ohist_max_null", "pifRcond_hidden_max_null"),
    (r"$\mathrm{AA}^{\mathrm{norm}}$",    "aaRcond_max",          "aaRcond_max",           "aaRcond_max_null",          "aaRcond_max_null"),
    (r"$\mathrm{DAI}^{\mathrm{norm}}$",   "daiOARcond_ohist_max", "daiRcond_hidden_max",   "daiOARcond_ohist_max_null", "daiRcond_hidden_max_null"),
]

# All (raw + normalised) columns shown in the console summary.
SUMMARY_COLS = [
    "oar_max", "oarR_max",
    "har_ohist_max", "harRcond_ohist_max",
    "har_hidden_max", "harRcond_hidden_max",
    "pif_ohist_max", "pifRcond_ohist_max",
    "pif_hidden_max", "pifRcond_hidden_max",
    "pifOA_ohist_max", "pifOARcond_ohist_max",
    "aa_max", "aaRcond_max",
    "daiOA_ohist_max", "daiOARcond_ohist_max",
    "dai_hidden_max", "daiRcond_hidden_max",
]


def _is_rnn_alg(alg_name):
    return "RNN" in alg_name.upper()


def _latex_escape(s):
    return str(s).replace("_", r"\_")


def rliable_mean_ci(vals, reps=5000):
    """Return ``(mean, ci_lower, ci_upper)`` via rliable stratified bootstrap."""
    rly, rly_metrics = _import_rliable()
    vals = np.asarray(vals, dtype=float).reshape(-1, 1)
    score_dict = {"m": vals}

    def agg_fn(x):
        return rly_metrics.aggregate_mean(x[:, 0:1])

    mean_scores, cis = rly.get_interval_estimates(score_dict, agg_fn, reps=reps)
    mean_val = mean_scores["m"]
    ci_vals = cis["m"]
    if isinstance(mean_val, np.ndarray):
        mean_val = float(mean_val.flat[0])
    if isinstance(ci_vals, np.ndarray):
        ci_flat = ci_vals.flatten()
        return float(mean_val), float(ci_flat[0]), float(ci_flat[1])
    return float(mean_val), float(ci_vals[0]), float(ci_vals[1])


def print_summary(df, rliable_reps=5000):
    """Print and return a per-(env, alg) summary table with bootstrap CIs."""
    rows = []
    print(f"\n{'='*60}")
    print("METRICS SUMMARY (Mean with 95% CI via rliable)")
    print(f"{'='*60}")

    for (env, alg), grp in df.groupby(["env_name", "alg"]):
        print(f"\n  {env} / {alg}  ({len(grp)} runs)")
        for col in SUMMARY_COLS:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna().values

            null_col = col + "_null"
            null_mean_val = np.nan
            null_suffix = ""
            if null_col in grp.columns:
                null_vals = grp[null_col].dropna().values
                if len(null_vals) > 0:
                    null_mean_val = float(np.nanmean(null_vals))
                    null_suffix = f"  (null={null_mean_val:.4f})"

            if len(vals) > 1:
                try:
                    mean_val, ci_lo, ci_hi = rliable_mean_ci(vals, reps=rliable_reps)
                    print(f"    {col:30s}: {mean_val:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]{null_suffix}")
                    rows.append({
                        "env_name": env, "alg": alg, "metric": col,
                        "mean": mean_val, "ci_lower": ci_lo, "ci_upper": ci_hi,
                        "null_mean": null_mean_val, "n_runs": len(grp),
                    })
                except Exception:
                    m, s = float(vals.mean()), float(vals.std())
                    print(f"    {col:30s}: {m:.4f} +/- {s:.4f} (CI failed){null_suffix}")
                    rows.append({
                        "env_name": env, "alg": alg, "metric": col,
                        "mean": m, "null_mean": null_mean_val, "n_runs": len(grp),
                    })
            elif len(vals) == 1:
                print(f"    {col:30s}: {vals[0]:.4f} [single run]{null_suffix}")
                rows.append({
                    "env_name": env, "alg": alg, "metric": col,
                    "mean": float(vals[0]), "null_mean": null_mean_val, "n_runs": 1,
                })

    print(f"{'='*60}")
    return rows


def generate_latex_table(df, output_path, rliable_reps=5000, norm_cols=None):
    """Write a LaTeX table of normalised diagnostics (paper Tbl. 1 format).

    For each ``(env, alg)`` group, FF policies use the observation/OA-history
    variants and RNN policies use the hidden-state variants. Values that
    exceed the permutation-null baseline are bolded; missing R_cond values
    (near-deterministic policies, ``H(A|O) ≈ 0``) are marked with †.
    """
    if norm_cols is None:
        norm_cols = NORM_COLS

    all_raw = set()
    for tup in norm_cols:
        all_raw.add(tup[1])
        all_raw.add(tup[2])
        if len(tup) >= 5:
            all_raw.add(tup[3])
            all_raw.add(tup[4])

    active = []
    for tup in norm_cols:
        latex_hdr, ff_c, rnn_c = tup[0], tup[1], tup[2]
        ff_null = tup[3] if len(tup) >= 5 else None
        rnn_null = tup[4] if len(tup) >= 5 else None
        ff_ok = ff_c in df.columns and df[ff_c].notna().any()
        rnn_ok = rnn_c in df.columns and df[rnn_c].notna().any()
        if ff_ok or rnn_ok:
            active.append((latex_hdr, ff_c, rnn_c, ff_null, rnn_null))

    if not active:
        logger.warning("No normalised metric columns with data — skipping LaTeX table.")
        return None

    n_metric_cols = len(active)

    stats = {}
    for (env, alg), grp in df.groupby(["env_name", "alg"]):
        for col in all_raw:
            if col not in grp.columns:
                stats[(env, alg, col)] = None
                continue
            vals = grp[col].dropna().values
            if len(vals) > 1:
                try:
                    mean_val, ci_lo, ci_hi = rliable_mean_ci(vals, reps=rliable_reps)
                    stats[(env, alg, col)] = (mean_val, ci_lo, ci_hi)
                except Exception:
                    stats[(env, alg, col)] = None
            elif len(vals) == 1:
                stats[(env, alg, col)] = (float(vals[0]), None, None)
            else:
                stats[(env, alg, col)] = None

    def _norm_to_raw_col(col):
        if "Rcond_" in col:
            return col.replace("Rcond_", "_")
        if col.startswith("oarR_"):
            return "oar_" + col[5:]
        return None

    def _raw_has_data(env, alg, norm_col):
        raw_col = _norm_to_raw_col(norm_col)
        if raw_col is None or raw_col not in df.columns:
            return False
        mask = (df["env_name"] == env) & (df["alg"] == alg)
        return df.loc[mask, raw_col].notna().any()

    def _fmt_cell(entry, null_entry=None, dagger=False):
        if entry is None:
            return r" & ---$^{\dagger}$" if dagger else r" & ---"
        mean_val, ci_lo, ci_hi = entry
        if null_entry is not None and null_entry[0] is not None and np.isfinite(null_entry[0]):
            bold = mean_val > null_entry[0]
        else:
            bold = mean_val > 0.05
        if ci_lo is not None and ci_hi is not None:
            num = f"{mean_val:.2f}"
            ci = f"[{ci_lo:.2f},\\,{ci_hi:.2f}]"
            if bold:
                return f" & $\\mathbf{{{num}}}$ {{\\tiny $\\mathbf{{{ci}}}$}}"
            return f" & ${num}$ {{\\tiny ${ci}$}}"
        if bold:
            return f" & $\\mathbf{{{mean_val:.2f}}}$"
        return f" & ${mean_val:.2f}$"

    has_dagger = False
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        "%%CAPTION_PLACEHOLDER%%",
        r"\label{tab:norm_metrics}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + ("ll" + "c" * n_metric_cols) + "}",
        r"\toprule",
    ]

    header = "Environment & Algorithm"
    for latex_hdr, *_ in active:
        header += f" & {latex_hdr}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    envs_seen = []
    env_algs = {}
    for (env, alg) in sorted({(e, a) for e, a, _ in stats.keys()}):
        if env not in env_algs:
            env_algs[env] = []
            envs_seen.append(env)
        if alg not in env_algs[env]:
            env_algs[env].append(alg)

    for ei, env in enumerate(envs_seen):
        algs = env_algs[env]
        for ai, alg in enumerate(algs):
            if ai == 0 and len(algs) > 1:
                env_cell = r"\multirow{" + str(len(algs)) + r"}{*}{" + _latex_escape(env) + "}"
            elif ai == 0:
                env_cell = _latex_escape(env)
            else:
                env_cell = ""

            is_rnn = _is_rnn_alg(alg)
            row_str = f"{env_cell} & {_latex_escape(alg)}"
            for _, ff_c, rnn_c, ff_null, rnn_null in active:
                col = rnn_c if is_rnn else ff_c
                null_col = (rnn_null if is_rnn else ff_null) if ff_null else None
                entry = stats.get((env, alg, col))
                null_entry = stats.get((env, alg, null_col)) if null_col else None
                dagger = entry is None and _raw_has_data(env, alg, col)
                if dagger:
                    has_dagger = True
                row_str += _fmt_cell(entry, null_entry, dagger=dagger)
            row_str += r" \\"
            lines.append(row_str)

        if ei < len(envs_seen) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")

    caption = (r"\caption{Normalised diagnostic metrics (mean with 95\% stratified bootstrap CI). "
               r"Values exceeding the permutation-null baseline are \textbf{bolded}. "
               r"For FF policies, $\mathrm{HAR}^{\mathrm{norm}}$ uses an observation-history window "
               r"and $\mathrm{PIF}^{\mathrm{norm}}$/$\mathrm{DAI}^{\mathrm{norm}}$ use an observation-action history window; "
               r"for RNN policies these use the hidden state. "
               r"$\mathrm{AA}^{\mathrm{norm}}$ is architecture-independent (uses current observations).")
    if has_dagger:
        caption += (r" $^{\dagger}$The normalised metric $R = I/H(A|O)$ is undefined because "
                    r"the policy is near-deterministic given observations ($H(A|O) \approx 0$); "
                    r"the unnormalised MI (in bits) remains well-defined.")
    caption += "}"
    lines = [caption if line == "%%CAPTION_PLACEHOLDER%%" else line for line in lines]
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)

    tex_path = output_path.replace(".csv", "_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    logger.info(f"Saved LaTeX table to {tex_path}")

    print(f"\n{'='*60}\nLATEX TABLE\n{'='*60}\n{latex_str}\n{'='*60}")
    return latex_str
