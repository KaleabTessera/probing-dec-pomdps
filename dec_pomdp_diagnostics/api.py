"""High-level user-facing API: trajectories in, paper Table 1 out.

Most users don't want the W&B collection pipeline or the packed-format
unpacker — they have their own MARL trajectories and want the four
Decision-Rule answers per scenario:

* Do agents benefit from memory?           (Diag 1, Δ_Mem Wilcoxon test)
* Do agents use hidden teammate info?      (Diag 4, PIF^norm > null)
* Does synchronous coordination emerge?    (Diag 5, AA^norm > null)
* Does temporal coordination emerge?       (Diag 6, DAI^norm > null)

Plus the underlying memory diagnostic:

* History dependence > null                (Diag 2, HAR^norm > null)

Pipeline:

    >>> data = UserData(
    ...     observations={"agent_0": obs0, ...},   # (N, obs_dim)
    ...     actions={"agent_0": act0, ...},        # (N,) discrete or (N, act_dim) continuous
    ...     timesteps={"agent_0": ts0, ...},       # (N,) step within episode
    ...     episode_ids={"agent_0": ep0, ...},     # (N,) parallel-env / episode id
    ...     hidden_states={"agent_0": h0, ...},    # optional, (N, hidden_dim) for RNN
    ... )
    >>> result = compute_diagnostics(data, history_k=3, null_reps=5)
    >>> result.metrics      # dict of diagnostic -> raw bits + normalised + null
    >>> result.flags        # dict of Decision Rule -> bool

For the full Table 1 across multiple scenarios and algorithms, build a list
of ``ScenarioRun`` and pass them to :func:`build_paper_table`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .data import _actions_are_continuous
from .estimators import _ensure_2d
from .pipeline import process_run

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# User-facing data container
# ----------------------------------------------------------------------


@dataclass
class UserData:
    """Per-agent trajectories from a single (env, alg, seed) run.

    Every field except ``hidden_states`` is required. All per-agent arrays
    must share the same length ``N`` for a given agent. Different agents may
    have different ``N`` provided their ``(episode_ids, timesteps)`` keys can
    be aligned.

    Required shapes (per agent ``a``)
    ----------------------------------
    observations[a] : ndarray, shape ``(N_a, obs_dim_a)``
    actions[a]      : ndarray, shape ``(N_a,)`` (discrete int) or
                                       ``(N_a, act_dim_a)`` (continuous float)
    timesteps[a]    : ndarray, shape ``(N_a,)``, int — step within episode
    episode_ids[a]  : ndarray, shape ``(N_a,)``, int — episode / parallel-env id
    hidden_states[a]: ndarray, shape ``(N_a, hidden_dim_a)`` — optional, RNN only

    Optional metadata
    -----------------
    env_name : str — environment / suite label (e.g. ``"MPE"``). Used as the
                     column in Table 1.
    alg_name : str — algorithm label; must contain ``"RNN"`` for recurrent
                     policies (Decision Rule 1 needs this).
    seed     : int — used for null-baseline RNG seeding.
    scenario_name : str, optional — unique scenario id within ``env_name``
                     (e.g. ``"simple_reference_v3"``). If ``None``, the
                     scenario id defaults to ``env_name`` (i.e. each
                     ``env_name`` is treated as a single scenario). Used by
                     :func:`build_paper_table` for the per-scenario verdict.
    """

    observations: Dict[str, np.ndarray]
    actions: Dict[str, np.ndarray]
    timesteps: Dict[str, np.ndarray]
    episode_ids: Dict[str, np.ndarray]
    hidden_states: Optional[Dict[str, np.ndarray]] = None
    env_name: str = "env"
    alg_name: str = "alg"
    seed: int = 0
    scenario_name: Optional[str] = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        agents = sorted(self.observations.keys())
        if not agents:
            raise ValueError("UserData.observations must contain at least one agent.")

        for field_name in ("actions", "timesteps", "episode_ids"):
            field_val = getattr(self, field_name)
            if set(field_val.keys()) != set(agents):
                raise ValueError(
                    f"UserData.{field_name} keys {sorted(field_val.keys())} must "
                    f"match observations keys {agents}."
                )

        if self.hidden_states is not None:
            extra = set(self.hidden_states.keys()) - set(agents)
            missing = set(agents) - set(self.hidden_states.keys())
            if extra or missing:
                raise ValueError(
                    f"UserData.hidden_states keys must match observations keys. "
                    f"Extra={sorted(extra)}, missing={sorted(missing)}."
                )

        for a in agents:
            obs = np.asarray(self.observations[a])
            act = np.asarray(self.actions[a])
            ts = np.asarray(self.timesteps[a])
            ep = np.asarray(self.episode_ids[a])

            if obs.ndim != 2:
                raise ValueError(
                    f"observations[{a!r}] must be 2D (N, obs_dim); got shape {obs.shape}."
                )
            n = obs.shape[0]
            if n < 2:
                raise ValueError(f"observations[{a!r}] has only {n} rows; need ≥ 2.")

            if act.ndim not in (1, 2):
                raise ValueError(
                    f"actions[{a!r}] must be 1D (discrete) or 2D (continuous); got {act.shape}."
                )
            if act.shape[0] != n:
                raise ValueError(
                    f"actions[{a!r}] length {act.shape[0]} != observations length {n}."
                )

            if ts.ndim != 1 or ts.shape[0] != n:
                raise ValueError(
                    f"timesteps[{a!r}] must be 1D length {n}; got shape {ts.shape}."
                )
            if not np.issubdtype(ts.dtype, np.integer):
                raise ValueError(f"timesteps[{a!r}] must be integer dtype; got {ts.dtype}.")

            if ep.ndim != 1 or ep.shape[0] != n:
                raise ValueError(
                    f"episode_ids[{a!r}] must be 1D length {n}; got shape {ep.shape}."
                )
            if not np.issubdtype(ep.dtype, np.integer):
                raise ValueError(f"episode_ids[{a!r}] must be integer dtype; got {ep.dtype}.")

            if self.hidden_states is not None:
                h = np.asarray(self.hidden_states[a])
                if h.ndim != 2 or h.shape[0] != n:
                    raise ValueError(
                        f"hidden_states[{a!r}] must be 2D (N, hidden_dim) with N={n}; "
                        f"got shape {h.shape}."
                    )

    def to_run_dict(self) -> dict:
        """Convert to the internal run-dict format consumed by ``process_run``."""
        per_agent = {}
        has_hidden = self.hidden_states is not None

        for a in sorted(self.observations.keys()):
            obs = _ensure_2d(np.asarray(self.observations[a]))
            act = np.asarray(self.actions[a])
            n = obs.shape[0]
            if has_hidden:
                h = _ensure_2d(np.asarray(self.hidden_states[a]))
            else:
                h = np.zeros((n, 1), dtype=np.float32)

            per_agent[str(a)] = {
                "states": obs,
                "actions": act,
                "hidden": h,
                "has_hidden": has_hidden,
                "agent_ids": np.full(n, hash(a) & 0xFFFF, dtype=np.int64),
                "timesteps": np.asarray(self.timesteps[a], dtype=np.int64),
                "episode_ids": np.asarray(self.episode_ids[a], dtype=np.int64),
            }

        return {
            "run_id": f"{self.env_name}/{self.alg_name}/seed{self.seed}",
            "run_name": f"{self.env_name}/{self.alg_name}",
            "map_name": self.env_name,
            "alg_name": self.alg_name,
            "config": {"SEED": self.seed},
            "per_agent": per_agent,
            "scenario_name": self.scenario_name,
        }


# ----------------------------------------------------------------------
# Single-run result wrapper
# ----------------------------------------------------------------------


_FLAG_DESC = {
    "history_dependence": "Do actions depend on history?  (Diag 2: HAR^norm > permutation null)",
    "uses_hidden_teammate_info": "Does teammate information help predict another agent's actions?   (Diag 4: PIF^norm > null)",
    "synchronous_coordination":  "Does synchronous coordination emerge?  (Diag 5: AA^norm > null)",
    "temporal_coordination":     "Does temporal coordination emerge?  (Diag 6: DAI^norm > null)",
}


@dataclass
class DiagnosticResult:
    """Per-run diagnostic output: raw + normalised metrics, null baselines, flags.

    Attributes
    ----------
    metrics : dict
        Raw and normalised metric values, and the corresponding null means.
        Keys follow the column convention used in the per-run CSV
        (e.g. ``"har_ohist_max"``, ``"harRcond_ohist_max"``,
        ``"harRcond_ohist_max_null"``).
    flags : dict[str, bool]
        Boolean Decision-Rule flags (see ``_FLAG_DESC`` for descriptions).
    raw_row : dict
        The full per-run row as produced by ``process_run``.
    """

    metrics: Dict[str, float]
    flags: Dict[str, bool]
    raw_row: Dict[str, object]

    def describe(self) -> str:
        lines = [f"Run {self.raw_row.get('run_id')}:"]
        
        # Determine if RNN or FF
        alg_name = self.raw_row.get("alg", "")
        is_rnn = "RNN" in alg_name.upper()
        
        # Map flags to their metric column pairs (value, null)
        flag_to_cols = {
            "history_dependence": (
                "harRcond_hidden_max" if is_rnn else "harRcond_ohist_max",
                ("harRcond_hidden_max_null" if is_rnn else "harRcond_ohist_max_null"),
            ),
            "uses_hidden_teammate_info": (
                "pifRcond_hidden_max" if is_rnn else "pifOARcond_ohist_max",
                ("pifRcond_hidden_max_null" if is_rnn else "pifOARcond_ohist_max_null"),
            ),
            "synchronous_coordination": (
                "aaRcond_max",
                "aaRcond_max_null",
            ),
            "temporal_coordination": (
                "daiRcond_hidden_max" if is_rnn else "daiOARcond_ohist_max",
                ("daiRcond_hidden_max_null" if is_rnn else "daiOARcond_ohist_max_null"),
            ),
        }
        
        for flag, desc in _FLAG_DESC.items():
            mark = "✓" if self.flags.get(flag, False) else "✗"
            
            # Get metric values if available
            if flag in flag_to_cols:
                val_col, null_col = flag_to_cols[flag]
                val = self.metrics.get(val_col)
                null_val = self.metrics.get(null_col)
                
                if val is not None and null_val is not None:
                    # Extract the variable names from description (e.g., "HAR^norm", "HAR^null")
                    # For now, use generic names based on flag type
                    if flag == "history_dependence":
                        metric_str = f"HAR^norm={val:.4f}, HAR^null={null_val:.4f}"
                    elif flag == "uses_hidden_teammate_info":
                        metric_str = f"PIF^norm={val:.4f}, PIF^null={null_val:.4f}"
                    elif flag == "synchronous_coordination":
                        metric_str = f"AA^norm={val:.4f}, AA^null={null_val:.4f}"
                    elif flag == "temporal_coordination":
                        metric_str = f"DAI^norm={val:.4f}, DAI^null={null_val:.4f}"
                    else:
                        metric_str = ""
                    
                    # Extract the part in parentheses from the description
                    if "(" in desc and ")" in desc:
                        paren_start = desc.rfind("(")
                        paren_end = desc.rfind(")")
                        base_desc = desc[:paren_start].rstrip()
                        paren_desc = desc[paren_start+1:paren_end]
                        desc_with_metrics = f"{base_desc}  ({paren_desc}, {metric_str})"
                    else:
                        desc_with_metrics = f"{desc}  ({metric_str})"
                    
                    lines.append(f"  {mark}  {desc_with_metrics}")
                else:
                    lines.append(f"  {mark}  {desc}")
            else:
                lines.append(f"  {mark}  {desc}")
        # Note about interpretation
        min_eff = float(self.raw_row.get("_min_effect", 0.01))
        lines.append(
            f"  Note: flags are guidance, not strict pass/fail; using min_effect={min_eff:.4f}"
        )

        # Print the normalised table values (no nulls) for the canonical metrics.
        norm_map = {
            "OAR": ("oarR_max", "oarR_max"),
            "HAR": ("harRcond_ohist_max", "harRcond_hidden_max"),
            "PIF": ("pifOARcond_ohist_max", "pifRcond_hidden_max"),
            "AA": ("aaRcond_max", "aaRcond_max"),
            "DAI": ("daiOARcond_ohist_max", "daiRcond_hidden_max"),
        }
        metric_parts = []
        for short, (ff_col, rnn_col) in norm_map.items():
            col = rnn_col if is_rnn else ff_col
            v = self.metrics.get(col)
            if v is None:
                continue
            try:
                metric_parts.append(f"{short}^norm={float(v):.4f}")
            except Exception:
                metric_parts.append(f"{short}^norm={v}")

        if metric_parts:
            lines.append("  Table values: " + ", ".join(metric_parts))

        return "\n".join(lines)


# ----------------------------------------------------------------------
# One-call API for a single run
# ----------------------------------------------------------------------


def compute_diagnostics(
    data: UserData,
    *,
    history_k: int = 3,
    cmi_k: int = 25,
    null_reps: int = 5,
    metrics: Sequence[str] = ("oar", "har", "pif", "aa", "dai"),
    force_continuous_A: Optional[bool] = None,
    max_samples: Optional[int] = 8000,
    parallel_workers: int = 1,
    posterior_alpha: float = 0.5,
    min_effect: float = 0.01,
) -> DiagnosticResult:
    """Compute every diagnostic for a single run and apply Decision Rules.

    Parameters
    ----------
    data : UserData
        The trajectories from a single (env, alg, seed) run.
    history_k : int, default 3
        Observation/action history window length (paper default).
    cmi_k : int, default 25
        kNN neighbours for CMI estimators (paper default).
    null_reps : int, default 5
        Permutation-null replicates per metric (paper uses 5).
    metrics : iterable of str
        Subset of ``{"oar", "har", "pif", "aa", "dai"}`` to compute.
    force_continuous_A : bool or None
        If ``None`` (default), auto-detect from the action arrays. Set ``True``
        for continuous-action environments where actions look integer-valued
        but should be treated as continuous.
    max_samples : int or None, default 8000
        Joint subsample per agent (preserves cross-agent alignment). Reduces
        kNN cost; 8000-10000 is usually enough for ``cmi_k <= 25``.
        Set to ``None`` to disable.
    parallel_workers : int, default 1
        Workers for kNN tree queries (use 1 inside joblib parallel runs).
    posterior_alpha : float, default 0.5
        Laplace smoothing strength for discrete-action kNN posterior
        estimators. The default preserves the original estimator behaviour.
    """
    if force_continuous_A is None:
        force_continuous_A = any(
            _actions_are_continuous(a) for a in data.actions.values()
        )

    if "RNN" in data.alg_name.upper() and data.hidden_states is None:
        logger.warning(
            "alg_name=%r looks recurrent, but UserData.hidden_states is None. "
            "Observation-history will be used instead of Hidden-state. Save/pass RNN "
            "hidden states if you want the RNN diagnostics used in the paper.",
            data.alg_name,
        )

    metrics_set = set(metrics)
    invalid = metrics_set - {"oar", "har", "pif", "aa", "dai"}
    if invalid:
        raise ValueError(f"Unknown metrics: {invalid}")

    raw_row = process_run(
        data.to_run_dict(),
        metrics_to_run=metrics_set,
        history_k=history_k,
        cmi_k=cmi_k,
        bits=True,
        force_continuous_A=force_continuous_A,
        null_reps=null_reps,
        kd_workers=parallel_workers,
        max_samples=max_samples,
        posterior_alpha=posterior_alpha,
    )
    if raw_row is None:
        raise RuntimeError("process_run returned None — check logs for unpack errors.")

    # Record the min-effect threshold used for flagging so `describe()` can report it.
    raw_row["_min_effect"] = float(min_effect)
    raw_row["scenario_name"] = data.scenario_name

    is_rnn = "RNN" in data.alg_name.upper() and data.hidden_states is not None
    flags = _decision_flags_from_row(raw_row, is_rnn=is_rnn, min_effect=min_effect)

    return DiagnosticResult(metrics=raw_row, flags=flags, raw_row=raw_row)


def _decision_flags_from_row(
    row: Dict[str, object], *, is_rnn: bool, min_effect: float = 0.0
) -> Dict[str, bool]:
    """Apply Decision Rules 2-4 (and the HAR-uses-history rule) to one run.

    For each diagnostic, the row supplies both the (normalised) value and the
    permutation-null mean. The flag is True iff the value strictly exceeds
    the null AND the effect size (value - null) meets `min_effect`.
    Returns ``False`` when either is missing/NaN.
    """

    def _flag(value_col: str, null_col: str) -> bool:
        v = row.get(value_col)
        n = row.get(null_col)
        if v is None or n is None:
            return False
        if not (isinstance(v, (int, float)) and isinstance(n, (int, float))):
            return False
        if not (np.isfinite(v) and np.isfinite(n)):
            return False
        v_f = float(v)
        n_f = float(n)
        # Require strictly greater than null and a minimum absolute effect size.
        return (v_f > n_f) and ((v_f - n_f) >= float(min_effect))

    har_v = "harRcond_hidden_max" if is_rnn else "harRcond_ohist_max"
    har_n = har_v + "_null"
    pif_v = "pifRcond_hidden_max" if is_rnn else "pifOARcond_ohist_max"
    pif_n = pif_v + "_null"
    dai_v = "daiRcond_hidden_max" if is_rnn else "daiOARcond_ohist_max"
    dai_n = dai_v + "_null"

    return {
        "history_dependence":         _flag(har_v, har_n),
        "uses_hidden_teammate_info":  _flag(pif_v, pif_n),
        "synchronous_coordination":   _flag("aaRcond_max", "aaRcond_max_null"),
        "temporal_coordination":      _flag(dai_v, dai_n),
    }


# ----------------------------------------------------------------------
# Memory-Reactive performance gap (Diagnostic 1)
# ----------------------------------------------------------------------


def memory_reactive_gap(
    rnn_returns: Sequence[float],
    ff_returns: Sequence[float],
    alpha: float = 0.05,
) -> dict:
    """Diag 1: Δ_Mem = J(π_RNN) − J(π_FF), one-sided paired Wilcoxon test.

    Parameters
    ----------
    rnn_returns, ff_returns : array-like of float
        Mean evaluation return per matched seed (same env, same algorithm
        family, same seed for RNN vs FF). Lengths must match.
    alpha : float, default 0.05
        Significance threshold.

    Returns
    -------
    dict with keys
        ``delta_mean`` — mean of paired differences ``J_RNN − J_FF``
        ``p_value``    — one-sided Wilcoxon signed-rank p-value (H1: Δ > 0)
        ``benefits_from_memory`` — bool, ``p < alpha and delta_mean > 0``
        ``n_pairs``    — number of matched seed pairs

    Notes
    -----
    Implements Decision Rule 1 (Sec. 6.1) — combine with the
    ``history_dependence`` flag from :func:`compute_diagnostics` (HAR > null
    on the RNN policy) for the full "Do agents benefit from memory?" answer.
    """
    rnn = np.asarray(rnn_returns, dtype=float)
    ff = np.asarray(ff_returns, dtype=float)
    if rnn.shape != ff.shape or rnn.ndim != 1:
        raise ValueError(
            f"rnn_returns and ff_returns must be matching 1D arrays; "
            f"got {rnn.shape} vs {ff.shape}."
        )
    if rnn.size < 2:
        raise ValueError("Need at least 2 paired seeds for the Wilcoxon test.")

    from scipy.stats import wilcoxon

    diffs = rnn - ff
    delta_mean = float(np.mean(diffs))
    try:
        stat = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
        p_value = float(stat.pvalue)
    except ValueError:
        # All differences zero — no evidence of an effect
        p_value = 1.0

    return {
        "delta_mean": delta_mean,
        "p_value": p_value,
        "benefits_from_memory": (p_value < alpha) and (delta_mean > 0),
        "n_pairs": int(rnn.size),
    }


# ----------------------------------------------------------------------
# Multi-run aggregation -> Paper Table 1
# ----------------------------------------------------------------------


@dataclass
class ScenarioRun:
    """One per-seed result, ready for aggregation into Paper Table 1.

    Attach all the seeds for one ``(env, scenario, alg)`` group to satisfy
    each Decision Rule per-scenario, then aggregate over scenarios per env.

    ``scenario_name`` distinguishes scenarios *within* an env. If left as
    ``None``, the scenario defaults to ``env_name`` (each env = one scenario).
    """

    env_name: str
    alg_name: str
    seed: int
    flags: Dict[str, bool]
    scenario_name: Optional[str] = None


def _is_rnn_alg(alg_name: str) -> bool:
    return "RNN" in alg_name.upper()


def build_paper_table(
    results: Sequence[Union[ScenarioRun, DiagnosticResult]],
    *,
    memory_gap_flags: Optional[Dict[str, bool]] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Build the paper Table 1: share of scenarios satisfying each Decision Rule.

    Per Sec. 6.1 the aggregation is two-stage:

    1. Within a run, take the *max* across agents — done already by
       ``process_run`` (the ``*_max`` columns).
    2. Per scenario, the rule is satisfied iff *any* ``(alg, seed)`` run in
       that scenario exhibits the property (conservative per-scenario
       verdict).
    3. The table reports the share of scenarios in each ``env_name`` where
       the rule holds.

    A "scenario" is identified by ``(env_name, scenario_name)``. When
    ``scenario_name`` is ``None`` (the default), each unique ``env_name`` is
    treated as a single scenario — so to reproduce Table 1's "3/3 in MPE",
    set ``env_name="MPE"`` and pass each map's id via ``scenario_name``.

    Parameters
    ----------
    results : sequence of ScenarioRun or DiagnosticResult
        Per-seed flags for every (env, scenario, alg, seed) run.
    memory_gap_flags : dict[str, bool], optional
        Map of scenario or env identifier -> bool from
        :func:`memory_reactive_gap`. Lookup is tried by scenario id first
        (``f"{env_name}/{scenario_name}"`` and ``scenario_name``), then by
        ``env_name``. If supplied, the "Do agents benefit from memory?" row
        uses ``(history_dependence ∧ benefits_from_memory)``; otherwise it
        falls back to ``history_dependence`` alone (HAR > null on RNN).
    output_path : str, optional
        If given, write a LaTeX table to ``<output_path>_paper_table.tex``.

    Returns
    -------
    pandas.DataFrame
        Rows: the four Decision Rules. Columns: ``env_name``s. Cells are the
        share of scenarios satisfying the rule, formatted as ``"100% (3/3)"``.
    """
    rows = []
    for r in results:
        if isinstance(r, DiagnosticResult):
            scenario_name = r.raw_row.get("scenario_name")
            rows.append(ScenarioRun(
                env_name=str(r.raw_row.get("env_name", "env")),
                alg_name=str(r.raw_row.get("alg", "alg")),
                seed=int(r.raw_row.get("seed", 0) or 0),
                flags=dict(r.flags),
                scenario_name=str(scenario_name) if scenario_name else None,
            ))
        elif isinstance(r, ScenarioRun):
            rows.append(r)
        else:
            raise TypeError(f"Expected ScenarioRun or DiagnosticResult, got {type(r)}")

    if not rows:
        raise ValueError("results is empty — cannot build a table.")

    df = pd.DataFrame([
        {"env": r.env_name,
         "scenario": r.scenario_name if r.scenario_name else r.env_name,
         "alg": r.alg_name, "seed": r.seed,
         "is_rnn": _is_rnn_alg(r.alg_name), **r.flags}
        for r in rows
    ])

    flag_cols = ["history_dependence", "uses_hidden_teammate_info",
                 "synchronous_coordination", "temporal_coordination"]
    for c in flag_cols:
        if c not in df.columns:
            df[c] = False

    # Per-scenario verdict: ANY across (alg, seed) within the scenario.
    scenario_df = df.groupby(["env", "scenario"], as_index=False)[flag_cols].any()

    # Decision Rule 1 — "Do agents benefit from memory?" needs the RNN-vs-FF
    # gap AND HAR > null on the RNN policy. Try scenario-level keys first
    # (``"{env}/{scenario}"`` then plain ``scenario``), then fall back to
    # ``env_name`` for backward compatibility.
    if memory_gap_flags is not None:
        def _lookup_gap(env: str, scenario: str) -> bool:
            for key in (f"{env}/{scenario}", scenario, env):
                if key in memory_gap_flags:
                    return bool(memory_gap_flags[key])
            return False

        scenario_df["history_dependence"] = scenario_df.apply(
            lambda row: bool(row["history_dependence"])
                        and _lookup_gap(row["env"], row["scenario"]),
            axis=1,
        )

    rule_labels = {
        "history_dependence":         "Do agents benefit from memory?",
        "uses_hidden_teammate_info":  "Do agents use hidden teammate information?",
        "synchronous_coordination":   "Does synchronous coordination emerge?",
        "temporal_coordination":      "Does temporal coordination emerge?",
    }

    envs = sorted(scenario_df["env"].unique())
    table = {}
    for env in envs:
        sub = scenario_df[scenario_df["env"] == env]
        n_total = len(sub)
        env_col = {}
        for col, label in rule_labels.items():
            n_pos = int(sub[col].sum())
            pct = (100 * n_pos / n_total) if n_total else 0.0
            env_col[label] = f"{pct:.0f}% ({n_pos}/{n_total})"
        table[env] = env_col

    out = pd.DataFrame(table).reindex(list(rule_labels.values()))

    if output_path is not None:
        _write_paper_table_latex(out, output_path)

    return out


def _latex_escape(s: str) -> str:
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _write_paper_table_latex(table: pd.DataFrame, output_path: str) -> None:
    stem = output_path
    for suffix in (".csv", ".tex"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    tex_path = stem + "_paper_table.tex"
    n_envs = len(table.columns)

    header_cells = [_latex_escape(c) for c in table.columns]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Diagnostics of learned behaviour across cooperative MARL benchmarks. "
        r"Share of scenarios (count/total) where trained policies satisfy each Decision Rule.}",
        r"\label{tab:paper_diagnostics}",
        r"\begin{tabular}{l" + "c" * n_envs + "}",
        r"\toprule",
        " & " + " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]
    for label, row in table.iterrows():
        cells = [_latex_escape(v) for v in row.values]
        lines.append(_latex_escape(label) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved paper-style Table 1 LaTeX to {tex_path}")
