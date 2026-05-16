"""Dec-POMDP behavioural diagnostics for cooperative MARL.

Reference implementation of the diagnostic framework from
*Probing Dec-POMDP Reasoning in Cooperative MARL* (Tessera et al., AAMAS 2026).

The five information-theoretic diagnostics:

* :func:`compute_oar` — Observation-Action Relevance, ``I(O^i_t ; A^i_t)``
* :func:`compute_har_ohist` / :func:`compute_har_hidden` — History-Action
  Relevance, ``I(H^i_t ; A^i_t | O^i_t)``
* :func:`compute_pif_oa_hist` / :func:`compute_pif_hidden` — Private
  Information Flow, ``I(τ^i_{t-1}, O^i_t ; A^j_t | τ^j_{t-1}, O^j_t)``
* :func:`compute_aa` — Action-Action Coupling, ``I(A^i_t ; A^j_t | O^i_t, O^j_t)``
* :func:`compute_dai_oa_hist` / :func:`compute_dai_hidden` — Directed Action
  Information, ``(1/T) Σ_t I(τ^i_{t-1} ; A^j_t | τ^j_{t-1})``

End-to-end pipeline (per-run computation, permutation null, summary):
:func:`process_run`, :func:`print_summary`, :func:`generate_latex_table`.
"""

from .api import (
    DiagnosticResult,
    ScenarioRun,
    UserData,
    build_paper_table,
    compute_diagnostics,
    memory_reactive_gap,
)
from .data import load_dataset, permute_actions, subsample_agents, unpack_run
from .estimators import (
    cmi_knn,
    cmi_mixed,
    mi_cc,
    mi_cd,
)
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
from .pipeline import AVAILABLE_METRICS, METRIC_COLUMNS, process_run
from .summary import (
    NORM_COLS,
    SUMMARY_COLS,
    generate_latex_table,
    print_summary,
    rliable_mean_ci,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # High-level user API (start here)
    "UserData", "DiagnosticResult", "ScenarioRun",
    "compute_diagnostics", "memory_reactive_gap", "build_paper_table",
    # Estimators
    "cmi_knn", "cmi_mixed", "mi_cd", "mi_cc",
    # Metrics
    "compute_oar",
    "compute_har_hidden", "compute_har_ohist",
    "compute_pif_hidden", "compute_pif_ohist", "compute_pif_oa_hist",
    "compute_aa",
    "compute_dai_hidden", "compute_dai_oa_hist",
    # Data
    "load_dataset", "unpack_run", "subsample_agents", "permute_actions",
    # Pipeline / summary
    "process_run", "print_summary", "generate_latex_table", "rliable_mean_ci",
    "AVAILABLE_METRICS", "METRIC_COLUMNS", "NORM_COLS", "SUMMARY_COLS",
]
