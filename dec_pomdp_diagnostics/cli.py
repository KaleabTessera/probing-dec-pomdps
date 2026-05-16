"""Command-line entry points: ``dec-pomdp-collect`` and ``dec-pomdp-metrics``.

Both commands are also exposed as ``python -m dec_pomdp_diagnostics.collect``
and ``python -m dec_pomdp_diagnostics.compute`` for users who prefer that.
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd
from joblib import Parallel, delayed

from .data import load_dataset
from .pipeline import AVAILABLE_METRICS, process_run
from .summary import generate_latex_table, print_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _cap_threads(n: int = 1) -> None:
    """Cap thread counts in BLAS/OpenMP libraries — required before joblib forks."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)


# Re-apply if a worker process inherits the flag from the parent
if os.environ.get("_METRICS_PARALLEL_WORKER") == "1":
    _cap_threads(1)


# ----------------------------------------------------------------------
# compute_metrics CLI
# ----------------------------------------------------------------------


def _compute_metrics_args():
    p = argparse.ArgumentParser(
        prog="dec-pomdp-metrics",
        description="Compute Dec-POMDP behavioural diagnostics from a collected dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=str, required=True, help="Path to dataset .pkl from dec-pomdp-collect")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path (default: <input_stem>_metrics_<set>.csv)")
    p.add_argument("--history-k", type=int, default=3, help="Observation/action history window length")
    p.add_argument("--cmi-k", type=int, default=25, help="kNN neighbours for CMI/MI estimators")
    p.add_argument("--posterior-alpha", type=float, default=0.5,
                   help="Laplace smoothing strength for discrete-action kNN posterior estimators")
    p.add_argument("--cmi-bits", action="store_true", default=True, help="Report MI in bits (default)")
    p.add_argument("--force-continuous-A", action="store_true", default=False,
                   help="Treat actions as continuous (use for continuous-action environments like MaBrax)")
    p.add_argument("--null-reps", type=int, default=0,
                   help="Permutation-null replicates per run (0 = skip; paper uses 5)")
    p.add_argument("--parallel", action="store_true", default=False, help="Use joblib parallel processing")
    p.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs (-1 = all cores)")
    p.add_argument("--metrics", nargs="+", default=["all"],
                   choices=list(AVAILABLE_METRICS) + ["all"],
                   help="Subset of diagnostics to compute")
    p.add_argument("--env", nargs="+", default=None, help="Filter to specific environment(s)")
    p.add_argument("--alg", nargs="+", default=None, help="Filter to specific algorithm(s)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Subsample to ≤ N samples per agent (stratified by episode); 8000-10000 fits k≤25 well")
    return p.parse_args()


def compute_metrics_main():
    args = _compute_metrics_args()

    metrics_to_run = set(args.metrics)
    if "all" in metrics_to_run:
        metrics_to_run = AVAILABLE_METRICS
    logger.info(f"Selected metrics: {sorted(metrics_to_run)}")
    if args.max_samples:
        logger.info(f"Subsampling to max {args.max_samples} samples per agent")

    logger.info(f"Loading dataset from {args.input}")
    dataset = load_dataset(args.input)
    runs = dataset["runs"]
    logger.info(f"Loaded {len(runs)} runs")

    if args.env or args.alg:
        env_set = set(args.env) if args.env else None
        alg_set = set(args.alg) if args.alg else None
        before = len(runs)
        runs = [
            r for r in runs
            if (env_set is None or r.get("map_name") in env_set)
            and (alg_set is None or r.get("alg_name") in alg_set)
        ]
        logger.info(f"Filtered to {len(runs)}/{before} runs (env={args.env}, alg={args.alg})")
        if not runs:
            all_combos = sorted({(r.get("map_name"), r.get("alg_name")) for r in dataset["runs"]})
            logger.error("No runs match the filter. Available (env, alg) combos:")
            for e, a in all_combos:
                logger.error(f"  {e} / {a}")
            return

    if args.output is None:
        stem = os.path.splitext(args.input)[0]
        metrics_part = "-".join(sorted(metrics_to_run))
        filter_parts = []
        if args.env:
            filter_parts.append("env_" + "+".join(args.env))
        if args.alg:
            filter_parts.append("alg_" + "+".join(args.alg))
        filter_suffix = "_" + "_".join(filter_parts) if filter_parts else ""
        args.output = f"{stem}_metrics_{metrics_part}{filter_suffix}.csv"

    kd_workers = 1 if args.parallel else -1

    if args.parallel:
        _cap_threads(1)
        os.environ["_METRICS_PARALLEL_WORKER"] = "1"
        logger.info(f"Processing in parallel ({args.n_jobs} jobs, kd_workers={kd_workers})...")
        rows = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(process_run)(
                r, metrics_to_run, args.history_k, args.cmi_k, args.cmi_bits,
                args.force_continuous_A, args.null_reps, kd_workers=kd_workers,
                max_samples=args.max_samples, posterior_alpha=args.posterior_alpha,
            )
            for r in runs
        )
    else:
        rows = []
        for i, r in enumerate(runs):
            logger.info(f"[{i+1}/{len(runs)}]")
            rows.append(process_run(
                r, metrics_to_run, args.history_k, args.cmi_k, args.cmi_bits,
                args.force_continuous_A, args.null_reps, max_samples=args.max_samples,
                posterior_alpha=args.posterior_alpha,
            ))

    rows = [r for r in rows if r is not None]
    if not rows:
        logger.error("No metrics computed — check your data.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["env_name", "alg", "seed"], na_position="last").reset_index(drop=True)
    df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(df)} rows to {args.output}")

    summary_rows = print_summary(df)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_output = args.output.replace(".csv", "_summary.csv")
        summary_df.to_csv(summary_output, index=False)
        logger.info(f"Saved summary to {summary_output}")
        generate_latex_table(df, args.output)


# ----------------------------------------------------------------------
# collect_data CLI
# ----------------------------------------------------------------------


def _collect_args():
    p = argparse.ArgumentParser(
        prog="dec-pomdp-collect",
        description="Download eval rollouts from W&B into a single pickle dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--entity", type=str, required=True, help="W&B entity")
    p.add_argument("--project", type=str, required=True, help="W&B project")
    p.add_argument("--tag", type=str, required=True, help="Tag to filter runs by")
    p.add_argument("--artifact-type", type=str, default="dataset", help="Artifact type to look for")
    p.add_argument("--artifact-name", type=str, default="final", help="Substring filter for artifact name")
    p.add_argument("--map-key", type=str, default="MAP_NAME",
                   help="Config key for map/env name (use ENV_KWARGS.<key> to nest)")
    p.add_argument("--alg-key", type=str, default="ALG", help="Config key for algorithm name")
    p.add_argument("--max-runs", type=int, default=None, help="Max runs to download (None = all)")
    p.add_argument("--output", type=str, required=True, help="Output pickle path")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory (default: cwd)")
    p.add_argument("--dry-run", action="store_true", help="List matching runs without downloading")
    return p.parse_args()


def collect_main():
    args = _collect_args()
    from .wandb_collect import collect_all, save_dataset

    output_path = args.output
    if args.output_dir:
        output_path = os.path.join(args.output_dir, output_path)

    logger.info(f"Collecting: entity={args.entity}, project={args.project}, tag={args.tag}")
    dataset = collect_all(
        entity=args.entity, project=args.project, tag=args.tag,
        artifact_type=args.artifact_type, artifact_name=args.artifact_name,
        map_key=args.map_key, alg_key=args.alg_key,
        max_runs=args.max_runs, dry_run=args.dry_run,
    )

    if not args.dry_run and dataset.get("runs"):
        save_dataset(dataset, output_path)
        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY")
        print("=" * 60)
        print(f"  Runs collected: {len(dataset['runs'])}")
        print(f"  Output file:    {output_path}")
        print("\n  Environments & Algorithms:")
        for (m, a), idxs in sorted(dataset["index"].items()):
            print(f"    {m} / {a}: {len(idxs)} run(s)")
        print("=" * 60)
    elif not args.dry_run:
        logger.warning("No data collected. Check your filters.")


if __name__ == "__main__":
    compute_metrics_main()
