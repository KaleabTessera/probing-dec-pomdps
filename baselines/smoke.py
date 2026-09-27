"""Exercise trainers, checkpoint reloads, rollouts and the metrics CLI."""

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.run import ALGORITHMS


def check_rollouts(run, expected_samples):
    """Reject missing, misaligned or non-finite data before computing metrics."""
    reference = next(iter(run["per_agent"].values()))
    recurrent = "_RNN_" in run["alg_name"]
    for fields in run["per_agent"].values():
        for key, value in fields.items():
            if key == "has_hidden":
                if value != recurrent:
                    raise AssertionError("Wrong hidden-state flag")
                continue
            if len(value) != expected_samples or not np.isfinite(value).all():
                raise AssertionError(f"Invalid {key} data")
        for key in ("timesteps", "episode_ids"):
            np.testing.assert_array_equal(fields[key], reference[key])
        keys = np.stack((fields["episode_ids"], fields["timesteps"]), axis=1)
        if len(np.unique(keys, axis=0)) != expected_samples:
            raise AssertionError("Repeated (episode, timestep) keys")
        for episode in np.unique(fields["episode_ids"]):
            t = fields["timesteps"][fields["episode_ids"] == episode]
            np.testing.assert_array_equal(t, np.arange(len(t)))
        if run["config"].get("CONTINUOUS_ACTIONS", False):
            np.testing.assert_allclose(fields["actions"], fields["action_mean"])
        else:
            logits = fields["pre_softmax_logits"]
            mask = fields.get("available_actions", np.ones_like(logits))
            np.testing.assert_array_equal(
                fields["actions"], (logits - (1 - mask) * 1e10).argmax(axis=-1)
            )
        if recurrent and not np.any(fields["hidden"] != 0):
            raise AssertionError("Missing recurrent memory")


def run_smoke(output, steps=1024, suite="mpe"):
    """Run short CPU jobs to check training, collection, and metric computation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    runs, reports = [], []
    algorithms = (
        ALGORITHMS
        if suite == "mpe"
        else tuple(a for a in ALGORITHMS if not a.endswith("_het"))
    )
    for algorithm in algorithms:
        config = "_".join(algorithm.split("_")[:2]) + "_" + suite
        run_dir = output / algorithm
        environment = (
            "MPE_simple_speaker_listener_v4"
            if algorithm.endswith("_het")
            else "MPE_simple_spread_v3"
        )
        if suite == "hanabi":
            environment = "hanabi"
        elif suite == "mabrax":
            environment = "hopper_3x1"
        command = [
            sys.executable,
            "-m",
            "baselines.run",
            "--algorithm",
            algorithm,
            "--config",
            config,
            "--output",
            str(run_dir),
            f"ENV_NAME={environment}",
            "NUM_ENVS=4",
            "NUM_STEPS=32",
            f"TOTAL_TIMESTEPS={steps}",
            "NUM_MINIBATCHES=2",
            "UPDATE_EPOCHS=2",
            "FC_DIM_SIZE=16",
            "TEST_NUM_ENVS=4",
            "TEST_NUM_STEPS=100",
            "SEED=0",
            "TEST_DURING_TRAINING=True",
        ]
        if "_rnn_" in algorithm:
            command.append("GRU_HIDDEN_DIM=16")
        if suite == "mabrax":
            command.extend(
                ["ENV_KWARGS.episode_length=25", "ENV_KWARGS.homogenisation_method=pad"]
            )
        print(f"Checking {algorithm}", flush=True)
        with (output / f"{algorithm}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            reloaded = run_dir / "reloaded.pkl"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "baselines.collect",
                    "--run",
                    str(run_dir),
                    "--output",
                    str(reloaded),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        with (run_dir / "rollouts.pkl").open("rb") as f:
            run = pickle.load(f)["runs"][0]
        with reloaded.open("rb") as f:
            restored = pickle.load(f)["runs"][0]
        check_rollouts(run, 400)
        for agent, fields in run["per_agent"].items():
            for key, value in fields.items():
                np.testing.assert_allclose(
                    value, restored["per_agent"][agent][key], atol=1e-6, rtol=1e-5
                )
        runs.append(run)
        reports.append(json.loads((run_dir / "report.json").read_text()))
    dataset = output / "rollouts.pkl"
    with dataset.open("wb") as f:
        pickle.dump({"runs": runs}, f, protocol=pickle.HIGHEST_PROTOCOL)
    csv = output / "metrics.csv"
    # Exercise the public command entry point, including bootstrap summaries and LaTeX.
    with (output / "metrics.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "dec_pomdp_diagnostics",
                "--input",
                str(dataset),
                "--output",
                str(csv),
                "--null-reps",
                "2",
                "--cmi-k",
                "5",
                "--history-k",
                "3",
            ]
            + (["--force-continuous-A"] if suite == "mabrax" else []),
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    frame = pd.read_csv(csv)
    if len(frame) != len(algorithms):
        raise AssertionError("Metrics CLI omitted runs")
    for _, row in frame.iterrows():
        recurrent = "_RNN_" in row["alg"]
        columns = ["oarR_max", "aaRcond_max"]
        columns += (
            ["harRcond_hidden_max", "pifRcond_hidden_max", "daiRcond_hidden_max"]
            if recurrent
            else ["harRcond_ohist_max", "pifOARcond_ohist_max", "daiOARcond_ohist_max"]
        )
        for column in columns:
            for key in (column, column + "_null"):
                if not np.isfinite(row[key]):
                    raise AssertionError(f"Missing metric {row['alg']}/{key}")
    for name in ("metrics_summary.csv", "metrics_table.tex"):
        if not (output / name).is_file():
            raise AssertionError(f"Missing CLI output: {name}")
    (output / "smoke_report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(
        f"Passed: {len(runs)} training/checkpoint/collection runs and all five metrics with nulls. Results: {output}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    parser.add_argument("--suite", choices=("mpe", "hanabi", "mabrax"), default="mpe")
    parser.add_argument(
        "--steps",
        type=int,
        default=1024,
        help="Steps per model, a positive multiple of 128",
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 128:
        parser.error("--steps must be a positive multiple of 128")
    # Avoid multiplying BLAS threads inside each estimator process.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    run_smoke(args.output, args.steps, args.suite)


if __name__ == "__main__":
    main()
