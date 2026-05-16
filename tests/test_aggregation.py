"""Tests for the permutation null and the Table 1 aggregation."""

import numpy as np
import pytest

import dec_pomdp_diagnostics as dpd
from dec_pomdp_diagnostics.data import permute_actions
from dec_pomdp_diagnostics.metrics import compute_dai_oa_hist


# ----------------------------------------------------------------------
# Permutation null
# ----------------------------------------------------------------------


def test_permute_actions_preserves_per_agent_marginals():
    rng = np.random.default_rng(0)
    Ad = {
        "agent_0": rng.integers(0, 4, size=500),
        "agent_1": rng.integers(0, 3, size=500),
    }
    Ad_null = permute_actions(Ad, np.random.default_rng(7))

    for a in Ad:
        before = np.bincount(Ad[a], minlength=4)
        after = np.bincount(Ad_null[a], minlength=4)
        np.testing.assert_array_equal(before, after)


def test_permute_actions_independent_across_agents():
    rng = np.random.default_rng(0)
    n = 1000
    a0 = rng.integers(0, 2, size=n)
    Ad = {"agent_0": a0.copy(), "agent_1": a0.copy()}  # perfectly correlated
    Ad_null = permute_actions(Ad, np.random.default_rng(0))

    real_corr = np.mean(Ad["agent_0"] == Ad["agent_1"])
    null_corr = np.mean(Ad_null["agent_0"] == Ad_null["agent_1"])
    assert real_corr > 0.99
    # After independent permutation, agreement should drop to ~chance (0.5).
    assert abs(null_corr - 0.5) < 0.1


def test_permutation_null_destroys_temporal_dependence():
    """Real trajectories with strong i->j temporal influence should produce a
    DAI value well above the permutation-null mean."""
    rng = np.random.default_rng(0)
    n_eps, horizon = 160, 12
    n = n_eps * horizon
    ts = np.tile(np.arange(horizon, dtype=np.int64), n_eps)
    eps = np.repeat(np.arange(n_eps, dtype=np.int64), horizon)

    # agent_0 has a hidden bit per timestep; agent_1 copies it one step later.
    a0_hist = rng.integers(0, 2, size=(n_eps, horizon))
    a0 = a0_hist.reshape(-1)
    a1 = np.empty(n, dtype=np.int64)
    for ep in range(n_eps):
        for t in range(horizon):
            row = ep * horizon + t
            a1[row] = a0_hist[ep, t - 1] if t > 0 else rng.integers(0, 2)

    Sd = {
        "agent_0": a0_hist.reshape(-1, 1).astype(float),
        "agent_1": rng.normal(size=(n, 1)),
    }
    Ad_real = {"agent_0": a0, "agent_1": a1}
    Td = {"agent_0": ts, "agent_1": ts}
    Ed = {"agent_0": eps, "agent_1": eps}

    real, _, _ = compute_dai_oa_hist(
        Sd, Ad_real, Td, Ed, k_window=1, k=25, posterior_alpha=0.01,
    )

    null_means = []
    for rep in range(3):
        Ad_null = permute_actions(Ad_real, np.random.default_rng(rep))
        null, _, _ = compute_dai_oa_hist(
            Sd, Ad_null, Td, Ed, k_window=1, k=25, posterior_alpha=0.01,
        )
        null_means.append(null["agent_1"])

    # Genuine temporal coupling should be well above the permutation null mean.
    assert real["agent_1"] > float(np.mean(null_means)) + 0.05


# ----------------------------------------------------------------------
# build_paper_table
# ----------------------------------------------------------------------


def _make_scenario_runs(env, scenarios_with_flags, algs=("IPPO_FF", "IPPO_RNN"), seeds=(0, 1)):
    """Build ScenarioRun list. ``scenarios_with_flags`` is a list of
    ``(scenario_name, flag_pattern)`` where ``flag_pattern`` is a dict from
    (alg, seed) -> dict of flags. Missing entries default to all-False."""
    runs = []
    for scen, pattern in scenarios_with_flags:
        for alg in algs:
            for seed in seeds:
                flags = pattern.get((alg, seed), {})
                full = {
                    "history_dependence": flags.get("history_dependence", False),
                    "uses_hidden_teammate_info": flags.get("uses_hidden_teammate_info", False),
                    "synchronous_coordination": flags.get("synchronous_coordination", False),
                    "temporal_coordination": flags.get("temporal_coordination", False),
                }
                runs.append(dpd.ScenarioRun(
                    env_name=env, scenario_name=scen,
                    alg_name=alg, seed=seed, flags=full,
                ))
    return runs


def test_build_paper_table_per_scenario_any_aggregation():
    """A flag set on ANY (alg, seed) within a scenario should flip the
    scenario verdict to True."""
    runs = _make_scenario_runs(
        env="MPE",
        scenarios_with_flags=[
            ("simple_reference", {("IPPO_RNN", 0): {"history_dependence": True}}),
            ("simple_spread", {("IPPO_FF", 1): {"history_dependence": True}}),
            ("simple_speaker_listener", {}),  # no flag in any run
        ],
    )
    table = dpd.build_paper_table(runs)
    assert "MPE" in table.columns
    assert table.loc["Do agents benefit from memory?", "MPE"] == "67% (2/3)"


def test_build_paper_table_all_scenarios_flagged_gives_full_share():
    runs = _make_scenario_runs(
        env="MPE",
        scenarios_with_flags=[
            (f"scen_{i}", {("IPPO_RNN", 0): {"synchronous_coordination": True}})
            for i in range(3)
        ],
    )
    table = dpd.build_paper_table(runs)
    assert table.loc["Does synchronous coordination emerge?", "MPE"] == "100% (3/3)"


def test_build_paper_table_falls_back_to_env_name_when_scenario_absent():
    """Backward-compat: a single env_name with no scenario_name = one scenario."""
    runs = [
        dpd.ScenarioRun(env_name="env", alg_name="IPPO_FF", seed=s,
                        flags={"history_dependence": s == 0,
                               "uses_hidden_teammate_info": False,
                               "synchronous_coordination": False,
                               "temporal_coordination": False})
        for s in range(3)
    ]
    table = dpd.build_paper_table(runs)
    # All 3 seeds collapse into the single scenario "env"; ANY → True.
    assert table.loc["Do agents benefit from memory?", "env"] == "100% (1/1)"


def test_build_paper_table_combines_memory_gap_flag_per_scenario():
    """Decision Rule 1 requires HAR > null AND the Wilcoxon gap to be
    positive. The gap flag can be keyed by scenario_name OR env_name."""
    runs = _make_scenario_runs(
        env="MPE",
        scenarios_with_flags=[
            ("scen_A", {("IPPO_RNN", 0): {"history_dependence": True}}),
            ("scen_B", {("IPPO_RNN", 0): {"history_dependence": True}}),
        ],
    )
    # scen_A has the memory advantage; scen_B does not → only 1/2 satisfies Rule 1.
    table = dpd.build_paper_table(runs, memory_gap_flags={"scen_A": True, "scen_B": False})
    assert table.loc["Do agents benefit from memory?", "MPE"] == "50% (1/2)"


def test_build_paper_table_memory_gap_flag_keyed_by_env_is_backward_compatible():
    runs = _make_scenario_runs(
        env="MPE",
        scenarios_with_flags=[
            ("scen_A", {("IPPO_RNN", 0): {"history_dependence": True}}),
            ("scen_B", {("IPPO_RNN", 0): {"history_dependence": True}}),
        ],
    )
    # Env-level key applies to every scenario in MPE.
    table = dpd.build_paper_table(runs, memory_gap_flags={"MPE": True})
    assert table.loc["Do agents benefit from memory?", "MPE"] == "100% (2/2)"

    table_off = dpd.build_paper_table(runs, memory_gap_flags={"MPE": False})
    assert table_off.loc["Do agents benefit from memory?", "MPE"] == "0% (0/2)"


def test_build_paper_table_rejects_empty_results():
    with pytest.raises(ValueError, match="empty"):
        dpd.build_paper_table([])


def test_build_paper_table_rejects_wrong_type():
    with pytest.raises(TypeError, match="ScenarioRun"):
        dpd.build_paper_table([{"env_name": "x"}])


def test_build_paper_table_separates_envs():
    """Two envs with different scenario counts should appear as separate columns."""
    mpe = _make_scenario_runs(
        env="MPE",
        scenarios_with_flags=[
            ("a", {("IPPO_RNN", 0): {"temporal_coordination": True}}),
            ("b", {}),
        ],
    )
    smax = _make_scenario_runs(
        env="SMAX V1",
        scenarios_with_flags=[
            ("3m", {("IPPO_RNN", 0): {"temporal_coordination": True}}),
            ("8m", {("IPPO_RNN", 0): {"temporal_coordination": True}}),
            ("2s3z", {("IPPO_RNN", 0): {"temporal_coordination": True}}),
        ],
    )
    table = dpd.build_paper_table(mpe + smax)
    assert set(table.columns) == {"MPE", "SMAX V1"}
    assert table.loc["Does temporal coordination emerge?", "MPE"] == "50% (1/2)"
    assert table.loc["Does temporal coordination emerge?", "SMAX V1"] == "100% (3/3)"


def test_build_paper_table_accepts_diagnostic_results_and_extracts_scenario_name():
    """End-to-end: compute_diagnostics → DiagnosticResult carrying
    scenario_name flows correctly into build_paper_table."""
    rng = np.random.default_rng(0)
    N, T, obs_dim = 240, 12, 4
    n_eps = N // T
    ts = np.tile(np.arange(T, dtype=np.int64), n_eps)
    eps = np.repeat(np.arange(n_eps, dtype=np.int64), T)
    obs = {a: rng.normal(size=(N, obs_dim)).astype(np.float32) for a in ("agent_0", "agent_1")}
    act = {a: rng.integers(0, 2, size=N, dtype=np.int64) for a in obs}

    results = []
    for scen in ("scen_x", "scen_y"):
        for seed in (0, 1):
            data = dpd.UserData(
                observations=obs, actions=act,
                timesteps={a: ts for a in obs},
                episode_ids={a: eps for a in obs},
                env_name="MPE", alg_name="IPPO_FF", seed=seed,
                scenario_name=scen,
            )
            results.append(dpd.compute_diagnostics(data, history_k=2, null_reps=1, cmi_k=10))

    table = dpd.build_paper_table(results)
    # Random data: no scenario should trigger any flag.
    assert table.loc["Does synchronous coordination emerge?", "MPE"] == "0% (0/2)"
