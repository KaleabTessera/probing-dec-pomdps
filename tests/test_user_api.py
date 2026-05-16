"""Tests for the user-facing API: UserData validation, compute_diagnostics,
memory_reactive_gap."""

import numpy as np
import pytest

import dec_pomdp_diagnostics as dpd


def _valid_userdata_args(N=240, T=12, obs_dim=4, n_agents=2, hidden=False, alg="IPPO_FF"):
    """Build a minimal valid kwargs dict for UserData."""
    rng = np.random.default_rng(0)
    n_eps = N // T
    ts = np.tile(np.arange(T, dtype=np.int64), n_eps)
    eps = np.repeat(np.arange(n_eps, dtype=np.int64), T)
    agents = [f"agent_{i}" for i in range(n_agents)]
    obs = {a: rng.normal(size=(N, obs_dim)).astype(np.float32) for a in agents}
    act = {a: rng.integers(0, 2, size=N, dtype=np.int64) for a in agents}
    kwargs = dict(
        observations=obs,
        actions=act,
        timesteps={a: ts for a in agents},
        episode_ids={a: eps for a in agents},
        env_name="env",
        alg_name=alg,
        seed=0,
    )
    if hidden:
        kwargs["hidden_states"] = {a: rng.normal(size=(N, 8)).astype(np.float32) for a in agents}
    return kwargs


# ----------------------------------------------------------------------
# UserData validation
# ----------------------------------------------------------------------


def test_userdata_accepts_minimal_valid_input():
    data = dpd.UserData(**_valid_userdata_args())
    assert data.env_name == "env"
    assert data.scenario_name is None


def test_userdata_accepts_scenario_name():
    kw = _valid_userdata_args()
    kw["scenario_name"] = "simple_reference_v3"
    data = dpd.UserData(**kw)
    assert data.scenario_name == "simple_reference_v3"


def test_userdata_rejects_empty_observations():
    with pytest.raises(ValueError, match="at least one agent"):
        dpd.UserData(
            observations={}, actions={}, timesteps={}, episode_ids={},
        )


def test_userdata_rejects_mismatched_agent_keys():
    kw = _valid_userdata_args()
    kw["actions"] = {"agent_0": kw["actions"]["agent_0"]}  # drop agent_1
    with pytest.raises(ValueError, match="actions keys"):
        dpd.UserData(**kw)


def test_userdata_rejects_1d_observations():
    kw = _valid_userdata_args()
    kw["observations"]["agent_0"] = kw["observations"]["agent_0"][:, 0]
    with pytest.raises(ValueError, match="must be 2D"):
        dpd.UserData(**kw)


def test_userdata_rejects_single_row():
    kw = _valid_userdata_args(N=2, T=2)
    # Drop down to 1 row to trigger n < 2.
    for k in ("observations", "actions", "timesteps", "episode_ids"):
        kw[k] = {a: v[:1] for a, v in kw[k].items()}
    with pytest.raises(ValueError, match="need ≥ 2"):
        dpd.UserData(**kw)


def test_userdata_rejects_mismatched_action_length():
    kw = _valid_userdata_args()
    kw["actions"]["agent_0"] = kw["actions"]["agent_0"][:-1]
    with pytest.raises(ValueError, match="actions.*length"):
        dpd.UserData(**kw)


def test_userdata_rejects_float_timesteps():
    kw = _valid_userdata_args()
    kw["timesteps"]["agent_0"] = kw["timesteps"]["agent_0"].astype(float)
    with pytest.raises(ValueError, match="timesteps.*integer"):
        dpd.UserData(**kw)


def test_userdata_rejects_float_episode_ids():
    kw = _valid_userdata_args()
    kw["episode_ids"]["agent_0"] = kw["episode_ids"]["agent_0"].astype(float)
    with pytest.raises(ValueError, match="episode_ids.*integer"):
        dpd.UserData(**kw)


def test_userdata_rejects_hidden_state_shape_mismatch():
    kw = _valid_userdata_args(hidden=True)
    kw["hidden_states"]["agent_0"] = kw["hidden_states"]["agent_0"][:-1]
    with pytest.raises(ValueError, match="hidden_states"):
        dpd.UserData(**kw)


def test_userdata_rejects_hidden_state_extra_agent_key():
    kw = _valid_userdata_args(hidden=True)
    kw["hidden_states"]["agent_extra"] = kw["hidden_states"]["agent_0"]
    with pytest.raises(ValueError, match="hidden_states keys"):
        dpd.UserData(**kw)


def test_userdata_continuous_actions_accepted():
    kw = _valid_userdata_args()
    N = kw["observations"]["agent_0"].shape[0]
    kw["actions"] = {a: np.random.default_rng(0).normal(size=(N, 3)).astype(np.float32)
                     for a in kw["observations"]}
    data = dpd.UserData(**kw)
    assert data.actions["agent_0"].ndim == 2


# ----------------------------------------------------------------------
# compute_diagnostics
# ----------------------------------------------------------------------


def test_compute_diagnostics_returns_all_flags_false_on_random_data():
    data = dpd.UserData(**_valid_userdata_args(N=480, T=12))
    result = dpd.compute_diagnostics(data, history_k=2, null_reps=2, cmi_k=10)
    assert set(result.flags.keys()) == {
        "history_dependence",
        "uses_hidden_teammate_info",
        "synchronous_coordination",
        "temporal_coordination",
    }
    # Random data has no genuine dependence; all flags should be False under
    # the permutation-null comparison.
    for name, val in result.flags.items():
        assert val is False, f"Random data triggered flag {name!r}"


def test_compute_diagnostics_describe_contains_run_id_and_flags():
    data = dpd.UserData(**_valid_userdata_args(N=240, T=12))
    result = dpd.compute_diagnostics(data, history_k=2, null_reps=1, cmi_k=10)
    text = result.describe()
    assert "Run" in text
    # Every Decision-Rule flag should be mentioned in the describe output.
    for line in ("history", "teammate", "synchronous", "temporal"):
        assert line in text.lower()


def test_compute_diagnostics_rejects_unknown_metric():
    data = dpd.UserData(**_valid_userdata_args(N=120, T=12))
    with pytest.raises(ValueError, match="Unknown metrics"):
        dpd.compute_diagnostics(data, metrics=("oar", "bogus"))


def test_compute_diagnostics_propagates_scenario_name_into_raw_row():
    kw = _valid_userdata_args(N=240, T=12)
    kw["scenario_name"] = "my_scenario"
    data = dpd.UserData(**kw)
    result = dpd.compute_diagnostics(data, history_k=2, null_reps=1, cmi_k=10)
    assert result.raw_row.get("scenario_name") == "my_scenario"


def test_compute_diagnostics_uses_hidden_when_rnn_alg_and_hidden_supplied():
    kw = _valid_userdata_args(N=240, T=12, hidden=True, alg="IPPO_RNN")
    data = dpd.UserData(**kw)
    result = dpd.compute_diagnostics(data, history_k=2, null_reps=1, cmi_k=10)
    # When RNN + hidden states, the *_hidden columns should be populated.
    assert np.isfinite(result.raw_row.get("har_hidden_max", np.nan))


def test_compute_diagnostics_warns_when_rnn_without_hidden_states(caplog):
    kw = _valid_userdata_args(N=240, T=12, hidden=False, alg="IPPO_RNN")
    data = dpd.UserData(**kw)
    with caplog.at_level("WARNING"):
        dpd.compute_diagnostics(data, history_k=2, null_reps=1, cmi_k=10)
    assert any("recurrent" in rec.message.lower() for rec in caplog.records)


# ----------------------------------------------------------------------
# memory_reactive_gap (Diag 1)
# ----------------------------------------------------------------------


def test_memory_reactive_gap_detects_clear_advantage():
    rng = np.random.default_rng(0)
    n = 10
    ff = rng.normal(loc=0.0, scale=1.0, size=n)
    rnn = ff + 5.0  # large, consistent advantage
    out = dpd.memory_reactive_gap(rnn_returns=rnn, ff_returns=ff)
    assert out["benefits_from_memory"] is True
    assert out["p_value"] < 0.05
    assert out["delta_mean"] == pytest.approx(5.0, abs=1e-6)
    assert out["n_pairs"] == n


def test_memory_reactive_gap_no_advantage_returns_false():
    rng = np.random.default_rng(1)
    n = 20
    ff = rng.normal(size=n)
    rnn = rng.normal(size=n)  # independent — no expected advantage
    out = dpd.memory_reactive_gap(rnn_returns=rnn, ff_returns=ff)
    assert out["benefits_from_memory"] is False
    assert out["n_pairs"] == n


def test_memory_reactive_gap_all_zero_diffs_returns_p_one():
    out = dpd.memory_reactive_gap(
        rnn_returns=[1.0, 2.0, 3.0],
        ff_returns=[1.0, 2.0, 3.0],
    )
    assert out["p_value"] == 1.0
    assert out["benefits_from_memory"] is False
    assert out["delta_mean"] == 0.0


def test_memory_reactive_gap_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="matching 1D"):
        dpd.memory_reactive_gap(rnn_returns=[1, 2, 3], ff_returns=[1, 2])


def test_memory_reactive_gap_rejects_too_few_pairs():
    with pytest.raises(ValueError, match="at least 2"):
        dpd.memory_reactive_gap(rnn_returns=[1.0], ff_returns=[0.0])
