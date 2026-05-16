import numpy as np

from dec_pomdp_diagnostics.metrics import (
    compute_aa,
    compute_dai_oa_hist,
    compute_har_ohist,
    compute_oar,
    compute_pif_oa_hist,
)


def _time_keys(n_episodes=160, horizon=12):
    timesteps = np.tile(np.arange(horizon), n_episodes).astype(np.int64)
    episode_ids = np.repeat(np.arange(n_episodes), horizon).astype(np.int64)
    return (
        {"agent_0": timesteps, "agent_1": timesteps},
        {"agent_0": episode_ids, "agent_1": episode_ids},
    )


def _obs(obs0, obs1):
    return {"agent_0": np.asarray(obs0), "agent_1": np.asarray(obs1)}


def _discrete_actions(a0, a1):
    return {
        "agent_0": np.asarray(a0, dtype=np.int64),
        "agent_1": np.asarray(a1, dtype=np.int64),
    }


def _continuous_actions(a0, a1):
    return {
        "agent_0": np.asarray(a0, dtype=float).reshape(-1, 1),
        "agent_1": np.asarray(a1, dtype=float).reshape(-1, 1),
    }


def _previous(values, rng):
    out = np.empty(values.size, dtype=values.dtype)
    n_episodes, horizon = values.shape
    for episode in range(n_episodes):
        for t in range(horizon):
            row = episode * horizon + t
            out[row] = values[episode, t - 1] if t > 0 else rng.normal()
    return out


def _previous_binary(values, rng):
    out = np.empty(values.size, dtype=np.int64)
    n_episodes, horizon = values.shape
    for episode in range(n_episodes):
        for t in range(horizon):
            row = episode * horizon + t
            out[row] = values[episode, t - 1] if t > 0 else rng.integers(0, 2)
    return out


def test_discrete_oar_high_when_observation_determines_action_low_when_independent():
    rng = np.random.default_rng(0)
    n = 1600
    signal = rng.integers(0, 2, size=n)
    observations = _obs(
        np.column_stack([signal, rng.normal(scale=0.05, size=n)]),
        rng.normal(size=(n, 2)),
    )

    high_actions = _discrete_actions(signal, rng.integers(0, 2, size=n))
    low_actions = _discrete_actions(
        rng.integers(0, 2, size=n),
        rng.integers(0, 2, size=n),
    )

    _, high = compute_oar(observations, high_actions, k=3)
    _, low = compute_oar(observations, low_actions, k=3)

    assert high["agent_0"] > 0.85
    assert low["agent_0"] < 0.20


def test_discrete_har_high_for_history_use_low_when_current_observation_explains_action():
    rng = np.random.default_rng(1)
    n_episodes, horizon = 160, 12
    n = n_episodes * horizon
    timesteps, episode_ids = _time_keys(n_episodes, horizon)

    latent = rng.integers(0, 2, size=(n_episodes, horizon))
    observations = _obs(latent.reshape(-1, 1).astype(float), rng.normal(size=(n, 1)))
    high_actions = _discrete_actions(
        _previous_binary(latent, rng),
        rng.integers(0, 2, size=n),
    )
    low_actions = _discrete_actions(
        latent.reshape(-1),
        rng.integers(0, 2, size=n),
    )

    _, high = compute_har_ohist(
        observations,
        high_actions,
        timesteps,
        episode_ids,
        k_window=1,
        k=25,
        posterior_alpha=0.01,
    )
    _, low = compute_har_ohist(
        observations,
        low_actions,
        timesteps,
        episode_ids,
        k_window=1,
        k=25,
        posterior_alpha=0.01,
    )

    assert high["agent_0"] > 0.85
    assert low["agent_0"] < 0.20


def test_discrete_pif_high_for_private_teammate_information_low_when_own_obs_explains_action():
    rng = np.random.default_rng(2)
    n_episodes, horizon = 160, 12
    n = n_episodes * horizon
    timesteps, episode_ids = _time_keys(n_episodes, horizon)

    agent0_private = rng.integers(0, 2, size=n)
    agent1_private = rng.integers(0, 2, size=n)
    observations = _obs(
        agent0_private.reshape(-1, 1).astype(float),
        agent1_private.reshape(-1, 1).astype(float),
    )
    high_actions = _discrete_actions(
        rng.integers(0, 2, size=n),
        agent0_private,
    )
    low_actions = _discrete_actions(
        rng.integers(0, 2, size=n),
        agent1_private,
    )

    _, high, n_pairs = compute_pif_oa_hist(
        observations,
        high_actions,
        timesteps,
        episode_ids,
        k_window=1,
        k=25,
        posterior_alpha=0.01,
    )
    _, low, _ = compute_pif_oa_hist(
        observations,
        low_actions,
        timesteps,
        episode_ids,
        k_window=1,
        k=25,
        posterior_alpha=0.01,
    )

    assert n_pairs == 2
    assert high["agent_1"] > 0.85
    assert low["agent_1"] < 0.20


def test_discrete_aa_high_for_action_coupling_low_when_observation_explains_coupling():
    rng = np.random.default_rng(3)
    n_episodes, horizon = 160, 12
    n = n_episodes * horizon
    timesteps, episode_ids = _time_keys(n_episodes, horizon)

    hidden_convention = rng.integers(0, 2, size=n)
    observed_convention = rng.integers(0, 2, size=n)
    high_observations = _obs(rng.normal(size=(n, 1)), rng.normal(size=(n, 1)))
    low_observations = _obs(
        observed_convention.reshape(-1, 1).astype(float),
        observed_convention.reshape(-1, 1).astype(float),
    )
    high_actions = _discrete_actions(hidden_convention, hidden_convention.copy())
    low_actions = _discrete_actions(observed_convention, observed_convention.copy())

    _, high, n_pairs = compute_aa(
        high_observations,
        high_actions,
        timesteps,
        episode_ids,
        k=25,
        posterior_alpha=0.01,
    )
    _, low, _ = compute_aa(
        low_observations,
        low_actions,
        timesteps,
        episode_ids,
        k=25,
        posterior_alpha=0.01,
    )

    assert n_pairs == 2
    assert high["agent_1"] > 0.85
    assert low["agent_1"] < 0.20


def test_discrete_dai_high_for_temporal_influence_low_when_own_history_explains_action():
    rng = np.random.default_rng(4)
    n_episodes, horizon = 160, 12
    n = n_episodes * horizon
    timesteps, episode_ids = _time_keys(n_episodes, horizon)

    agent0_history = rng.integers(0, 2, size=(n_episodes, horizon))
    agent1_history = rng.integers(0, 2, size=(n_episodes, horizon))
    observations = _obs(
        agent0_history.reshape(-1, 1).astype(float),
        agent1_history.reshape(-1, 1).astype(float),
    )
    high_actions = _discrete_actions(
        rng.integers(0, 2, size=n),
        _previous_binary(agent0_history, rng),
    )
    low_actions = _discrete_actions(
        rng.integers(0, 2, size=n),
        _previous_binary(agent1_history, rng),
    )

    _, high, n_pairs = compute_dai_oa_hist(
        observations,
        high_actions,
        timesteps,
        episode_ids,
        k_window=1,
        k=25,
        posterior_alpha=0.01,
    )
    _, low, _ = compute_dai_oa_hist(
        observations,
        low_actions,
        timesteps,
        episode_ids,
        k_window=1,
        k=25,
        posterior_alpha=0.01,
    )

    assert n_pairs == 2
    assert high["agent_1"] > 0.85
    assert low["agent_1"] < 0.20


def test_continuous_action_metrics_separate_signal_from_conditioned_controls():
    rng = np.random.default_rng(5)
    n_episodes, horizon = 160, 12
    n = n_episodes * horizon
    timesteps, episode_ids = _time_keys(n_episodes, horizon)
    eps = 0.001
    control_noise = 0.25

    obs0 = rng.normal(size=(n, 1))
    obs1 = rng.normal(size=(n, 1))
    _, oar_high = compute_oar(
        _obs(obs0, obs1),
        _continuous_actions(obs0[:, 0] + eps * rng.normal(size=n), rng.normal(size=n)),
        k=3,
        force_continuous_A=True,
    )
    _, oar_low = compute_oar(
        _obs(obs0, obs1),
        _continuous_actions(rng.normal(size=n), rng.normal(size=n)),
        k=3,
        force_continuous_A=True,
    )

    latent = rng.normal(size=(n_episodes, horizon))
    observations = _obs(latent.reshape(-1, 1), rng.normal(size=(n, 1)))
    _, har_high = compute_har_ohist(
        observations,
        _continuous_actions(
            _previous(latent, rng) + eps * rng.normal(size=n),
            rng.normal(size=n),
        ),
        timesteps,
        episode_ids,
        k_window=1,
        k=3,
        force_continuous_A=True,
    )
    _, har_low = compute_har_ohist(
        observations,
        _continuous_actions(
            latent.reshape(-1) + control_noise * rng.normal(size=n),
            rng.normal(size=n),
        ),
        timesteps,
        episode_ids,
        k_window=1,
        k=3,
        force_continuous_A=True,
    )

    agent0_private = rng.normal(size=(n, 1))
    agent1_private = rng.normal(size=(n, 1))
    observations = _obs(agent0_private, agent1_private)
    _, pif_high, _ = compute_pif_oa_hist(
        observations,
        _continuous_actions(
            rng.normal(size=n),
            agent0_private[:, 0] + eps * rng.normal(size=n),
        ),
        timesteps,
        episode_ids,
        k_window=1,
        k=3,
        force_continuous_A=True,
    )
    _, pif_low, _ = compute_pif_oa_hist(
        observations,
        _continuous_actions(
            rng.normal(size=n),
            agent1_private[:, 0] + control_noise * rng.normal(size=n),
        ),
        timesteps,
        episode_ids,
        k_window=1,
        k=3,
        force_continuous_A=True,
    )

    hidden_convention = rng.normal(size=n)
    observed_convention = rng.normal(size=(n, 1))
    _, aa_high, _ = compute_aa(
        _obs(rng.normal(size=(n, 1)), rng.normal(size=(n, 1))),
        _continuous_actions(
            hidden_convention,
            hidden_convention + eps * rng.normal(size=n),
        ),
        timesteps,
        episode_ids,
        k=3,
        force_continuous_A=True,
    )
    _, aa_low, _ = compute_aa(
        _obs(observed_convention, observed_convention),
        _continuous_actions(
            observed_convention[:, 0] + control_noise * rng.normal(size=n),
            observed_convention[:, 0] + control_noise * rng.normal(size=n),
        ),
        timesteps,
        episode_ids,
        k=3,
        force_continuous_A=True,
    )

    agent0_history = rng.normal(size=(n_episodes, horizon))
    agent1_history = rng.normal(size=(n_episodes, horizon))
    observations = _obs(
        agent0_history.reshape(-1, 1),
        agent1_history.reshape(-1, 1),
    )
    _, dai_high, _ = compute_dai_oa_hist(
        observations,
        _continuous_actions(
            rng.normal(size=n),
            _previous(agent0_history, rng) + eps * rng.normal(size=n),
        ),
        timesteps,
        episode_ids,
        k_window=1,
        k=3,
        force_continuous_A=True,
    )
    _, dai_low, _ = compute_dai_oa_hist(
        observations,
        _continuous_actions(
            rng.normal(size=n),
            _previous(agent1_history, rng) + control_noise * rng.normal(size=n),
        ),
        timesteps,
        episode_ids,
        k_window=1,
        k=3,
        force_continuous_A=True,
    )

    scores = [
        oar_high["agent_0"],
        oar_low["agent_0"],
        har_high["agent_0"],
        har_low["agent_0"],
        pif_high["agent_1"],
        pif_low["agent_1"],
        aa_high["agent_1"],
        aa_low["agent_1"],
        dai_high["agent_1"],
        dai_low["agent_1"],
    ]
    assert all(np.isfinite(score) for score in scores)
    assert oar_high["agent_0"] > 0.70
    assert oar_low["agent_0"] < 0.15
    assert har_high["agent_0"] > har_low["agent_0"] + 0.05
    assert pif_high["agent_1"] > pif_low["agent_1"] + 0.02
    assert aa_high["agent_1"] > aa_low["agent_1"] + 0.02
    assert dai_high["agent_1"] > dai_low["agent_1"] + 0.02
