"""Numerical and input-contract tests for MRI-005 using synthetic observations."""

from datetime import date

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from mri.models import FittedGaussianHMM, HMMConfig, HMMError, filter_new_sessions, fit_gaussian_hmm


def _frame(count: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(20261116)
    calendar = xcals.get_calendar("XNYS", start="2024-01-02", end="2025-12-31")
    dates = [stamp.date() for stamp in calendar.sessions[:count]]
    risk = np.where(np.arange(count) % 50 < 25, 0.10, 0.35)
    returns = rng.normal(0, risk / 16)
    volatility = risk + rng.normal(0, 0.012, count)
    returns[:20] = np.nan
    volatility[:20] = np.nan
    return pd.DataFrame(
        {"date": dates, "log_return": returns, "realized_volatility_20": volatility}
    )


def test_fit_is_seeded_normalized_and_training_only() -> None:
    frame = _frame()
    config = HMMConfig(n_states=2, n_starts=2, max_iter=50, seed=14)
    left = fit_gaussian_hmm(frame.iloc[:120], config)
    right = fit_gaussian_hmm(frame.iloc[:120].assign(ignored_future_target=999), config)
    assert left.train_end == frame.iloc[119]["date"]
    assert np.isfinite(left.log_likelihood)
    assert np.allclose(left.initial.sum(), 1)
    assert np.allclose(left.transition.sum(axis=1), 1)
    assert np.all(left.variances >= config.min_variance)
    assert left.means[0, 1] <= left.means[1, 1]
    assert np.array_equal(left.means, right.means)
    assert not left.transition.flags.writeable
    filtered = filter_new_sessions(left, frame.iloc[120:])
    assert len(filtered) == 30
    assert np.allclose(filtered.iloc[:, 1:].sum(axis=1), 1)


def test_future_observation_cannot_change_prior_filtered_probability() -> None:
    frame = _frame()
    model = fit_gaussian_hmm(frame.iloc[:100], HMMConfig(n_starts=2, max_iter=40))
    original = filter_new_sessions(model, frame.iloc[100:110])
    altered = frame.iloc[100:110].copy()
    altered.loc[altered.index[-1], "realized_volatility_20"] = 2.0
    changed = filter_new_sessions(model, altered)
    np.testing.assert_array_equal(
        original.iloc[:-1, 1:].to_numpy(), changed.iloc[:-1, 1:].to_numpy()
    )


def test_fixed_parameter_filter_matches_hand_calculation() -> None:
    config = HMMConfig(n_states=2)
    model = FittedGaussianHMM(
        train_start=date(2024, 1, 1),
        train_end=date(2024, 1, 2),
        feature_columns=("log_return", "realized_volatility_20"),
        scale_mean=np.array([0.0, 0.0]),
        scale_std=np.array([1.0, 1.0]),
        initial=np.array([0.5, 0.5]),
        transition=np.array([[0.9, 0.1], [0.2, 0.8]]),
        means=np.array([[0.0, 0.0], [2.0, 2.0]]),
        variances=np.ones((2, 2)),
        last_filtered=np.array([0.75, 0.25]),
        log_likelihood=0.0,
        iterations=1,
        converged=True,
        selected_start=0,
        config=config,
    )
    frame = pd.DataFrame(
        {"date": [date(2024, 1, 3)], "log_return": [0.0], "realized_volatility_20": [0.0]}
    )
    result = filter_new_sessions(model, frame)
    prior = np.array([0.75, 0.25]) @ model.transition
    weighted = prior * np.array([1.0, np.exp(-4.0)])
    expected = weighted / weighted.sum()  # Common Gaussian normalization cancels.
    np.testing.assert_allclose(result.iloc[0, 1:].to_numpy(dtype=float), expected)


@pytest.mark.parametrize("change", ["reverse_dates", "interior_nan", "constant", "short"])
def test_invalid_training_inputs_fail(change: str) -> None:
    frame = _frame()
    if change == "reverse_dates":
        frame.loc[70, "date"] = frame.loc[69, "date"]
    elif change == "interior_nan":
        frame.loc[70, "log_return"] = np.nan
    elif change == "constant":
        frame.loc[:, "realized_volatility_20"] = 0.1
    else:
        frame = frame.iloc[:50]
    with pytest.raises(HMMError):
        fit_gaussian_hmm(frame)


def test_filter_rejects_training_overlap_and_missing_observations() -> None:
    frame = _frame()
    model = fit_gaussian_hmm(frame.iloc[:100], HMMConfig(n_starts=1, max_iter=20))
    with pytest.raises(HMMError, match="after the training cutoff"):
        filter_new_sessions(model, frame.iloc[99:102])
    broken = frame.iloc[100:103].copy()
    broken.loc[101, "realized_volatility_20"] = np.nan
    with pytest.raises(HMMError, match="Incomplete"):
        filter_new_sessions(model, broken)
    with pytest.raises(HMMError, match="consecutive XNYS"):
        filter_new_sessions(model, frame.iloc[101:103])


def test_config_rejects_invalid_state_count() -> None:
    with pytest.raises(HMMError):
        HMMConfig(n_states=5)


@pytest.mark.parametrize("states", [3, 4])
def test_supported_higher_state_counts(states: int) -> None:
    frame = _frame(300)
    rng = np.random.default_rng(49)
    risk = np.array([0.08, 0.16, 0.28, 0.42])[(np.arange(len(frame)) // 30) % states]
    frame.loc[20:, "log_return"] = rng.normal(0, risk[20:] / 16)
    frame.loc[20:, "realized_volatility_20"] = risk[20:] + rng.normal(0, 0.006, len(frame) - 20)
    model = fit_gaussian_hmm(frame.iloc[:270], HMMConfig(n_states=states, n_starts=3, max_iter=35))
    filtered = filter_new_sessions(model, frame.iloc[270:])
    assert filtered.shape == (30, states + 1)
    assert np.allclose(filtered.iloc[:, 1:].sum(axis=1), 1)
