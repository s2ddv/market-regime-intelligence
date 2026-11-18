"""Regression checks on the parameters actually returned by an HMM start."""

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from mri.features import build_features
from mri.models import HMMConfig, HMMError, fit_gaussian_hmm
from mri.models import gaussian_hmm as hmm


def sparse_regimes(*, from_prices: bool) -> pd.DataFrame:
    seed, count = (9, 100) if from_prices else (1, 80)
    rng = np.random.default_rng(seed)
    dates = [
        stamp.date()
        for stamp in xcals.get_calendar("XNYS", start="2024-01-01", end="2024-12-31").sessions[
            :count
        ]
    ]
    regime = rng.choice([0, 1], size=count, p=[0.95, 0.05])
    returns = rng.normal(0, 0.01, count) * (1 + regime * 4)
    if not from_prices:
        return pd.DataFrame(
            {
                "date": dates,
                "log_return": returns,
                "realized_volatility_20": 0.1 + rng.uniform(0, 0.02, count) + regime * 0.3,
            }
        )
    prices = 100 * np.exp(np.cumsum(returns))
    return build_features(
        pd.DataFrame(
            {
                "date": dates,
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "adjusted_close": prices,
                "volume": 1000.0,
                "symbol": "SPY",
                "source": "synthetic",
            }
        )
    )


@pytest.mark.parametrize("exit_path", ["iteration_limit", "convergence"])
def test_underoccupied_final_parameters_are_rejected(exit_path):
    limited = exit_path == "iteration_limit"
    config = (
        HMMConfig(n_states=4, n_starts=1, max_iter=1, seed=9)
        if limited
        else HMMConfig(n_states=3, n_starts=1, max_iter=100, tolerance=0.5, seed=1)
    )
    with pytest.raises(HMMError, match="occupancy"):
        fit_gaussian_hmm(sparse_regimes(from_prices=limited), config)


@pytest.mark.parametrize("final_likelihood", [np.nan, -1e9])
def test_last_update_cannot_bypass_likelihood_checks(monkeypatch, final_likelihood):
    original = hmm._expectation
    calls = 0

    def damaged_final_expectation(*args):
        nonlocal calls
        gamma, transitions, likelihood, last = original(*args)
        calls += 1
        if calls == 2:
            likelihood = final_likelihood
        return gamma, transitions, likelihood, last

    monkeypatch.setattr(hmm, "_expectation", damaged_final_expectation)
    with pytest.raises(HMMError, match="likelihood"):
        fit_gaussian_hmm(
            sparse_regimes(from_prices=True),
            HMMConfig(n_states=2, n_starts=1, max_iter=1, min_occupancy=1e-6, seed=9),
        )


def test_failed_start_does_not_prevent_a_later_valid_start(monkeypatch):
    original = hmm._fit_start
    calls = 0

    def fail_first_start(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HMMError("A state has insufficient training occupancy")
        return original(*args)

    monkeypatch.setattr(hmm, "_fit_start", fail_first_start)
    model = fit_gaussian_hmm(
        sparse_regimes(from_prices=True), HMMConfig(n_starts=2, max_iter=1, seed=9)
    )
    assert model.selected_start == 1
    assert calls == 2
    assert not model.converged


def test_occupancy_exactly_at_floor_is_accepted(monkeypatch):
    def balanced_expectation(observations, initial, transition, means, variances):
        count = len(observations)
        return np.full((count, 2), 0.5), np.ones((2, 2)), -10.0, np.full(2, 0.5)

    monkeypatch.setattr(hmm, "_expectation", balanced_expectation)
    model = fit_gaussian_hmm(
        sparse_regimes(from_prices=True),
        HMMConfig(n_starts=1, max_iter=1, min_occupancy=40, seed=9),
    )
    assert model.iterations == 1


def test_invalid_posterior_mass_cannot_satisfy_occupancy_gate(monkeypatch):
    original = hmm._expectation

    def inflated_expectation(*args):
        gamma, transitions, likelihood, last = original(*args)
        return gamma * 2, transitions, likelihood, last

    monkeypatch.setattr(hmm, "_expectation", inflated_expectation)
    with pytest.raises(HMMError, match="state probabilities"):
        fit_gaussian_hmm(
            sparse_regimes(from_prices=True),
            HMMConfig(n_starts=1, max_iter=1, min_occupancy=1e-6, seed=9),
        )
