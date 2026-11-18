"""Synthetic features through a planned monthly HMM refit and filter."""

from datetime import date

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from mri.evaluation import WalkForwardConfig, build_walk_forward_plan
from mri.features import build_features, build_forward_targets
from mri.models import HMMConfig, filter_new_sessions, fit_gaussian_hmm


def test_monthly_plan_binds_hmm_fit_and_filter():
    # Include warm-up before the 252 complete training sessions required by the plan.
    calendar = xcals.get_calendar("XNYS", start="2023-11-01", end="2025-04-30")
    sessions = [stamp.date() for stamp in calendar.sessions]
    rng = np.random.default_rng(606)
    shocks = rng.normal(0, np.where(np.arange(len(sessions)) % 90 < 45, 0.006, 0.018))
    prices = 100 * np.exp(np.cumsum(shocks))
    canonical = pd.DataFrame(
        {
            "date": sessions,
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "adjusted_close": prices,
            "volume": np.full(len(sessions), 1000),
            "symbol": ["SPY"] * len(sessions),
            "source": ["fixture"] * len(sessions),
        }
    )
    features = build_features(canonical)
    targets = build_forward_targets(features, horizons=(5,))
    maturity_only = targets[["date", "label_end_session_5"]]
    plan = build_walk_forward_plan(
        features,
        maturity_only,
        WalkForwardConfig(
            development_end=date(2024, 12, 31),
            validation_end=date(2025, 3, 31),
            horizons=(5,),
        ),
    )
    fold = plan.folds[0]
    assert sessions[fold.prediction_start] == date(2025, 1, 2)
    model = fit_gaussian_hmm(
        features.iloc[: fold.train_feature_stop], HMMConfig(n_starts=1, max_iter=30)
    )
    filtered = filter_new_sessions(
        model, features.iloc[fold.prediction_start : fold.prediction_stop]
    )
    assert model.train_end == sessions[fold.fit_end_index]
    assert filtered["date"].tolist() == list(sessions[fold.prediction_start : fold.prediction_stop])
    assert np.allclose(filtered.iloc[:, 1:].sum(axis=1), 1)
