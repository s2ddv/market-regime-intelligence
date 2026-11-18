"""The planner and HMM must agree on usable observations and warm-up."""

from datetime import date

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from mri.evaluation import WalkForwardConfig, WalkForwardError, build_walk_forward_plan
from mri.features import build_features, build_forward_targets
from mri.models import HMMConfig, HMMError, fit_gaussian_hmm


def sample():
    dates = [
        stamp.date()
        for stamp in xcals.get_calendar("XNYS", start="2024-01-02", end="2024-06-28").sessions
    ]
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(17).normal(0, 0.01, len(dates))))
    features = build_features(
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
    targets = build_forward_targets(features, horizons=(5,))
    return features, targets[["date", "label_end_session_5"]]


def config(end=date(2024, 3, 31), minimum=60):
    return WalkForwardConfig(
        development_end=end,
        validation_end=date(2024, 5, 31),
        horizons=(5,),
        min_training_sessions=minimum,
    )


def test_warmup_rows_do_not_satisfy_training_minimum():
    features, maturity = sample()
    training = features[features.date <= date(2024, 3, 31)]
    assert len(training) == 61
    assert training[["log_return", "realized_volatility_20"]].dropna().shape[0] == 41
    with pytest.raises(WalkForwardError, match="insufficient development"):
        build_walk_forward_plan(features, maturity, config())


@pytest.mark.parametrize("warmup", [False, True])
def test_exact_minimum_is_accepted_but_one_more_is_not(warmup):
    features, maturity = sample()
    if not warmup:
        features = features.iloc[20:].reset_index(drop=True)
        maturity = maturity.iloc[20:].reset_index(drop=True)
    end = date(2024, 4, 30)
    count = (
        features.loc[features.date <= end, ["log_return", "realized_volatility_20"]]
        .dropna()
        .shape[0]
    )
    plan = build_walk_forward_plan(features, maturity, config(end, count))
    fold = plan.folds[0]
    model = fit_gaussian_hmm(
        features.iloc[: fold.train_feature_stop], HMMConfig(n_starts=2, max_iter=10)
    )
    assert model.train_end == end
    with pytest.raises(WalkForwardError, match="insufficient development"):
        build_walk_forward_plan(features, maturity, config(end, count + 1))


@pytest.mark.parametrize(
    "row,column,value",
    [
        (30, "log_return", np.nan),
        (30, "log_return", np.inf),
        (30, "realized_volatility_20", -0.1),
        (10, "log_return", np.inf),
        (10, "realized_volatility_20", -0.1),
        (20, "log_return", np.nan),
    ],
)
def test_invalid_training_values_are_rejected_by_both_consumers(row, column, value):
    features, maturity = sample()
    features.loc[row, column] = value
    with pytest.raises(WalkForwardError):
        build_walk_forward_plan(features, maturity, config(date(2024, 4, 30)))
    with pytest.raises(HMMError):
        fit_gaussian_hmm(features[features.date <= date(2024, 4, 30)])


def test_no_complete_development_observations_fail_before_planning():
    features, maturity = sample()
    features.loc[features.date <= date(2024, 4, 30), "realized_volatility_20"] = np.nan
    with pytest.raises(WalkForwardError):
        build_walk_forward_plan(features, maturity, config(date(2024, 4, 30)))
