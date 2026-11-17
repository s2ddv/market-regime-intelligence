"""Hand-checkable phase and maturity boundaries for MRI-006."""

from datetime import date

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from mri.evaluation import (
    WalkForwardConfig,
    WalkForwardError,
    build_walk_forward_plan,
    training_event_threshold,
    training_target_rows,
)


def sample():
    calendar = xcals.get_calendar("XNYS", start="2024-01-02", end="2024-06-28")
    dates = [stamp.date() for stamp in calendar.sessions]
    features = pd.DataFrame(
        {
            "date": dates,
            "log_return": np.full(len(dates), 0.01),
            "realized_volatility_20": np.full(len(dates), 0.2),
        }
    )
    ends = [dates[index + 2] if index + 2 < len(dates) else None for index in range(len(dates))]
    maturity = pd.DataFrame({"date": dates, "label_end_session_2": ends})
    values = np.arange(len(dates), dtype=float) / 100 + 0.1
    targets = pd.DataFrame(
        {
            "date": dates,
            "forward_variance_2": values**2 / 252,
            "forward_volatility_2": values,
            "label_end_session_2": ends,
        }
    )
    config = WalkForwardConfig(
        development_end=date(2024, 2, 29),
        validation_end=date(2024, 4, 30),
        horizons=(2,),
        min_training_sessions=30,
    )
    return dates, features, maturity, targets, config


def test_phase_boundaries_purge_and_seal():
    dates, features, maturity, targets, config = sample()
    plan = build_walk_forward_plan(features, maturity, config)
    assert dates[plan.validation_start_index] == date(2024, 3, 1)
    assert dates[plan.final_start_index] == date(2024, 5, 1)
    march, april, may = plan.folds[:3]
    assert [fold.phase for fold in (march, april, may)] == ["validation", "validation", "final"]
    assert dates[march.fit_end_index] == date(2024, 2, 29)
    assert dates[april.fit_end_index] == date(2024, 3, 28)
    assert dates[may.fit_end_index] == date(2024, 4, 30)
    assert march.train_feature_stop == march.prediction_start
    feb_27 = dates.index(date(2024, 2, 27))
    feb_28 = dates.index(date(2024, 2, 28))
    assert feb_27 in march.train_label_indices
    assert feb_28 in march.purged_label_indices
    assert feb_28 not in april.train_label_indices
    assert dates.index(date(2024, 3, 1)) in april.sealed_label_indices
    apr_26 = dates.index(date(2024, 4, 26))
    apr_29 = dates.index(date(2024, 4, 29))
    assert apr_26 in may.train_label_indices
    assert apr_29 in may.purged_label_indices
    assert dates.index(date(2024, 5, 1)) not in may.train_label_indices
    assert len(plan.folds) == 4  # March, April, May, June

    march_threshold = training_event_threshold(plan, march, targets)
    expected = np.quantile(
        targets.iloc[list(march.train_label_indices)]["forward_volatility_2"], 0.9
    )
    assert march_threshold.value == pytest.approx(expected)
    changed = targets.copy()
    changed.loc[plan.validation_start_index :, "forward_volatility_2"] = 1e9
    assert training_event_threshold(plan, march, changed).value == march_threshold.value
    assert training_event_threshold(plan, april, changed).value == march_threshold.value
    # Final may use prior matured validation labels, but never final outcomes.
    may_threshold = training_event_threshold(plan, may, targets)
    changed = targets.copy()
    changed.loc[plan.final_start_index :, "forward_volatility_2"] = np.nan
    assert training_event_threshold(plan, may, changed).value == may_threshold.value
    assert list(training_target_rows(plan, may, changed).columns) == [
        "date",
        "forward_variance_2",
        "forward_volatility_2",
        "label_end_session_2",
    ]


def test_bad_calendar_maturity_and_target_values_fail_closed():
    dates, features, maturity, targets, config = sample()
    with pytest.raises(WalkForwardError, match="consecutive XNYS"):
        build_walk_forward_plan(features.drop(index=8).reset_index(drop=True), maturity, config)
    bad = maturity.copy()
    bad.loc[0, "label_end_session_2"] = dates[1]
    with pytest.raises(WalkForwardError, match="exact horizon"):
        build_walk_forward_plan(features, bad, config)
    with pytest.raises(WalkForwardError, match="only aligned"):
        build_walk_forward_plan(features, maturity.assign(forward_volatility_2=1), config)
    bad_features = features.copy()
    bad_features.loc[dates.index(date(2024, 3, 1)), "log_return"] = np.nan
    with pytest.raises(WalkForwardError, match="complete finite"):
        build_walk_forward_plan(bad_features, maturity, config)
    plan = build_walk_forward_plan(features, maturity, config)
    bad_target = targets.copy()
    bad_target.loc[0, "forward_variance_2"] = np.nan
    with pytest.raises(WalkForwardError, match="finite"):
        training_target_rows(plan, plan.folds[0], bad_target)
    with pytest.raises(WalkForwardError, match="insufficient development"):
        build_walk_forward_plan(
            features,
            maturity,
            WalkForwardConfig(
                development_end=date(2024, 2, 29),
                validation_end=date(2024, 4, 30),
                horizons=(2,),
                min_training_sessions=100,
            ),
        )
