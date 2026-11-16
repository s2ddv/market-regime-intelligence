from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from mri.features import (
    FeatureError,
    FeatureSpec,
    build_features,
    build_forward_targets,
    load_feature_spec,
)


def canonical(prices: list[float]) -> pd.DataFrame:
    sessions = xcals.get_calendar("XNYS", start="2024-01-02", end="2024-06-30")
    dates = [stamp.date() for stamp in sessions.sessions[: len(prices)]]
    return pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": [price * 1.01 for price in prices],
            "low": [price * 0.99 for price in prices],
            "close": prices,
            "adjusted_close": prices,
            "volume": [1000] * len(prices),
            "symbol": ["SPY"] * len(prices),
            "source": ["fixture"] * len(prices),
        }
    )


def test_hand_calculated_features_and_forward_target():
    prices = canonical([100.0, 110.0, 121.0])
    spec = FeatureSpec(volatility_sessions=2, momentum_sessions=2, drawdown_sessions=3)
    features = build_features(prices, spec)
    expected_return = np.log(1.1)
    assert np.isnan(features.loc[0, "log_return"])
    assert features.loc[1, "log_return"] == pytest.approx(expected_return)
    assert features.loc[2, "log_return"] == pytest.approx(expected_return)
    assert features.loc[2, "realized_volatility_2"] == pytest.approx(np.sqrt(252) * expected_return)
    assert features.loc[2, "momentum_2"] == pytest.approx(np.log(1.21))
    assert features.loc[2, "drawdown_3"] == pytest.approx(0.0)
    targets = build_forward_targets(features, horizons=(2,))
    assert targets.loc[0, "forward_variance_2"] == pytest.approx(expected_return**2)
    assert targets.loc[0, "forward_volatility_2"] == pytest.approx(np.sqrt(252) * expected_return)
    assert targets.loc[0, "label_end_session_2"] == prices.loc[2, "date"]
    assert targets["forward_variance_2"].iloc[1:].isna().all()
    assert targets["label_end_session_2"].iloc[1:].isna().all()
    assert build_features(canonical([100.0, 90.0, 95.0]), spec).loc[2, "drawdown_3"] == (
        pytest.approx(-0.05)
    )


def test_default_first_valid_indices_and_target_maturity():
    frame = canonical([100.0 + 0.1 * i for i in range(80)])
    features = build_features(frame)
    targets = build_forward_targets(features)
    assert features["log_return"].first_valid_index() == 1
    assert features["realized_volatility_20"].first_valid_index() == 20
    assert features["momentum_20"].first_valid_index() == 20
    assert features["drawdown_60"].first_valid_index() == 59
    for horizon in (5, 20):
        assert targets[f"forward_variance_{horizon}"].first_valid_index() == 0
        assert targets[f"forward_variance_{horizon}"].last_valid_index() == 79 - horizon
        assert targets[f"label_end_session_{horizon}"].iloc[0] == frame["date"].iloc[horizon]
        assert targets[f"forward_variance_{horizon}"].iloc[-horizon:].isna().all()


def test_feature_prefix_is_invariant_to_future_prices():
    frame = canonical([100.0 + 0.1 * i for i in range(80)])
    full = build_features(frame)
    prefix = build_features(frame.iloc[:60].copy())
    pd.testing.assert_frame_equal(full.iloc[:60].reset_index(drop=True), prefix)
    changed = frame.copy()
    changed.loc[60, ["open", "high", "low", "close", "adjusted_close"]] *= 1.1
    changed_features = build_features(changed)
    pd.testing.assert_frame_equal(full.iloc[:60], changed_features.iloc[:60])
    original_targets = build_forward_targets(full)
    changed_targets = build_forward_targets(changed_features)
    assert (
        original_targets.loc[55, "forward_variance_5"]
        != changed_targets.loc[55, "forward_variance_5"]
    )


def test_invalid_feature_inputs_and_config():
    frame = canonical([100.0, 101.0, 102.0])
    with pytest.raises(FeatureError):
        build_features(frame.assign(adjusted_close=[100.0, -1.0, 102.0]))
    duplicate = frame.copy()
    duplicate.loc[1, "date"] = duplicate.loc[0, "date"]
    with pytest.raises(FeatureError):
        build_features(duplicate)
    with pytest.raises(FeatureError):
        FeatureSpec(volatility_sessions=0)
    spec = load_feature_spec(Path("configs/features.yaml"))
    assert spec == FeatureSpec()
    with pytest.raises(FeatureError):
        build_forward_targets(build_features(frame), horizons=(2, 2))
    with pytest.raises(FeatureError):
        build_forward_targets(build_features(frame), horizons=(0,))
