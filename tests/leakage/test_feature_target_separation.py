"""Regression checks for MRI-004 look-ahead boundaries."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from mri.features import FeatureSpec, build_features, build_forward_targets


def test_future_return_is_label_only():
    dates = [date(2024, 1, day) for day in (2, 3, 4, 5)]
    prices = [100.0, 100.0, 100.0, 200.0]
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "adjusted_close": prices,
            "volume": [1] * 4,
            "symbol": ["SPY"] * 4,
            "source": ["synthetic"] * 4,
        }
    )
    features = build_features(
        frame, FeatureSpec(volatility_sessions=2, momentum_sessions=2, drawdown_sessions=3)
    )
    assert features.loc[2, "log_return"] == 0
    assert features.loc[2, "realized_volatility_2"] == 0
    assert features.loc[2, "momentum_2"] == 0
    targets = build_forward_targets(features, horizons=(1,))
    assert targets.loc[2, "forward_variance_1"] == pytest.approx(np.log(2) ** 2)
    assert targets.loc[2, "label_end_session_1"] == dates[3]
