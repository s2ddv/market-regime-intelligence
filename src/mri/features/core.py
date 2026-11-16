"""Deterministic session features and separately matured forward targets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from mri.data.schema import COLUMNS


class FeatureError(ValueError):
    """Input or feature configuration violates the MRI-004 contract."""


@dataclass(frozen=True)
class FeatureSpec:
    """Window lengths count trading-session rows, not calendar days."""

    volatility_sessions: int = 20
    momentum_sessions: int = 20
    drawdown_sessions: int = 60
    annualization: int = 252

    def __post_init__(self) -> None:
        for value in (
            self.volatility_sessions,
            self.momentum_sessions,
            self.drawdown_sessions,
            self.annualization,
        ):
            if type(value) is not int or value <= 0:
                raise FeatureError("Window lengths and annualization must be positive integers")


def load_feature_spec(path: Path) -> FeatureSpec:
    """Read the frozen feature configuration without applying a fitted transform."""
    raw: Any = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or raw.get("return") != "log_adjusted_close":
        raise FeatureError("Unsupported feature configuration")
    names = {
        "realized_volatility_sessions",
        "momentum_sessions",
        "drawdown_sessions",
        "realized_volatility_annualization",
        "random_seed",
        "return",
    }
    if set(raw) != names or type(raw["random_seed"]) is not int:
        raise FeatureError("Feature configuration keys or seed are invalid")
    return FeatureSpec(
        volatility_sessions=raw["realized_volatility_sessions"],
        momentum_sessions=raw["momentum_sessions"],
        drawdown_sessions=raw["drawdown_sessions"],
        annualization=raw["realized_volatility_annualization"],
    )


def _dates(frame: pd.DataFrame) -> list[date]:
    if frame.empty or "date" not in frame:
        raise FeatureError("Missing session dates")
    dates = frame["date"].tolist()
    if any(type(value) is not date for value in dates):
        raise FeatureError("Session dates must be date values")
    if any(previous >= current for previous, current in zip(dates, dates[1:], strict=False)):
        raise FeatureError("Session dates must be strictly increasing")
    return dates


def build_features(frame: pd.DataFrame, spec: FeatureSpec | None = None) -> pd.DataFrame:
    """Compute backward-looking features from a validated canonical daily table."""
    config = spec or FeatureSpec()
    if list(frame.columns) != COLUMNS:
        raise FeatureError("Expected the canonical daily schema")
    dates = _dates(frame)
    if frame["symbol"].ne("SPY").any() or frame["source"].nunique() != 1:
        raise FeatureError("Mixed symbol or source")
    try:
        prices = frame["adjusted_close"].to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise FeatureError("Adjusted close must be numeric") from exc
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise FeatureError("Adjusted close must be finite and positive")
    log_price = pd.Series(np.log(prices))
    returns = log_price.diff()
    volatility = (
        returns.pow(2)
        .rolling(window=config.volatility_sessions, min_periods=config.volatility_sessions)
        .sum()
        .mul(config.annualization / config.volatility_sessions)
        .pow(0.5)
    )
    momentum = log_price - log_price.shift(config.momentum_sessions)
    price_series = pd.Series(prices)
    drawdown = (
        price_series
        / price_series.rolling(
            window=config.drawdown_sessions, min_periods=config.drawdown_sessions
        ).max()
        - 1.0
    )
    return pd.DataFrame(
        {
            "date": dates,
            "log_return": returns,
            f"realized_volatility_{config.volatility_sessions}": volatility,
            f"momentum_{config.momentum_sessions}": momentum,
            f"drawdown_{config.drawdown_sessions}": drawdown,
        }
    )


def build_forward_targets(
    features: pd.DataFrame, horizons: tuple[int, ...] = (5, 20), annualization: int = 252
) -> pd.DataFrame:
    """Build labels from future returns, isolated from the feature table."""
    dates = _dates(features)
    if "log_return" not in features:
        raise FeatureError("Missing log returns")
    if not horizons or len(set(horizons)) != len(horizons):
        raise FeatureError("Horizons must be nonempty and unique")
    if type(annualization) is not int or annualization <= 0:
        raise FeatureError("Annualization must be positive")
    if any(type(horizon) is not int or horizon <= 0 for horizon in horizons):
        raise FeatureError("Horizons must be positive integers")
    try:
        returns = features["log_return"].to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise FeatureError("Log returns must be numeric") from exc
    if not np.isnan(returns[0]) or not np.isfinite(returns[1:]).all():
        raise FeatureError("Expected a missing first return and finite subsequent returns")
    squared = pd.Series(returns).pow(2)
    result = pd.DataFrame({"date": dates})
    for horizon in horizons:
        variance = squared.rolling(window=horizon, min_periods=horizon).sum().shift(-horizon)
        variance = variance / horizon
        result[f"forward_variance_{horizon}"] = variance
        result[f"forward_volatility_{horizon}"] = (annualization * variance).pow(0.5)
        result[f"label_end_session_{horizon}"] = pd.Series(dates).shift(-horizon)
    return result
