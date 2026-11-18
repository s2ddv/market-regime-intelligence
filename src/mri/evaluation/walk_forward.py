"""Maturity-only monthly walk-forward planner and restricted training-target access."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from mri.features._validation import observation_values
from mri.features.core import FeatureError


class WalkForwardError(ValueError):
    """A temporal boundary, session table or label gate is invalid."""


@dataclass(frozen=True)
class WalkForwardConfig:
    development_end: date = date(2015, 12, 31)
    validation_end: date = date(2020, 12, 31)
    horizons: tuple[int, ...] = (5, 20)
    min_training_sessions: int = 252
    event_quantile: float = 0.90

    def __post_init__(self) -> None:
        if type(self.development_end) is not date or type(self.validation_end) is not date:
            raise WalkForwardError("Phase boundaries must be dates")
        if self.development_end >= self.validation_end:
            raise WalkForwardError("Development must end before validation")
        if (
            not self.horizons
            or len(set(self.horizons)) != len(self.horizons)
            or any(type(horizon) is not int or horizon <= 0 for horizon in self.horizons)
        ):
            raise WalkForwardError("Horizons must be unique positive integers")
        if type(self.min_training_sessions) is not int or self.min_training_sessions < 1:
            raise WalkForwardError("Minimum training sessions must be positive")
        if not np.isfinite(self.event_quantile) or not 0 < self.event_quantile < 1:
            raise WalkForwardError("Event quantile must be between zero and one")


@dataclass(frozen=True)
class WalkForwardFold:
    phase: str
    horizon: int
    fit_end_index: int
    prediction_start: int
    prediction_stop: int
    phase_start_index: int
    train_label_indices: tuple[int, ...]
    purged_label_indices: tuple[int, ...]
    sealed_label_indices: tuple[int, ...]
    embargo_sessions: int = 0

    @property
    def train_feature_stop(self) -> int:
        """Exclusive stop; features at indices [0, stop) may be fitted."""
        return self.prediction_start


@dataclass(frozen=True)
class WalkForwardPlan:
    dates: tuple[date, ...]
    folds: tuple[WalkForwardFold, ...]
    validation_start_index: int
    final_start_index: int | None
    config: WalkForwardConfig


@dataclass(frozen=True)
class EventThreshold:
    phase: str
    horizon: int
    fit_end: date
    quantile: float
    value: float
    training_label_count: int


def _dates(frame: pd.DataFrame) -> tuple[date, ...]:
    if frame.empty or "date" not in frame:
        raise WalkForwardError("Missing session dates")
    dates = tuple(frame["date"].tolist())
    if any(type(value) is not date for value in dates):
        raise WalkForwardError("Session dates must be date values")
    if any(left >= right for left, right in zip(dates, dates[1:], strict=False)):
        raise WalkForwardError("Session dates must be strictly increasing")
    return dates


def _require_calendar(dates: tuple[date, ...]) -> None:
    calendar = xcals.get_calendar(
        "XNYS", start=dates[0] - timedelta(days=14), end=dates[-1] + timedelta(days=14)
    )
    expected = tuple(stamp.date() for stamp in calendar.sessions_in_range(dates[0], dates[-1]))
    if dates != expected:
        raise WalkForwardError("Expected consecutive XNYS sessions")


def _maturities(
    maturity_only: pd.DataFrame, dates: tuple[date, ...], horizons: tuple[int, ...]
) -> dict[int, tuple[int | None, ...]]:
    expected_columns = ["date", *(f"label_end_session_{horizon}" for horizon in horizons)]
    if list(maturity_only.columns) != expected_columns or _dates(maturity_only) != dates:
        raise WalkForwardError("Maturity table must contain only aligned dates and label ends")
    result: dict[int, tuple[int | None, ...]] = {}
    for horizon in horizons:
        ends: list[int | None] = []
        for index, value in enumerate(maturity_only[f"label_end_session_{horizon}"].tolist()):
            expected_index = index + horizon if index + horizon < len(dates) else None
            expected_date = dates[expected_index] if expected_index is not None else None
            actual_date = None if pd.isna(value) else value
            if actual_date != expected_date or (
                actual_date is not None and type(actual_date) is not date
            ):
                raise WalkForwardError("Label maturity must equal the exact horizon session")
            ends.append(expected_index)
        result[horizon] = tuple(ends)
    return result


def build_walk_forward_plan(
    features: pd.DataFrame,
    maturity_only: pd.DataFrame,
    config: WalkForwardConfig | None = None,
) -> WalkForwardPlan:
    """Plan folds without receiving or reading any forward target values."""
    settings = config or WalkForwardConfig()
    dates = _dates(features)
    _require_calendar(dates)
    ends_by_horizon = _maturities(maturity_only, dates, settings.horizons)
    validation_start = next(
        (index for index, session in enumerate(dates) if session > settings.development_end), None
    )
    if validation_start is None or validation_start < settings.min_training_sessions:
        raise WalkForwardError("No validation period or insufficient development sessions")
    final_start = next(
        (index for index, session in enumerate(dates) if session > settings.validation_end), None
    )
    if final_start is not None and final_start <= validation_start:
        raise WalkForwardError("Validation period has no observed sessions")
    columns = ("log_return", "realized_volatility_20")
    try:
        _, training_values = observation_values(
            features.iloc[:validation_start], columns, allow_warmup=True
        )
    except FeatureError as exc:
        raise WalkForwardError(f"Invalid development features: {exc}") from exc
    if len(training_values) < settings.min_training_sessions:
        raise WalkForwardError("No validation period or insufficient development sessions")
    # The first valid prefix establishes the minimum. Every later prefix only
    # adds fully observed rows, so readiness holds for all expanding folds.
    try:
        observation_values(features.iloc[validation_start:], columns, allow_warmup=False)
    except FeatureError as exc:
        raise WalkForwardError(
            f"Prediction origins require complete finite features: {exc}"
        ) from exc

    folds: list[WalkForwardFold] = []
    start = validation_start
    while start < len(dates):
        phase = "validation" if final_start is None or start < final_start else "final"
        phase_start = validation_start if phase == "validation" else final_start
        assert phase_start is not None
        stop = start + 1
        while (
            stop < len(dates)
            and (dates[stop].year, dates[stop].month) == (dates[start].year, dates[start].month)
            and (final_start is None or not (start < final_start <= stop))
        ):
            stop += 1
        fit_end = start - 1
        for horizon in settings.horizons:
            ends = ends_by_horizon[horizon]
            eligible_list: list[int] = []
            purged_list: list[int] = []
            for index in range(phase_start):
                end = ends[index]
                if end is not None and end <= fit_end and end < phase_start:
                    eligible_list.append(index)
                else:
                    purged_list.append(index)
            folds.append(
                WalkForwardFold(
                    phase=phase,
                    horizon=horizon,
                    fit_end_index=fit_end,
                    prediction_start=start,
                    prediction_stop=stop,
                    phase_start_index=phase_start,
                    train_label_indices=tuple(eligible_list),
                    purged_label_indices=tuple(purged_list),
                    sealed_label_indices=tuple(range(phase_start, start)),
                )
            )
        start = stop
    return WalkForwardPlan(dates, tuple(folds), validation_start, final_start, settings)


def training_target_rows(
    plan: WalkForwardPlan, fold: WalkForwardFold, targets: pd.DataFrame
) -> pd.DataFrame:
    """Reveal only a fold's eligible variance and volatility labels."""
    if fold not in plan.folds:
        raise WalkForwardError("Fold does not belong to this plan")
    if _dates(targets) != plan.dates:
        raise WalkForwardError("Targets do not align with planned sessions")
    columns = [
        "date",
        f"forward_variance_{fold.horizon}",
        f"forward_volatility_{fold.horizon}",
        f"label_end_session_{fold.horizon}",
    ]
    if any(column not in targets for column in columns):
        raise WalkForwardError("Missing target columns")
    selected = targets.iloc[list(fold.train_label_indices)][columns].copy().reset_index(drop=True)
    if selected.empty:
        raise WalkForwardError("No matured training labels in this fold")
    expected_ends = [plan.dates[index + fold.horizon] for index in fold.train_label_indices]
    if selected[columns[-1]].tolist() != expected_ends:
        raise WalkForwardError("Training target maturities do not match the plan")
    try:
        values = selected[columns[1:3]].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise WalkForwardError("Training targets must be numeric") from exc
    if not np.isfinite(values).all() or (values < 0).any():
        raise WalkForwardError("Training targets must be finite and nonnegative")
    return selected


def training_event_threshold(
    plan: WalkForwardPlan, fold: WalkForwardFold, targets: pd.DataFrame
) -> EventThreshold:
    """Compute a training-only event threshold for a frozen prediction block."""
    selected = training_target_rows(plan, fold, targets)
    values = selected[f"forward_volatility_{fold.horizon}"].to_numpy(dtype=float)
    return EventThreshold(
        phase=fold.phase,
        horizon=fold.horizon,
        fit_end=plan.dates[fold.fit_end_index],
        quantile=plan.config.event_quantile,
        value=float(np.quantile(values, plan.config.event_quantile, method="linear")),
        training_label_count=len(values),
    )
