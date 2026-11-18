"""Training-only diagonal Gaussian HMM and frozen-parameter forward filter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import logsumexp  # type: ignore[import-untyped]

from mri.features._validation import observation_values
from mri.features.core import FeatureError


class HMMError(ValueError):
    """Invalid data, configuration, or failed numerical model fit."""


@dataclass(frozen=True)
class HMMConfig:
    n_states: int = 2
    n_starts: int = 3
    max_iter: int = 100
    tolerance: float = 1e-4
    min_variance: float = 1e-3
    min_occupancy: float = 2.0
    seed: int = 20261116

    def __post_init__(self) -> None:
        for name in ("n_states", "n_starts", "max_iter", "seed"):
            if type(getattr(self, name)) is not int:
                raise HMMError(f"{name} must be an integer")
        if not 2 <= self.n_states <= 4 or self.n_starts < 1 or self.max_iter < 1 or self.seed < 0:
            raise HMMError("Unsupported state count, starts, iterations, or seed")
        for name in ("tolerance", "min_variance", "min_occupancy"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise HMMError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class FittedGaussianHMM:
    """A model fitted through train_end, with validated read-only parameters."""

    train_start: date
    train_end: date
    feature_columns: tuple[str, str]
    scale_mean: NDArray[np.float64]
    scale_std: NDArray[np.float64]
    initial: NDArray[np.float64]
    transition: NDArray[np.float64]
    means: NDArray[np.float64]
    variances: NDArray[np.float64]
    last_filtered: NDArray[np.float64]
    log_likelihood: float
    iterations: int
    converged: bool
    selected_start: int
    config: HMMConfig

    def __post_init__(self) -> None:
        states = self.config.n_states
        shapes = {
            "scale_mean": (2,),
            "scale_std": (2,),
            "initial": (states,),
            "transition": (states, states),
            "means": (states, 2),
            "variances": (states, 2),
            "last_filtered": (states,),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise HMMError(f"Invalid fitted parameter {name}")
            object.__setattr__(self, name, _freeze(value))
        if self.train_start > self.train_end or self.feature_columns != (
            "log_return",
            "realized_volatility_20",
        ):
            raise HMMError("Invalid fitted training dates or feature columns")
        if (self.scale_std <= 0).any() or (self.variances <= 0).any():
            raise HMMError("Fitted scales and variances must be positive")
        if (
            (self.initial <= 0).any()
            or (self.transition <= 0).any()
            or (self.last_filtered < 0).any()
        ):
            raise HMMError("Fitted probabilities must be valid")
        if (
            not np.isclose(self.initial.sum(), 1)
            or not np.allclose(self.transition.sum(axis=1), 1)
            or not np.isclose(self.last_filtered.sum(), 1)
        ):
            raise HMMError("Fitted probabilities must sum to one")
        if not np.isfinite(self.log_likelihood) or self.iterations < 1:
            raise HMMError("Invalid fitted diagnostics")


def _freeze(array: NDArray[np.float64]) -> NDArray[np.float64]:
    frozen = np.array(array, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    return frozen


def _observations(
    frame: pd.DataFrame, columns: tuple[str, str], *, allow_warmup: bool
) -> tuple[list[date], NDArray[np.float64]]:
    if frame.empty or "date" not in frame or any(column not in frame for column in columns):
        raise HMMError("Missing dates or required observation columns")
    dates = frame["date"].tolist()
    if any(type(value) is not date for value in dates):
        raise HMMError("Session dates must be date values")
    if any(left >= right for left, right in zip(dates, dates[1:], strict=False)):
        raise HMMError("Session dates must be strictly increasing")
    try:
        first, values = observation_values(frame, columns, allow_warmup=allow_warmup)
    except FeatureError as exc:
        raise HMMError(str(exc)) from exc
    return dates[first:], values


def _require_contiguous_sessions(dates: list[date], start: date | None = None) -> None:
    first = start or dates[0]
    calendar = xcals.get_calendar(
        "XNYS", start=first - timedelta(days=14), end=dates[-1] + timedelta(days=14)
    )
    expected = [stamp.date() for stamp in calendar.sessions_in_range(first, dates[-1])]
    if start is not None:
        expected = [session for session in expected if session > start]
    if dates != expected:
        raise HMMError("Observation rows must cover consecutive XNYS sessions")


def _log_emissions(
    observations: NDArray[np.float64], means: NDArray[np.float64], variances: NDArray[np.float64]
) -> NDArray[np.float64]:
    residual = observations[:, None, :] - means[None, :, :]
    return -0.5 * np.sum(
        np.log(2.0 * np.pi * variances)[None, :, :] + residual**2 / variances[None, :, :],
        axis=2,
    )


def _forward(
    log_emissions: NDArray[np.float64],
    initial: NDArray[np.float64],
    transition: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    count, states = log_emissions.shape
    log_a = np.log(transition)
    alpha = np.empty((count, states), dtype=np.float64)
    scales = np.empty(count, dtype=np.float64)
    row = np.log(initial) + log_emissions[0]
    scales[0] = logsumexp(row)
    alpha[0] = row - scales[0]
    for index in range(1, count):
        row = logsumexp(alpha[index - 1, :, None] + log_a, axis=0) + log_emissions[index]
        scales[index] = logsumexp(row)
        alpha[index] = row - scales[index]
    return alpha, scales, float(scales.sum())


def _expectation(
    observations: NDArray[np.float64],
    initial: NDArray[np.float64],
    transition: NDArray[np.float64],
    means: NDArray[np.float64],
    variances: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, NDArray[np.float64]]:
    emissions = _log_emissions(observations, means, variances)
    alpha, scales, likelihood = _forward(emissions, initial, transition)
    count, states = alpha.shape
    beta = np.zeros((count, states), dtype=np.float64)
    log_a = np.log(transition)
    for index in range(count - 2, -1, -1):
        beta[index] = (
            logsumexp(log_a + emissions[index + 1][None, :] + beta[index + 1][None, :], axis=1)
            - scales[index + 1]
        )
    log_gamma = alpha + beta
    log_gamma -= logsumexp(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)
    transitions = np.zeros((states, states), dtype=np.float64)
    for index in range(count - 1):
        log_xi = (
            alpha[index, :, None]
            + log_a
            + emissions[index + 1][None, :]
            + beta[index + 1][None, :]
            - scales[index + 1]
        )
        transitions += np.exp(log_xi - logsumexp(log_xi))
    return gamma, transitions, likelihood, np.exp(alpha[-1])


def _initialize(
    observations: NDArray[np.float64], states: int, rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    quantiles = np.linspace(0.1, 0.9, states)
    volatility_centers = np.quantile(observations[:, 1], quantiles)
    selected = np.abs(observations[:, 1, None] - volatility_centers).argmin(axis=0)
    means = observations[selected].copy() + rng.normal(0.0, 0.15, size=(states, 2))
    variances = np.tile(np.maximum(observations.var(axis=0), 0.1), (states, 1))
    transition = np.full((states, states), 0.1 / (states - 1), dtype=np.float64)
    np.fill_diagonal(transition, 0.9)
    transition = 0.9 * transition + 0.1 * rng.dirichlet(np.ones(states), size=states)
    initial = np.full(states, 1.0 / states, dtype=np.float64)
    return initial, transition, means, variances


def _validate_fit_state(
    gamma: NDArray[np.float64], likelihood: float, previous: float, minimum: float
) -> NDArray[np.float64]:
    """Gate the current parameter set before convergence, updating, or return."""
    if not np.isfinite(likelihood):
        raise HMMError("Non-finite training likelihood")
    if likelihood < previous - 1e-6:
        raise HMMError("Training likelihood decreased beyond numerical tolerance")
    if (
        not np.isfinite(gamma).all()
        or (gamma < 0).any()
        or not np.allclose(gamma.sum(axis=1), 1.0, atol=1e-10, rtol=1e-10)
    ):
        raise HMMError("Invalid training state probabilities")
    occupancy = gamma.sum(axis=0)
    if (occupancy < minimum).any():
        raise HMMError("A state has insufficient training occupancy")
    return occupancy


def _fit_start(
    observations: NDArray[np.float64], config: HMMConfig, rng: np.random.Generator
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    int,
    bool,
]:
    initial, transition, means, variances = _initialize(observations, config.n_states, rng)
    previous = -np.inf
    for iteration in range(1, config.max_iter + 1):
        gamma, transitions, likelihood, last = _expectation(
            observations, initial, transition, means, variances
        )
        occupancy = _validate_fit_state(gamma, likelihood, previous, config.min_occupancy)
        if iteration > 1 and likelihood - previous <= config.tolerance * (1 + abs(previous)):
            return initial, transition, means, variances, last, likelihood, iteration, True
        previous = likelihood
        initial = np.maximum(gamma[0], 1e-12)
        initial /= initial.sum()
        transition = np.maximum(transitions, 1e-12)
        transition /= transition.sum(axis=1, keepdims=True)
        means = gamma.T @ observations / occupancy[:, None]
        differences = observations[:, None, :] - means[None, :, :]
        variances = np.maximum(
            np.einsum("ts,tsd->sd", gamma, differences**2) / occupancy[:, None],
            config.min_variance,
        )
    gamma, _, likelihood, last = _expectation(observations, initial, transition, means, variances)
    _validate_fit_state(gamma, likelihood, previous, config.min_occupancy)
    return initial, transition, means, variances, last, likelihood, iteration, False


def fit_gaussian_hmm(frame: pd.DataFrame, config: HMMConfig | None = None) -> FittedGaussianHMM:
    """Fit on the supplied historical prefix only; no future labels are accepted."""
    settings = config or HMMConfig()
    columns = ("log_return", "realized_volatility_20")
    dates, raw = _observations(frame, columns, allow_warmup=True)
    _require_contiguous_sessions(dates)
    if len(raw) < max(60, 10 * settings.n_states):
        raise HMMError("Insufficient complete training sessions")
    scale_mean = raw.mean(axis=0)
    scale_std = raw.std(axis=0)
    if not np.isfinite(scale_std).all() or (scale_std < 1e-12).any():
        raise HMMError("A training observation column is constant")
    observations = (raw - scale_mean) / scale_std
    master = np.random.default_rng(settings.seed)
    best: (
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            float,
            int,
            bool,
        ]
        | None
    ) = None
    best_start = -1
    failures: list[str] = []
    for start in range(settings.n_starts):
        rng = np.random.default_rng(int(master.integers(0, 2**32)))
        try:
            result = _fit_start(observations, settings, rng)
        except HMMError as exc:
            failures.append(f"start {start}: {exc}")
            continue
        if best is None or result[5] > best[5]:
            best, best_start = result, start
    if best is None:
        raise HMMError("All seeded HMM fits failed diagnostic gates: " + "; ".join(failures))
    initial, transition, means, variances, last, likelihood, iterations, converged = best
    risk_means = means[:, 1] * scale_std[1] + scale_mean[1]
    order = np.lexsort((np.arange(settings.n_states), risk_means))
    return FittedGaussianHMM(
        train_start=dates[0],
        train_end=dates[-1],
        feature_columns=columns,
        scale_mean=_freeze(scale_mean),
        scale_std=_freeze(scale_std),
        initial=_freeze(initial[order]),
        transition=_freeze(transition[np.ix_(order, order)]),
        means=_freeze(means[order]),
        variances=_freeze(variances[order]),
        last_filtered=_freeze(last[order]),
        log_likelihood=likelihood,
        iterations=iterations,
        converged=converged,
        selected_start=best_start,
        config=settings,
    )


def filter_new_sessions(model: FittedGaussianHMM, frame: pd.DataFrame) -> pd.DataFrame:
    """Filter strictly later observations with parameters frozen at train_end."""
    dates, raw = _observations(frame, model.feature_columns, allow_warmup=False)
    if dates[0] <= model.train_end:
        raise HMMError("Filter input must begin after the training cutoff")
    _require_contiguous_sessions(dates, start=model.train_end)
    observations = (raw - model.scale_mean) / model.scale_std
    emissions = _log_emissions(observations, model.means, model.variances)
    posterior = np.array(model.last_filtered, copy=True)
    probabilities = np.empty((len(dates), model.config.n_states), dtype=np.float64)
    for index, emission in enumerate(emissions):
        prior = posterior @ model.transition
        log_row = np.log(prior) + emission
        posterior = np.exp(log_row - logsumexp(log_row))
        probabilities[index] = posterior
    return pd.DataFrame(
        {
            "date": dates,
            **{
                f"state_{state}_probability": probabilities[:, state]
                for state in range(model.config.n_states)
            },
        }
    )
