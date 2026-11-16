"""Validated synthetic SPY snapshot through features and frozen HMM filtering."""

from datetime import date

import exchange_calendars as xcals
import numpy as np

from mri.data.provider import SourceResponse
from mri.data.storage import persist, reload_snapshot
from mri.features import build_features
from mri.models import HMMConfig, filter_new_sessions, fit_gaussian_hmm


def test_synthetic_snapshot_to_causal_state_probabilities(tmp_path):
    calendar = xcals.get_calendar("XNYS", start="2024-01-02", end="2025-03-31")
    sessions = [stamp.date() for stamp in calendar.sessions[:220]]
    rng = np.random.default_rng(31)
    risk = np.where(np.arange(len(sessions)) % 60 < 30, 0.005, 0.02)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, risk)))
    lines = ["Date,Open,High,Low,Close,Volume"]
    for session, price in zip(sessions, prices, strict=True):
        lines.append(
            f"{session.isoformat()},{price:.8f},{price * 1.01:.8f},"
            f"{price * 0.99:.8f},{price:.8f},1000"
        )
    source = SourceResponse(
        ("\n".join(lines) + "\n").encode(),
        "fixture:mri005",
        "stooq",
        "SPY",
        date(2024, 1, 2),
        sessions[-1],
        "adjusted_ohlc",
        False,
    )
    paths = persist(source, tmp_path)
    features = build_features(reload_snapshot(paths.manifest))
    model = fit_gaussian_hmm(features.iloc[:180], HMMConfig(n_starts=2, max_iter=50))
    filtered = filter_new_sessions(model, features.iloc[180:])
    assert model.train_end == sessions[179]
    assert filtered["date"].tolist() == sessions[180:]
    assert np.allclose(filtered.iloc[:, 1:].sum(axis=1), 1)
