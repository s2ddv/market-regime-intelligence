"""Synthetic source bytes through validated snapshot and MRI-004 transforms."""

from datetime import date
from pathlib import Path

import exchange_calendars as xcals

from mri.data.provider import SourceResponse
from mri.data.storage import persist, reload_snapshot
from mri.features import build_features, build_forward_targets, load_feature_spec


def test_snapshot_to_features_and_matured_targets(tmp_path):
    calendar = xcals.get_calendar("XNYS", start="2024-01-02", end="2024-05-31")
    sessions = [stamp.date() for stamp in calendar.sessions[:80]]
    rows = ["Date,Open,High,Low,Close,Volume"]
    for index, session in enumerate(sessions):
        close = 100 + index / 10
        rows.append(f"{session.isoformat()},{close},{close + 1},{close - 1},{close},1000")
    source = SourceResponse(
        ("\n".join(rows) + "\n").encode(),
        "fixture:mri004",
        "stooq",
        "SPY",
        date(2024, 1, 2),
        sessions[-1],
        "adjusted_ohlc",
        False,
    )
    paths = persist(source, tmp_path)
    frame = reload_snapshot(paths.manifest)
    features = build_features(frame, load_feature_spec(Path("configs/features.yaml")))
    targets = build_forward_targets(features)
    assert len(frame) == len(features) == len(targets) == 80
    assert features["realized_volatility_20"].first_valid_index() == 20
    assert targets["label_end_session_20"].iloc[0] == sessions[20]
