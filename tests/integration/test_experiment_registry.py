"""Frozen synthetic run identity and attempt provenance across a real fixture path."""

from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pytest

from mri.data.provider import SourceResponse
from mri.data.storage import persist, reload_snapshot
from mri.evaluation import WalkForwardConfig, build_walk_forward_plan
from mri.experiments import ExperimentError, freeze_run, record_attempt
from mri.features import build_features, build_forward_targets


def prepared_inputs(tmp_path):
    dates = [
        stamp.date()
        for stamp in xcals.get_calendar("XNYS", start="2024-01-02", end="2024-06-28").sessions
    ]
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(707).normal(0, 0.01, len(dates))))
    lines = ["Date,Open,High,Low,Close,Volume"]
    for session, price in zip(dates, prices, strict=True):
        lines.append(
            f"{session.isoformat()},{price:.8f},{price * 1.01:.8f},"
            f"{price * 0.99:.8f},{price:.8f},1000"
        )
    response = SourceResponse(
        ("\n".join(lines) + "\n").encode(),
        "fixture:mri007",
        "stooq",
        "SPY",
        dates[0],
        dates[-1],
        "adjusted_ohlc",
        False,
    )
    paths = persist(response, tmp_path)
    features = build_features(reload_snapshot(paths.manifest))
    targets = build_forward_targets(features, horizons=(5,))
    plan = build_walk_forward_plan(
        features,
        targets[["date", "label_end_session_5"]],
        WalkForwardConfig(
            development_end=date(2024, 3, 28),
            validation_end=date(2024, 5, 31),
            horizons=(5,),
            min_training_sessions=30,
        ),
    )
    return paths.manifest, plan


def synthetic_config(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "schema_version: 1\nname: mri007-test\nmode: synthetic\nphase: validation\n"
        "seed: 20261118\ndevelopment_end: '2024-03-28'\n"
        "validation_end: '2024-05-31'\nhorizons: [5]\n"
        "min_training_sessions: 30\nevent_quantile: 0.90\n"
        "retrain: monthly\nmodel_candidates: [historical-20, hmm-2]\n"
    )
    return path


def test_run_and_attempt_are_exclusive_and_bound_to_verified_plan(tmp_path):
    manifest, plan = prepared_inputs(tmp_path)
    repo = Path(__file__).resolve().parents[2]
    config = synthetic_config(tmp_path)
    run = freeze_run(config, manifest, plan, repo / "uv.lock", repo, tmp_path / "runs")
    assert (run / "run.json").exists()
    attempt = record_attempt(run, "hmm-jan", plan, plan.folds[0], "hmm-2", 20261118)
    assert attempt.exists()
    assert '"status": "planned"' in attempt.read_text()
    with pytest.raises(FileExistsError):
        record_attempt(run, "hmm-jan", plan, plan.folds[0], "hmm-2", 20261118)
    with pytest.raises(FileExistsError):
        freeze_run(config, manifest, plan, repo / "uv.lock", repo, tmp_path / "runs")
    with pytest.raises(ExperimentError, match="Model or seed"):
        record_attempt(run, "unfrozen", plan, plan.folds[0], "garch-11", 20261118)
    final = next(fold for fold in plan.folds if fold.phase == "final")
    with pytest.raises(ExperimentError, match="outside the frozen phase"):
        record_attempt(run, "final", plan, final, "hmm-2", 20261118)


def test_fixture_and_plan_identity_fail_closed(tmp_path):
    manifest, plan = prepared_inputs(tmp_path)
    repo = Path(__file__).resolve().parents[2]
    config = synthetic_config(tmp_path)
    run = freeze_run(config, manifest, plan, repo / "uv.lock", repo, tmp_path / "runs")
    other = prepared_inputs(tmp_path / "other")[1]
    # Same sessions, but a different fold boundary must be rejected.
    from dataclasses import replace

    changed = replace(other, config=replace(other.config, development_end=date(2024, 3, 27)))
    with pytest.raises(ExperimentError, match="plan differs"):
        record_attempt(run, "changed-plan", changed, changed.folds[0], "hmm-2", 20261118)
    broken = manifest.read_text().replace("fixture:mri007", "https://example.invalid/spy")
    bad_manifest = manifest.with_name("outside.json")
    bad_manifest.write_text(broken)
    with pytest.raises(ExperimentError, match="fixture snapshots only"):
        freeze_run(config, bad_manifest, plan, repo / "uv.lock", repo, tmp_path / "other-runs")
