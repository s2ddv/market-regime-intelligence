"""Immutable identity and attempt records for synthetic research preparation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from mri.data.storage import reload_snapshot
from mri.evaluation import WalkForwardFold, WalkForwardPlan


class ExperimentError(ValueError):
    """A run or attempt violates the frozen synthetic-only contract."""


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ExperimentError(f"Cannot inspect Git {args[0]} state")
    return result.stdout


def _config(raw: bytes) -> dict[str, Any]:
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ExperimentError("Invalid experiment YAML") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "name",
        "mode",
        "phase",
        "seed",
        "model_candidates",
        "development_end",
        "validation_end",
        "horizons",
        "min_training_sessions",
        "event_quantile",
        "retrain",
    }:
        raise ExperimentError("Unsupported experiment configuration fields")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(value["name"]) is not str
        or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", value["name"]) is None
        or value["mode"] != "synthetic"
        or value["phase"] != "validation"
        or type(value["seed"]) is not int
        or value["seed"] < 0
        or not isinstance(value["model_candidates"], list)
        or not value["model_candidates"]
        or any(
            type(name) is not str or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name) is None
            for name in value["model_candidates"]
        )
        or len(set(value["model_candidates"])) != len(value["model_candidates"])
        or type(value["development_end"]) is not str
        or type(value["validation_end"]) is not str
        or not isinstance(value["horizons"], list)
        or not value["horizons"]
        or any(type(horizon) is not int or horizon <= 0 for horizon in value["horizons"])
        or len(set(value["horizons"])) != len(value["horizons"])
        or type(value["min_training_sessions"]) is not int
        or value["min_training_sessions"] < 1
        or type(value["event_quantile"]) not in (int, float)
        or not 0 < value["event_quantile"] < 1
        or value["retrain"] != "monthly"
    ):
        raise ExperimentError("Only named synthetic validation preparations are supported")
    try:
        development = date.fromisoformat(value["development_end"])
        validation = date.fromisoformat(value["validation_end"])
    except ValueError as exc:
        raise ExperimentError("Phase boundaries must be ISO dates") from exc
    if development >= validation:
        raise ExperimentError("Development must end before validation")
    return value


def _plan_sha(plan: WalkForwardPlan) -> str:
    return _sha(
        _json_bytes(
            {
                "dates": [value.isoformat() for value in plan.dates],
                "config": {
                    **asdict(plan.config),
                    "development_end": plan.config.development_end.isoformat(),
                    "validation_end": plan.config.validation_end.isoformat(),
                },
                "folds": [asdict(fold) for fold in plan.folds],
            }
        )
    )


def freeze_run(
    config_path: Path,
    source_manifest_path: Path,
    plan: WalkForwardPlan,
    lock_path: Path,
    repo_path: Path,
    output_root: Path,
) -> Path:
    """Verify fixture bytes and create one exclusive, content-addressed run record."""
    config_bytes = config_path.read_bytes()
    config = _config(config_bytes)
    if (
        config["development_end"] != plan.config.development_end.isoformat()
        or config["validation_end"] != plan.config.validation_end.isoformat()
        or config["horizons"] != list(plan.config.horizons)
        or config["min_training_sessions"] != plan.config.min_training_sessions
        or config["event_quantile"] != plan.config.event_quantile
    ):
        raise ExperimentError("Configuration differs from the frozen fold plan")
    manifest_bytes = source_manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ExperimentError("Invalid source manifest") from exc
    if not isinstance(manifest, dict) or not str(manifest.get("source_url", "")).startswith(
        "fixture:"
    ):
        raise ExperimentError("MRI-007 accepts synthetic fixture snapshots only")
    frame = reload_snapshot(source_manifest_path)
    if frame["date"].tolist() != list(plan.dates):
        raise ExperimentError("Planned sessions differ from the verified source")
    lock_bytes = lock_path.read_bytes()
    head = _git(repo_path, "rev-parse", "HEAD").strip()
    status = _git(repo_path, "status", "--porcelain=v1").strip()
    dirty = bool(status)
    hashes = {
        "config_sha256": _sha(config_bytes),
        "source_manifest_sha256": _sha(manifest_bytes),
        "lock_sha256": _sha(lock_bytes),
        "raw_sha256": manifest["raw_sha256"],
        "processed_sha256": manifest["processed_sha256"],
        "plan_sha256": _plan_sha(plan),
        "tracked_diff_sha256": _sha(_git(repo_path, "diff", "--binary", "HEAD").encode()),
    }
    identity = _sha(_json_bytes({"hashes": hashes, "head": head, "dirty": dirty}))[:20]
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "scipy", "exchange-calendars", "scikit-learn")
    }
    record = {
        "schema_version": 1,
        "run_id": identity,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "source_key": manifest["key"],
        "source_provider": manifest["provider"],
        "source_adjustment_verified": manifest["adjustment_verified"],
        "hashes": hashes,
        "git_head": head,
        "git_dirty": dirty,
        "git_status": status.splitlines(),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "status": "prepared_only",
    }
    path = output_root / identity
    path.mkdir(parents=True, exist_ok=False)
    with (path / "run.json").open("xb") as handle:
        handle.write(_json_bytes(record))
    return path


def record_attempt(
    run_path: Path,
    attempt_id: str,
    plan: WalkForwardPlan,
    fold: WalkForwardFold,
    model: str,
    seed: int,
) -> Path:
    """Record a planned variant before execution; never edit an existing attempt."""
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", attempt_id) is None:
        raise ExperimentError("Invalid attempt ID")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", model) is None:
        raise ExperimentError("Invalid model name")
    if type(seed) is not int or seed < 0:
        raise ExperimentError("Invalid seed")
    run = json.loads((run_path / "run.json").read_text())
    if run.get("run_id") != run_path.name or run.get("status") != "prepared_only":
        raise ExperimentError("Invalid run identity")
    if fold not in plan.folds or fold.phase != run["config"]["phase"]:
        raise ExperimentError("Attempt fold is outside the frozen phase")
    if _plan_sha(plan) != run["hashes"]["plan_sha256"]:
        raise ExperimentError("Attempt plan differs from the frozen run")
    if fold.horizon not in plan.config.horizons or fold.phase != "validation":
        raise ExperimentError("Unsupported attempt horizon or phase")
    if model not in run["config"]["model_candidates"] or seed != run["config"]["seed"]:
        raise ExperimentError("Model or seed differs from the frozen configuration")
    record = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "attempt_id": attempt_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase": fold.phase,
        "horizon": fold.horizon,
        "fit_end_index": fold.fit_end_index,
        "prediction_start": fold.prediction_start,
        "prediction_stop": fold.prediction_stop,
        "fit_end_date": plan.dates[fold.fit_end_index].isoformat(),
        "model": model,
        "seed": seed,
        "status": "planned",
    }
    directory = run_path / "attempts"
    directory.mkdir(exist_ok=True)
    path = directory / f"{attempt_id}.json"
    with path.open("xb") as handle:
        handle.write(_json_bytes(record))
    return path
