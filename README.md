# Market Regime Intelligence

Research foundation for testing whether causally filtered market-regime probabilities improve SPY risk forecasts. Current scope: MRI-001 research protocol, MRI-002 Python foundation, MRI-003 historical data pipeline, MRI-004 deterministic session features and forward targets, MRI-005 synthetic Gaussian HMM fitting and causal filtering, MRI-006 walk-forward planning and target access gates, and MRI-007 synthetic experiment provenance. No empirical performance result exists yet.

See [protocol](docs/research_protocol.md), [architecture](docs/architecture.md), [data contract](docs/data_contract.md), [feature contract](docs/mri004_feature_contract.md), [HMM contract](docs/mri005_hmm_contract.md), [walk-forward contract](docs/mri006_walk_forward_contract.md), [experiment record contract](docs/mri007_experiment_contract.md), and [source-readiness decision](docs/source_readiness.md).

## Setup

Requires Python 3.12. Install `uv`, then run `uv sync --locked --all-groups` and `uv run --locked pytest`. Run `uv run --locked ruff check .`, `uv run --locked ruff format --check .`, and `uv run --locked mypy src` for all checks. `uv.lock` is committed.

## Historical SPY data

`uv run --locked python -m mri.data.cli fetch --start 2005-01-01` fetches Stooq CSV through the last completed XNYS session, validates it, and writes immutable raw and processed snapshots under `data/`. An explicit `--end YYYY-MM-DD` may narrow the range. This requires provider access and carries Stooq's unverified adjusted-OHLC assumption. `uv run --locked python -m mri.data.cli reload --manifest data/processed/<id>.json` verifies and reloads offline. The live Stooq endpoint returned HTTP 404 during initial validation on 2026-11-16; no real SPY snapshot was obtained. To demonstrate the full ingestion path offline, run `uv run --locked python -m mri.data.cli ingest-fixture --path tests/fixtures/stooq_spy_2024-01-02_2024-01-04.csv --start 2024-01-02 --end 2024-01-04`. This writes an explicitly labeled synthetic fixture snapshot; then use `reload` with its manifest path. Tests use deterministic synthetic responses and never access the internet. The protocol prevents interpreting this source as point-in-time until independently audited.

## Session features and targets

`mri.features.build_features` computes log return, trailing volatility, momentum, and drawdown from a validated canonical SPY table. `mri.features.build_forward_targets` creates a separate table of 5- and 20-session forward variance, volatility, and label maturity dates. Load the frozen window settings with `mri.features.load_feature_spec(Path("configs/features.yaml"))`. These functions have been checked on synthetic paths only; the repository does not yet contain an authorized real SPY snapshot.

## Gaussian HMM

`mri.models.fit_gaussian_hmm(training_features, HMMConfig(...))` fits only the supplied, chronologically complete training prefix using log return and 20-session realized volatility. It records the last training session and frozen parameters. `mri.models.filter_new_sessions(model, later_features)` produces forward-filtered state probabilities for consecutive XNYS sessions strictly after that cutoff. This is a synthetic software demonstration; a real source audit remains necessary before market claims.

## Walk-forward planning

`mri.evaluation.build_walk_forward_plan(features, maturity_only, config)` creates monthly expanding-prefix blocks for validation and final periods. Its input contains only feature dates and target maturity dates; forward values are accessed separately through `training_target_rows` and `training_event_threshold`, which expose fully matured labels from earlier phases. The plan supplies exact fit and prediction indices for HMM calls. It does not train forecasting mappings, score predictions, open final outcomes, or validate historical data publication timing. Tests exercise the boundary rules and one synthetic HMM refit.

## Synthetic experiment records

`mri.experiments.freeze_run` verifies a fixture snapshot and binds its manifest, exact fold plan, configuration, package lock and code state to an exclusive run record. `record_attempt` records a planned variant under that run before execution. The example configuration is in `configs/experiments/mri007_synthetic.yaml`. This infrastructure does not run or score models; the full research preregistration and real-source audit remain pending.
