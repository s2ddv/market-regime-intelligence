# Market Regime Intelligence

Research foundation for testing whether causally filtered market-regime probabilities improve SPY risk forecasts. Current scope: MRI-001 research protocol, MRI-002 Python foundation, MRI-003 historical data pipeline, and MRI-004 deterministic session features and forward targets. No forecasting model or empirical performance result exists yet.

See [protocol](docs/research_protocol.md), [architecture](docs/architecture.md), [data contract](docs/data_contract.md), [feature contract](docs/mri004_feature_contract.md), and [source-readiness decision](docs/source_readiness.md).

## Setup

Requires Python 3.12. Install `uv`, then run `uv sync --locked --all-groups` and `uv run --locked pytest`. Run `uv run --locked ruff check .`, `uv run --locked ruff format --check .`, and `uv run --locked mypy src` for all checks. `uv.lock` is committed.

## Historical SPY data

`uv run --locked python -m mri.data.cli fetch --start 2005-01-01` fetches Stooq CSV through the last completed XNYS session, validates it, and writes immutable raw and processed snapshots under `data/`. An explicit `--end YYYY-MM-DD` may narrow the range. This requires provider access and carries Stooq's unverified adjusted-OHLC assumption. `uv run --locked python -m mri.data.cli reload --manifest data/processed/<id>.json` verifies and reloads offline. The live Stooq endpoint returned HTTP 404 during initial validation on 2026-11-16; no real SPY snapshot was obtained. To demonstrate the full ingestion path offline, run `uv run --locked python -m mri.data.cli ingest-fixture --path tests/fixtures/stooq_spy_2024-01-02_2024-01-04.csv --start 2024-01-02 --end 2024-01-04`. This writes an explicitly labeled synthetic fixture snapshot; then use `reload` with its manifest path. Tests use deterministic synthetic responses and never access the internet. The protocol prevents interpreting this source as point-in-time until independently audited.

## Session features and targets

`mri.features.build_features` computes log return, trailing volatility, momentum, and drawdown from a validated canonical SPY table. `mri.features.build_forward_targets` creates a separate table of 5- and 20-session forward variance, volatility, and label maturity dates. Load the frozen window settings with `mri.features.load_feature_spec(Path("configs/features.yaml"))`. These functions have been checked on synthetic paths only; the repository does not yet contain an authorized real SPY snapshot.
