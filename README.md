# Market Regime Intelligence

Research foundation for testing whether causally filtered market-regime probabilities improve SPY risk forecasts. Current scope: MRI-001 research protocol, MRI-002 Python foundation, MRI-003 historical data pipeline. No forecasting model or empirical performance result exists yet.

See [protocol](docs/research_protocol.md), [architecture](docs/architecture.md), and [data contract](docs/data_contract.md).

## Setup

Requires Python 3.12. Install `uv`, then run `uv sync --locked --all-groups` and `uv run --locked pytest`. Run `uv run --locked ruff check .`, `uv run --locked ruff format --check .`, and `uv run --locked mypy src` for all checks. `uv.lock` is committed.

## Historical SPY data

`uv run --locked python -m mri.data.cli fetch --start 2005-01-01` fetches Stooq CSV through the last completed XNYS session, validates it, and writes immutable raw and processed snapshots under `data/`. An explicit `--end YYYY-MM-DD` may narrow the range. This requires provider access and carries Stooq's unverified adjusted-OHLC assumption. `uv run --locked python -m mri.data.cli reload --manifest data/processed/<id>.json` verifies and reloads offline. Tests use deterministic synthetic responses and never access the internet. The protocol prevents interpreting this source as point-in-time until independently audited.
