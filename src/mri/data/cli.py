"""Explicit online fetch, fixture ingestion, and offline snapshot reload."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import exchange_calendars as xcals
import yaml

from mri.data.provider import SourceResponse, StooqProvider
from mri.data.storage import persist, reload_snapshot


def latest_completed_session(now: datetime | None = None) -> date:
    """Return the most recent XNYS session whose market close has passed."""
    instant = now or datetime.now(UTC)
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    calendar = xcals.get_calendar("XNYS")
    session = calendar.date_to_session(instant.date(), direction="previous")
    if calendar.session_close(session).to_pydatetime() > instant:
        session = calendar.previous_session(session)
    return cast(date, session.date())


def load_data_config(path: Path) -> dict[str, Any]:
    """Load the explicit source selection and research coverage."""
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Data config must be a mapping")
    if config.get("symbol") != "SPY" or config.get("provider") != "stooq":
        raise ValueError("Only the Stooq SPY adapter is currently implemented")
    if config.get("calendar") != "XNYS" or config.get("price_basis") != "adjusted_ohlc":
        raise ValueError("Unsupported calendar or price basis")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m mri.data.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    fetch.add_argument("--start", type=date.fromisoformat)
    fetch.add_argument("--end", type=date.fromisoformat)
    fetch.add_argument("--root", type=Path, default=Path.cwd())
    fixture = sub.add_parser("ingest-fixture")
    fixture.add_argument("--path", type=Path, required=True)
    fixture.add_argument("--start", type=date.fromisoformat, required=True)
    fixture.add_argument("--end", type=date.fromisoformat, required=True)
    fixture.add_argument("--root", type=Path, default=Path.cwd())
    reload_cmd = sub.add_parser("reload")
    reload_cmd.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        config = load_data_config(args.config)
        start = args.start or date.fromisoformat(config["start"])
        end = args.end or latest_completed_session()
        if end > latest_completed_session():
            parser.error("--end must be a completed XNYS session")
        response = StooqProvider().fetch(config["symbol"], start, end)
        paths = persist(response, args.root)
        print(f"validated manifest: {paths.manifest}")
    elif args.command == "ingest-fixture":
        response = SourceResponse(
            args.path.read_bytes(),
            f"fixture:{args.path.name}",
            "stooq",
            "SPY",
            args.start,
            args.end,
            "adjusted_ohlc",
            False,
        )
        paths = persist(response, args.root)
        print(f"validated fixture manifest: {paths.manifest}")
    else:
        frame = reload_snapshot(args.manifest)
        print(f"validated {len(frame)} rows: {frame['date'].iloc[0]} to {frame['date'].iloc[-1]}")


if __name__ == "__main__":
    main()
