"""Content-addressed, exclusive dataset snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from mri.data.provider import SourceResponse
from mri.data.schema import COLUMNS, DataValidationError, parse_stooq, validate_daily

SCHEMA_VERSION = 1


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


@dataclass(frozen=True)
class SnapshotPaths:
    raw: Path
    parquet: Path
    manifest: Path


def persist(
    response: SourceResponse, root: Path, retrieved_at: datetime | None = None
) -> SnapshotPaths:
    """Validate before writing; reject collisions and preserve source bytes."""
    frame = parse_stooq(response)
    validate_daily(frame, response)
    timestamp = retrieved_at or response.retrieved_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("retrieved_at must have a timezone")
    raw_hash = digest(response.content)
    key = digest(
        json.dumps(
            [
                response.provider,
                response.symbol,
                response.start.isoformat(),
                response.end.isoformat(),
                raw_hash,
            ],
            separators=(",", ":"),
        ).encode()
    )[:20]
    raw_path = root / "data" / "raw" / f"{key}.csv"
    parquet_path = root / "data" / "processed" / f"{key}.parquet"
    manifest_path = root / "data" / "processed" / f"{key}.json"
    if any(path.exists() for path in (raw_path, parquet_path, manifest_path)):
        raise FileExistsError(f"Snapshot {key} already exists; reload it instead")
    buf = BytesIO()
    frame.to_parquet(buf, index=False)
    parquet_bytes = buf.getvalue()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "key": key,
        "provider": response.provider,
        "symbol": response.symbol,
        "start": response.start.isoformat(),
        "end": response.end.isoformat(),
        "source_url": response.url,
        "price_basis": response.price_basis,
        "adjustment_verified": response.adjustment_verified,
        "calendar": "XNYS",
        "retrieved_at": timestamp.astimezone(UTC).isoformat(),
        "ingested_at": datetime.now(UTC).isoformat(),
        "raw_sha256": raw_hash,
        "processed_sha256": digest(parquet_bytes),
        "rows": len(frame),
        "first_date": frame["date"].iloc[0].isoformat(),
        "last_date": frame["date"].iloc[-1].isoformat(),
        "validation": "passed",
        "processing_version": "mri-0.1.0",
    }
    _write_exclusive(raw_path, response.content)
    _write_exclusive(parquet_path, parquet_bytes)
    _write_exclusive(
        manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    )
    return SnapshotPaths(raw_path, parquet_path, manifest_path)


def reload_snapshot(manifest_path: Path) -> pd.DataFrame:
    """Reload offline and verify source and canonical hashes and validation."""
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("validation") != "passed":
        raise DataValidationError("Unknown manifest schema or validation status")
    key = manifest["key"]
    if not isinstance(key, str) or len(key) != 20 or any(c not in "0123456789abcdef" for c in key):
        raise DataValidationError("Invalid snapshot key")
    raw_path = manifest_path.parent.parent / "raw" / f"{key}.csv"
    parquet_path = manifest_path.parent / f"{key}.parquet"
    raw_bytes = raw_path.read_bytes()
    parquet_bytes = parquet_path.read_bytes()
    if (
        digest(raw_bytes) != manifest["raw_sha256"]
        or digest(parquet_bytes) != manifest["processed_sha256"]
    ):
        raise DataValidationError("Snapshot hash mismatch")
    response = SourceResponse(
        raw_bytes,
        manifest["source_url"],
        manifest["provider"],
        manifest["symbol"],
        datetime.fromisoformat(manifest["start"]).date(),
        datetime.fromisoformat(manifest["end"]).date(),
        manifest["price_basis"],
        manifest["adjustment_verified"],
        datetime.fromisoformat(manifest["retrieved_at"]),
    )
    frame = pd.read_parquet(BytesIO(parquet_bytes))
    if list(frame.columns) != COLUMNS or not frame.equals(parse_stooq(response)):
        raise DataValidationError("Processed data differs from raw source")
    validate_daily(frame, response)
    if (
        len(frame) != manifest["rows"]
        or frame["date"].iloc[0].isoformat() != manifest["first_date"]
        or frame["date"].iloc[-1].isoformat() != manifest["last_date"]
    ):
        raise DataValidationError("Manifest coverage mismatch")
    return frame
