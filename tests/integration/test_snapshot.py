from datetime import UTC, date, datetime

import pytest

from mri.data.provider import SourceResponse
from mri.data.schema import DataValidationError
from mri.data.storage import persist, reload_snapshot

SOURCE = SourceResponse(
    b"Date,Open,High,Low,Close,Volume\n2024-01-02,100,102,99,101,1000\n2024-01-03,101,103,100,102,1100\n2024-01-04,102,104,101,103,1200\n",
    "https://stooq.com/fixture",
    "stooq",
    "SPY",
    date(2024, 1, 2),
    date(2024, 1, 4),
    "adjusted_ohlc",
    False,
)


def test_persist_reload_offline_and_no_overwrite(tmp_path):
    paths = persist(SOURCE, tmp_path, datetime(2024, 1, 5, tzinfo=UTC))
    assert paths.raw.read_bytes() == SOURCE.content
    frame = reload_snapshot(paths.manifest)
    assert len(frame) == 3
    assert frame["adjusted_close"].tolist() == [101, 102, 103]
    with pytest.raises(FileExistsError):
        persist(SOURCE, tmp_path)


def test_corrupt_snapshot_rejected(tmp_path):
    paths = persist(SOURCE, tmp_path)
    paths.raw.write_bytes(b"tampered")
    with pytest.raises(DataValidationError, match="hash mismatch"):
        reload_snapshot(paths.manifest)
