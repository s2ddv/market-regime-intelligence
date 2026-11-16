"""MRI-003 enforces source session bounds; future feature tests arrive in MRI-004."""

from datetime import date

import pytest

from mri.data.provider import SourceResponse
from mri.data.schema import DataValidationError, parse_stooq, validate_daily


def test_reject_source_row_outside_requested_sessions():
    source = SourceResponse(
        b"Date,Open,High,Low,Close,Volume\n2024-01-02,100,102,99,101,1000\n2024-01-03,101,103,100,102,1100\n2024-01-04,102,104,101,103,1200\n",
        "fixture",
        "stooq",
        "SPY",
        date(2024, 1, 2),
        date(2024, 1, 3),
        "adjusted_ohlc",
        False,
    )
    with pytest.raises(DataValidationError, match="unexpected"):
        validate_daily(parse_stooq(source), source)
