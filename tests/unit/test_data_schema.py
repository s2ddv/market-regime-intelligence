from datetime import date

import pytest

from mri.data.provider import ProviderError, SourceResponse, StooqProvider
from mri.data.schema import DataValidationError, parse_stooq, validate_daily

BASE = (
    b"Date,Open,High,Low,Close,Volume\n"
    b"2024-01-02,100,102,99,101,1000\n"
    b"2024-01-03,101,103,100,102,1100\n"
    b"2024-01-04,102,104,101,103,1200\n"
)


def response(content=BASE):
    return SourceResponse(
        content,
        "https://stooq.com/fixture",
        "stooq",
        "SPY",
        date(2024, 1, 2),
        date(2024, 1, 4),
        "adjusted_ohlc",
        False,
    )


def test_valid_daily():
    source = response()
    validate_daily(parse_stooq(source), source)


@pytest.mark.parametrize(
    "content",
    [
        BASE + b"2024-01-04,102,104,101,103,1200\n",
        BASE.replace(b"2024-01-03,101,103,100,102,1100\n", b""),
        BASE.replace(b"100,102,99,101,1000", b"0,102,99,101,1000"),
        BASE.replace(b"100,102,99,101,1000", b"100,102,99,101,-1"),
        BASE.replace(b"100,102,99,101,1000", b"100,98,99,101,1000"),
        BASE.replace(b"100,102,99,101,1000", b"100,inf,99,101,1000"),
        BASE.replace(b"Date,Open,High,Low,Close,Volume", b"Date,Open,Close,Volume"),
    ],
)
def test_reject_bad_responses(content):
    source = response(content)
    with pytest.raises(DataValidationError):
        validate_daily(parse_stooq(source), source)


def test_reject_unordered_dates():
    lines = BASE.splitlines(keepends=True)
    source = response(lines[0] + lines[2] + lines[1] + lines[3])
    with pytest.raises(DataValidationError, match="non-monotonic"):
        validate_daily(parse_stooq(source), source)


def test_reject_adjustment_inconsistency():
    source = response()
    frame = parse_stooq(source)
    frame.loc[0, "adjusted_close"] = 99.0
    with pytest.raises(DataValidationError, match="adjusted_close"):
        validate_daily(frame, source)


def test_provider_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("mri.data.provider.urlopen", fail)
    with pytest.raises(ProviderError, match="offline"):
        StooqProvider().fetch("SPY", date(2024, 1, 2), date(2024, 1, 4))


def test_latest_completed_session_uses_exchange_close():
    from datetime import UTC, datetime

    from mri.data.cli import latest_completed_session

    assert latest_completed_session(datetime(2024, 1, 3, 20, tzinfo=UTC)) == date(2024, 1, 2)
    assert latest_completed_session(datetime(2024, 1, 3, 22, tzinfo=UTC)) == date(2024, 1, 3)
