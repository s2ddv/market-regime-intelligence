"""Canonical parsing and exchange-session validation."""

from __future__ import annotations

from datetime import date
from io import BytesIO

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from mri.data.provider import SourceResponse

COLUMNS = ["date", "open", "high", "low", "close", "adjusted_close", "volume", "symbol", "source"]
CSV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


class DataValidationError(ValueError):
    """Source observations violate the daily contract."""


def parse_stooq(response: SourceResponse) -> pd.DataFrame:
    """Parse exact Stooq CSV schema without sorting or repairing observations."""
    if response.provider != "stooq" or response.price_basis != "adjusted_ohlc":
        raise DataValidationError("Unsupported provider or price basis")
    try:
        raw = pd.read_csv(BytesIO(response.content))
    except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
        raise DataValidationError("Invalid CSV") from exc
    if list(raw.columns) != CSV_COLUMNS:
        raise DataValidationError(f"Expected CSV columns {CSV_COLUMNS}")
    try:
        parsed_dates = pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="raise")
        frame = pd.DataFrame(
            {
                "date": parsed_dates.dt.date,
                "open": pd.to_numeric(raw["Open"], errors="raise"),
                "high": pd.to_numeric(raw["High"], errors="raise"),
                "low": pd.to_numeric(raw["Low"], errors="raise"),
                "close": pd.to_numeric(raw["Close"], errors="raise"),
                "adjusted_close": pd.to_numeric(raw["Close"], errors="raise"),
                "volume": pd.to_numeric(raw["Volume"], errors="raise"),
                "symbol": response.symbol,
                "source": response.provider,
            }
        )
    except (ValueError, TypeError, AttributeError) as exc:
        raise DataValidationError("Invalid date or numeric value") from exc
    return frame[COLUMNS]


def validate_daily(frame: pd.DataFrame, response: SourceResponse) -> None:
    """Reject incomplete, unordered, mixed-basis, or malformed daily data."""
    if list(frame.columns) != COLUMNS or frame.empty:
        raise DataValidationError("Invalid or empty canonical schema")
    if response.start > response.end or response.symbol != "SPY":
        raise DataValidationError("Invalid request metadata")
    if response.price_basis not in {"adjusted_ohlc", "raw_ohlc_adjusted_close"}:
        raise DataValidationError("Unknown adjustment convention")
    if frame["date"].isna().any() or not frame["date"].map(lambda x: isinstance(x, date)).all():
        raise DataValidationError("Invalid dates")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise DataValidationError("Duplicate or non-monotonic dates")
    if frame["symbol"].ne(response.symbol).any() or frame["source"].ne(response.provider).any():
        raise DataValidationError("Mixed symbol or source")
    numeric = frame[["open", "high", "low", "close", "adjusted_close", "volume"]]
    try:
        values = numeric.to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise DataValidationError("Non-numeric market values") from exc
    if not np.isfinite(values).all():
        raise DataValidationError("Non-finite market values")
    if (values[:, :5] <= 0).any() or (values[:, 5] < 0).any():
        raise DataValidationError("Invalid price or volume")
    if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any() or (
        frame["low"] > frame[["open", "high", "close"]].min(axis=1)
    ).any():
        raise DataValidationError("Invalid OHLC relationship")
    if response.price_basis == "adjusted_ohlc" and not np.array_equal(values[:, 4], values[:, 3]):
        raise DataValidationError("Adjusted OHLC requires adjusted_close = close")
    calendar = xcals.get_calendar("XNYS")
    try:
        expected = {
            stamp.date() for stamp in calendar.sessions_in_range(response.start, response.end)
        }
    except (ValueError, TypeError) as exc:
        raise DataValidationError("Date range outside exchange calendar") from exc
    observed = set(frame["date"])
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise DataValidationError(
            f"Exchange sessions mismatch: missing={missing}, unexpected={unexpected}"
        )
