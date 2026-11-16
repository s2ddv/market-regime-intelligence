"""Provider boundary for immutable daily source responses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Protocol
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen


class ProviderError(RuntimeError):
    """A source could not provide daily observations."""


@dataclass(frozen=True)
class SourceResponse:
    content: bytes
    url: str
    provider: str
    symbol: str
    start: date
    end: date
    price_basis: str
    adjustment_verified: bool
    retrieved_at: datetime | None = None


class DailyProvider(Protocol):
    """Fetch source bytes without interpreting downstream research data."""

    def fetch(self, symbol: str, start: date, end: date) -> SourceResponse: ...


class StooqProvider:
    """Stooq's CSV endpoint; adjusted-OHLC status remains unverified."""

    def fetch(self, symbol: str, start: date, end: date) -> SourceResponse:
        if symbol != "SPY" or end < start:
            raise ValueError("Stooq adapter accepts SPY and an ordered date range")
        query = urlencode(
            {"s": "spy.us", "i": "d", "d1": start.strftime("%Y%m%d"), "d2": end.strftime("%Y%m%d")}
        )
        url = f"https://stooq.com/q/d/l/?{query}"
        try:
            with urlopen(url, timeout=30) as handle:  # noqa: S310 - fixed HTTPS host
                content = handle.read()
        except (OSError, URLError) as exc:
            raise ProviderError(f"Stooq request failed: {exc}") from exc
        if not content or content.lstrip().startswith(b"No data"):
            raise ProviderError("Stooq returned no CSV data")
        return SourceResponse(
            content, url, "stooq", symbol, start, end, "adjusted_ohlc", False, datetime.now(UTC)
        )
