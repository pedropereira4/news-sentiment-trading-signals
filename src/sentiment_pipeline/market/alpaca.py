"""Alpaca market data: historical 1-minute bars for many symbols at once.

GET /v2/stocks/bars?symbols=A,B,...  returns {"bars": {"A": [...], ...}, "next_page_token"}.
`limit` caps the total across symbols, so a window is read by following next_page_token.

Feeds: "sip" (all US exchanges, real volume) is what the event study wants; Alpaca's free
plan serves it with a ~15-minute delay, so requests end 16 minutes in the past. If the
account cannot use SIP at all (HTTP 403), the client switches to "iex" (one exchange,
free, real-time) and says so, instead of failing every cycle.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bar:
    ticker: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: int | None
    vwap: float | None
    feed: str


class AlpacaAuthError(RuntimeError):
    """Invalid keys: retrying cannot help."""


class AlpacaFeedNotPermitted(RuntimeError):
    """The account's market-data plan does not include the requested feed/window."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return isinstance(exc, httpx.TransportError)


def parse_bars(payload: dict[str, Any], feed: str) -> list[Bar]:
    out: list[Bar] = []
    for ticker, rows in (payload.get("bars") or {}).items():
        for r in rows or []:
            out.append(
                Bar(
                    ticker=ticker,
                    ts=datetime.fromisoformat(r["t"].replace("Z", "+00:00")),
                    open=float(r["o"]),
                    high=float(r["h"]),
                    low=float(r["l"]),
                    close=float(r["c"]),
                    volume=int(r["v"]),
                    trade_count=int(r["n"]) if r.get("n") is not None else None,
                    vwap=float(r["vw"]) if r.get("vw") is not None else None,
                    feed=feed,
                )
            )
    return out


class AlpacaBarsClient:
    DATA_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        feed: str = "sip",
        timeout_s: float = 30.0,
        max_retries: int = 5,
        backoff_initial_s: float = 2.0,
        page_limit: int = 10_000,
        http: httpx.Client | None = None,
    ) -> None:
        self.http = http or httpx.Client(timeout=timeout_s)
        self.http.headers.update({"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key})
        self.feed = feed
        self.page_limit = page_limit
        self._retrying = retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential_jitter(
                initial=backoff_initial_s, max=60, jitter=backoff_initial_s
            ),
            reraise=True,
            before_sleep=lambda rs: log.warning(
                "Alpaca request failed (%s), retry %d", rs.outcome.exception(), rs.attempt_number
            ),
        )

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        resp = self.http.get(f"{self.DATA_URL}/v2/stocks/bars", params=params)
        if resp.status_code == 401:
            raise AlpacaAuthError(
                "Alpaca rejected the keys (HTTP 401). Check ALPACA_API_KEY / ALPACA_SECRET_KEY "
                "in .env (paper-trading keys work for market data)."
            )
        if resp.status_code == 403:
            raise AlpacaFeedNotPermitted(resp.text[:300])
        resp.raise_for_status()
        return resp.json()

    def _pages(
        self, symbols: Sequence[str], start: datetime, end: datetime, feed: str
    ) -> Iterator[list[Bar]]:
        params: dict[str, Any] = {
            "symbols": ",".join(symbols),
            "timeframe": "1Min",
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "limit": self.page_limit,
            "adjustment": "raw",
            "feed": feed,
            "sort": "asc",
        }
        while True:
            payload = self._retrying(self._get)(params)
            yield parse_bars(payload, feed)
            token = payload.get("next_page_token")
            if not token:
                return
            params = {**params, "page_token": token}

    def bars(self, symbols: Sequence[str], start: datetime, end: datetime) -> Iterator[list[Bar]]:
        """Yield bars page by page (so callers can write incrementally)."""
        if not symbols or start >= end:
            return
        try:
            yield from self._pages(symbols, start, end, self.feed)
        except AlpacaFeedNotPermitted as exc:
            if self.feed == "iex":
                raise
            log.warning(
                "Feed %r not permitted for this account (%s); switching to 'iex'. "
                "Set ALPACA_DATA_FEED=iex to silence this.",
                self.feed,
                exc,
            )
            self.feed = "iex"
            yield from self._pages(symbols, start, end, self.feed)

    def close(self) -> None:
        self.http.close()
