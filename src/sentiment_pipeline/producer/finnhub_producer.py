"""Finnhub company news -> Kafka producer.

For every ticker in config/watchlist.yaml, polls Finnhub's /company-news endpoint, turns
the items into RawArticle (with `tickers` already filled in, so the LLM gets them as a hint)
and publishes the new ones to `news.raw`.

Details that matter downstream:
- One story is often returned for several watchlist tickers. Within a cycle those copies are
  merged into a single article carrying all its tickers, so it is classified (and later
  traded) once.
- `published_at` is Finnhub's own `datetime` when present (timestamp_source="feed").
- Requests are paced below the free-tier limit; 429/5xx/network errors are retried with
  backoff, while an invalid key stops the service with a clear message.
- The API key travels in the X-Finnhub-Token header, never in the URL, so it cannot leak
  into logs or tracebacks.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from sentiment_pipeline.common import (
    GracefulShutdown,
    delivery_report,
    make_producer,
    setup_logging,
)
from sentiment_pipeline.config import FinnhubSettings
from sentiment_pipeline.producer.seen_store import SeenStore
from sentiment_pipeline.schemas import RawArticle, article_id, normalise_ticker, utcnow
from sentiment_pipeline.watchlist import Watchlist, WatchlistEntry, load_watchlist

log = logging.getLogger("finnhub-producer")

__all__ = ["Watchlist", "WatchlistEntry", "load_watchlist"]  # re-exported for callers

SOURCE = "finnhub"


# ---------------------------------------------------------------- API client
class FinnhubAuthError(RuntimeError):
    """Invalid or missing API key: retrying cannot help."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return isinstance(exc, httpx.TransportError)


class FinnhubClient:
    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_s: float = 15.0,
        min_interval_s: float = 1.1,
        max_retries: int = 4,
        backoff_initial_s: float = 2.0,
        http: httpx.Client | None = None,
    ) -> None:
        self.http = http or httpx.Client(timeout=timeout_s)
        self.http.headers["X-Finnhub-Token"] = api_key
        self.min_interval_s = min_interval_s
        self._last_request = 0.0
        self._retrying = retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential_jitter(
                initial=backoff_initial_s, max=60, jitter=backoff_initial_s
            ),
            reraise=True,
            before_sleep=lambda rs: log.warning(
                "Finnhub request failed (%s), retry %d",
                rs.outcome.exception(),
                rs.attempt_number,
            ),
        )

    def _pace(self) -> None:
        wait = self._last_request + self.min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _get(self, path: str, params: dict[str, str]) -> Any:
        self._pace()
        resp = self.http.get(f"{self.BASE_URL}{path}", params=params)
        if resp.status_code in (401, 403):
            raise FinnhubAuthError(
                f"Finnhub rejected the API key (HTTP {resp.status_code}). "
                "Check FINNHUB_API_KEY in .env against https://finnhub.io/dashboard"
            )
        resp.raise_for_status()
        return resp.json()

    def company_news(self, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
        params = {"symbol": symbol, "from": start.isoformat(), "to": end.isoformat()}
        data = self._retrying(self._get)("/company-news", params)
        if isinstance(data, dict) and "error" in data:
            raise ValueError(f"Finnhub error for {symbol}: {data['error']}")
        if not isinstance(data, list):
            raise ValueError(f"Unexpected Finnhub response for {symbol}: {str(data)[:200]}")
        return data

    def close(self) -> None:
        self.http.close()


# ---------------------------------------------------------------- parsing
def _related_tickers(related: Any, universe: set[str]) -> list[str]:
    """Finnhub's `related` is a comma-separated string; keep the watchlist ones only."""
    if not isinstance(related, str):
        return []
    out = []
    for part in related.split(","):
        t = normalise_ticker(part)
        if t and t in universe:
            out.append(t)
    return out


def parse_company_news(
    items: Iterable[dict[str, Any]], symbol: str, universe: set[str]
) -> list[RawArticle]:
    articles = []
    for item in items:
        headline = (item.get("headline") or "").strip()
        if not headline:
            continue
        url = item.get("url") or None
        ts = item.get("datetime")
        if isinstance(ts, int | float) and ts > 0:
            published_at, ts_source = datetime.fromtimestamp(ts, tz=UTC), "feed"
        else:
            published_at, ts_source = utcnow(), "ingested"
        articles.append(
            RawArticle(
                id=article_id(url, headline),
                source=SOURCE,
                publisher=(item.get("source") or None),
                title=headline,
                summary=(item.get("summary") or None),
                url=url,
                tickers=[symbol, *_related_tickers(item.get("related"), universe)],
                published_at=published_at,
                timestamp_source=ts_source,
            )
        )
    return articles


def merge_articles(into: dict[str, RawArticle], articles: Iterable[RawArticle]) -> None:
    """Collapse copies of the same story (returned for several tickers) into one article."""
    for art in articles:
        existing = into.get(art.id)
        if existing is None:
            into[art.id] = art
            continue
        tickers = existing.tickers + [t for t in art.tickers if t not in existing.tickers]
        into[art.id] = existing.model_copy(update={"tickers": tickers})


# ---------------------------------------------------------------- service
def main() -> None:
    settings = FinnhubSettings()
    setup_logging(settings.log_level)
    if not settings.finnhub_api_key:
        raise SystemExit("FINNHUB_API_KEY is not set. Get a free key at https://finnhub.io")

    watchlist = load_watchlist(settings.watchlist_file)
    universe = set(watchlist.symbols)
    client = FinnhubClient(
        settings.finnhub_api_key,
        timeout_s=settings.http_timeout_seconds,
        min_interval_s=settings.finnhub_min_request_interval_seconds,
    )
    seen = SeenStore(settings.finnhub_state_file, settings.finnhub_seen_cache_size)
    producer = make_producer(settings.kafka_bootstrap_servers, "finnhub-producer")
    shutdown = GracefulShutdown()
    log.info(
        "Finnhub producer started: %d tickers, polling every %ss",
        len(universe),
        settings.finnhub_poll_interval_seconds,
    )

    try:
        while shutdown.running:
            cycle_start = time.monotonic()
            end = datetime.now(UTC).date()
            start = end - timedelta(days=settings.finnhub_lookback_days)

            batch: dict[str, RawArticle] = {}
            failed: list[str] = []
            for symbol in watchlist.symbols:
                if not shutdown.running:
                    break
                try:
                    items = client.company_news(symbol, start, end)
                except FinnhubAuthError as exc:
                    raise SystemExit(str(exc)) from None
                except Exception as exc:  # one bad ticker must not stop the others
                    log.warning("News for %s failed: %s", symbol, exc)
                    failed.append(symbol)
                    continue
                merge_articles(batch, parse_company_news(items, symbol, universe))

            new = sorted(
                (a for a in batch.values() if a.id not in seen), key=lambda a: a.published_at
            )
            for art in new:
                producer.produce(
                    settings.topic_raw,
                    key=art.source.encode(),
                    value=art.model_dump_json().encode(),
                    on_delivery=delivery_report,
                )
                seen.add(art.id)
                producer.poll(0)

            remaining = producer.flush(30)
            if remaining:
                log.error("%d messages were not delivered", remaining)
            else:
                seen.save()  # only persist ids once Kafka has acknowledged them
            log.info(
                "Cycle done: %d stories fetched, %d new published%s",
                len(batch),
                len(new),
                f", failed: {failed}" if failed else "",
            )

            while (
                shutdown.running
                and time.monotonic() - cycle_start < settings.finnhub_poll_interval_seconds
            ):
                time.sleep(1)
    finally:
        producer.flush(10)
        seen.save()
        client.close()
        log.info("Finnhub producer stopped")


if __name__ == "__main__":
    main()
