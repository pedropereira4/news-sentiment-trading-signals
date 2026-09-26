"""Alpaca 1-minute bars -> PostgreSQL (price_bars), for the watchlist and the benchmark.

Each cycle asks for everything after the last stored bar of each ticker, up to
`now - PRICE_DELAY_MINUTES`. Tickers with no history yet (first run, or newly added to the
watchlist) are backfilled PRICE_BACKFILL_DAYS in a separate request, so adding one ticker
does not re-download the whole universe. Writes are upserts, so overlaps are harmless.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sentiment_pipeline.common import GracefulShutdown, setup_logging
from sentiment_pipeline.config import PriceIngestorSettings
from sentiment_pipeline.market.alpaca import AlpacaAuthError, AlpacaBarsClient
from sentiment_pipeline.storage.postgres import (
    ensure_schema,
    last_bar_times,
    sync_tickers,
    upsert_bars,
)
from sentiment_pipeline.watchlist import load_watchlist
from sentiment_pipeline.writer.postgres_writer import connect

log = logging.getLogger("price-ingestor")

ONE_MINUTE = timedelta(minutes=1)


def plan_requests(
    symbols: Sequence[str],
    last: dict[str, datetime],
    now: datetime,
    backfill_days: int,
    delay_minutes: int,
) -> list[tuple[list[str], datetime, datetime]]:
    """Split the universe into (symbols, start, end) requests.

    - symbols with history: one request from the oldest "last bar + 1 min"
    - symbols without history: one backfill request
    """
    end = (now - timedelta(minutes=delay_minutes)).replace(second=0, microsecond=0)
    known = [s for s in symbols if s in last]
    new = [s for s in symbols if s not in last]
    plan = []
    if known:
        start = min(last[s] for s in known) + ONE_MINUTE
        if start < end:
            plan.append((known, start, end))
    if new:
        plan.append((new, end - timedelta(days=backfill_days), end))
    return plan


def run_cycle(conn, client: AlpacaBarsClient, symbols: Sequence[str], settings) -> int:
    stored = 0
    plan = plan_requests(
        symbols,
        last_bar_times(conn),
        datetime.now(UTC),
        settings.price_backfill_days,
        settings.price_delay_minutes,
    )
    for group, start, end in plan:
        for page in client.bars(group, start, end):
            stored += upsert_bars(conn, page)  # commit page by page: progress survives a crash
        log.info(
            "%d symbols %s -> %s (feed=%s)",
            len(group),
            start.isoformat(timespec="minutes"),
            end.isoformat(timespec="minutes"),
            client.feed,
        )
    return stored


def main() -> None:
    settings = PriceIngestorSettings()
    setup_logging(settings.log_level)
    if not (settings.alpaca_api_key and settings.alpaca_secret_key):
        raise SystemExit("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")

    watchlist = load_watchlist(settings.watchlist_file)
    symbols = [*watchlist.symbols, watchlist.benchmark]
    conn = connect(settings.postgres_conninfo)
    ensure_schema(conn)
    sync_tickers(conn, watchlist)
    client = AlpacaBarsClient(
        settings.alpaca_api_key, settings.alpaca_secret_key, feed=settings.alpaca_data_feed
    )
    shutdown = GracefulShutdown()
    log.info(
        "Price ingestor started: %d symbols (incl. %s), every %ss, feed=%s",
        len(symbols),
        watchlist.benchmark,
        settings.price_poll_interval_seconds,
        client.feed,
    )

    try:
        while shutdown.running:
            cycle_start = time.monotonic()
            try:
                n = run_cycle(conn, client, symbols, settings)
                log.info("Cycle done: %d bars stored", n)
            except AlpacaAuthError as exc:
                raise SystemExit(str(exc)) from None
            except Exception:
                log.exception("Price cycle failed; retrying next cycle")
            while (
                shutdown.running
                and time.monotonic() - cycle_start < settings.price_poll_interval_seconds
            ):
                time.sleep(1)
    finally:
        client.close()
        conn.close()
        log.info("Price ingestor stopped")


if __name__ == "__main__":
    main()
