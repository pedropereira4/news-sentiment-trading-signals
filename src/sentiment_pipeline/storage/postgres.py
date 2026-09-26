"""PostgreSQL storage: schema bootstrap and idempotent batch writes.

Writes are safe to repeat (at-least-once delivery from Kafka): an article is upserted by id
and its signals are replaced, all inside one transaction per batch. Replaying a topic, or
re-enriching with another model, therefore overwrites rows instead of duplicating them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from importlib.resources import files
from typing import Any

import psycopg

from sentiment_pipeline.schemas import EnrichedArticle
from sentiment_pipeline.watchlist import Watchlist

_ARTICLE_COLUMNS = (
    "id", "source", "publisher", "title", "summary", "url", "source_tickers",
    "published_at", "timestamp_source", "ingested_at", "enriched_at",
    "sentiment", "score", "confidence", "topics",
    "llm_provider", "llm_model", "llm_latency_ms", "schema_version",
)  # fmt: skip

_SIGNAL_COLUMNS = (
    "article_id", "ticker", "published_at", "feed_timestamp",
    "sentiment", "score", "confidence", "event_type", "is_new_info", "llm_model",
)  # fmt: skip

_UPSERT_ARTICLE = (
    f"INSERT INTO articles ({', '.join(_ARTICLE_COLUMNS)}) "
    f"VALUES ({', '.join(['%s'] * len(_ARTICLE_COLUMNS))}) "
    "ON CONFLICT (id) DO UPDATE SET "
    + ", ".join(f"{c} = EXCLUDED.{c}" for c in _ARTICLE_COLUMNS if c != "id")
    + ", stored_at = now()"
)

_INSERT_SIGNAL = (
    f"INSERT INTO ticker_signals ({', '.join(_SIGNAL_COLUMNS)}) "
    f"VALUES ({', '.join(['%s'] * len(_SIGNAL_COLUMNS))})"
)

_UPSERT_TICKER = (
    "INSERT INTO tickers (ticker, name, cap_group, sector) VALUES (%s, %s, %s, %s) "
    "ON CONFLICT (ticker) DO UPDATE SET name = EXCLUDED.name, cap_group = EXCLUDED.cap_group, "
    "sector = EXCLUDED.sector, updated_at = now()"
)


def schema_sql() -> str:
    return files("sentiment_pipeline.storage").joinpath("schema.sql").read_text(encoding="utf-8")


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.transaction():
        conn.execute(schema_sql())


def sync_tickers(conn: psycopg.Connection, watchlist: Watchlist) -> int:
    rows = [(e.ticker, e.name, e.group, e.sector) for e in watchlist.tickers]
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(_UPSERT_TICKER, rows)
    return len(rows)


def article_row(a: EnrichedArticle) -> tuple[Any, ...]:
    return (
        a.id, a.source, a.publisher, a.title, a.summary, a.url, list(a.tickers),
        a.published_at, a.timestamp_source, a.ingested_at, a.enriched_at,
        a.sentiment, a.score, a.confidence, list(a.topics),
        a.llm_provider, a.llm_model, a.llm_latency_ms, a.schema_version,
    )  # fmt: skip


def signal_rows(a: EnrichedArticle) -> list[tuple[Any, ...]]:
    feed_ts = a.timestamp_source == "feed"
    return [
        (
            a.id,
            s.ticker,
            a.published_at,
            feed_ts,
            s.sentiment,
            s.score,
            s.confidence,
            s.event_type,
            s.is_new_info,
            a.llm_model,
        )  # fmt: skip
        for s in a.signals
    ]


def latest_per_id(articles: Iterable[EnrichedArticle]) -> list[EnrichedArticle]:
    """A redelivered message can put the same article twice in one batch: keep the last."""
    by_id: dict[str, EnrichedArticle] = {}
    for a in articles:
        by_id[a.id] = a
    return list(by_id.values())


def write_batch(conn: psycopg.Connection, articles: Sequence[EnrichedArticle]) -> tuple[int, int]:
    """Upsert articles and replace their signals atomically. Returns (articles, signals)."""
    batch = latest_per_id(articles)
    if not batch:
        return 0, 0
    signals = [row for a in batch for row in signal_rows(a)]
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(_UPSERT_ARTICLE, [article_row(a) for a in batch])
        cur.execute(
            "DELETE FROM ticker_signals WHERE article_id = ANY(%s)", ([a.id for a in batch],)
        )
        if signals:
            cur.executemany(_INSERT_SIGNAL, signals)
    return len(batch), len(signals)
