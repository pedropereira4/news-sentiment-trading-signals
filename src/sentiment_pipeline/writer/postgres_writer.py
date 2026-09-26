"""news.enriched -> PostgreSQL (system of record for the event study)

Runs as its own consumer group, independent of the InfluxDB writer: each sink keeps its
own offsets, so one can lag, fail or be replayed without touching the other. A new
group starts from the earliest offset, so on first start it backfills everything still
retained in the topic.

Offsets are committed only after the database transaction commits (at-least-once);
the idempotent upserts in storage.postgres make redelivery harmless.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import psycopg
from confluent_kafka import Message, Producer
from pydantic import ValidationError

from sentiment_pipeline.common import (
    GracefulShutdown,
    delivery_report,
    make_consumer,
    make_producer,
    poll_batch,
    setup_logging,
)
from sentiment_pipeline.config import PostgresWriterSettings
from sentiment_pipeline.schemas import DeadLetter, EnrichedArticle
from sentiment_pipeline.storage.postgres import ensure_schema, sync_tickers, write_batch
from sentiment_pipeline.watchlist import load_watchlist

log = logging.getLogger("postgres-writer")


def connect(conninfo: str, attempts: int = 10, delay_s: float = 3.0) -> psycopg.Connection:
    """Postgres can take a few seconds to accept connections after a cold start."""
    for attempt in range(1, attempts + 1):
        try:
            return psycopg.connect(conninfo)
        except psycopg.OperationalError as exc:
            if attempt == attempts:
                raise
            log.warning("Postgres not ready (%s), retry %d/%d", exc, attempt, attempts)
            time.sleep(delay_s)
    raise RuntimeError("unreachable")


def parse_messages(msgs: list[Message], dlq: Producer, dlq_topic: str) -> list[EnrichedArticle]:
    articles = []
    for m in msgs:
        try:
            articles.append(EnrichedArticle.model_validate_json(m.value()))
        except ValidationError as exc:
            dl = DeadLetter(
                stage="postgres_writer",
                error=str(exc)[:1000],
                source_topic=m.topic(),
                partition=m.partition(),
                offset=m.offset(),
                payload=(m.value() or b"").decode("utf-8", "replace"),
            )
            dlq.produce(
                dlq_topic,
                key=m.key(),
                value=dl.model_dump_json().encode(),
                on_delivery=delivery_report,
            )
    return articles


def main() -> None:
    settings = PostgresWriterSettings()
    setup_logging(settings.log_level)

    conn = connect(settings.postgres_conninfo)
    ensure_schema(conn)
    if settings.watchlist_file and Path(settings.watchlist_file).exists():
        n = sync_tickers(conn, load_watchlist(settings.watchlist_file))
        log.info("Synced %d tickers from %s", n, settings.watchlist_file)

    consumer = make_consumer(
        settings.kafka_bootstrap_servers, settings.consumer_group, [settings.topic_enriched]
    )
    dlq = make_producer(settings.kafka_bootstrap_servers, "postgres-writer")
    shutdown = GracefulShutdown()
    log.info(
        "Postgres writer started -> %s:%s/%s",
        settings.postgres_host,
        settings.postgres_port,
        settings.postgres_db,
    )

    try:
        while shutdown.running:
            msgs = poll_batch(consumer, settings.postgres_writer_batch_size, 2.0)
            if not msgs:
                continue
            articles = parse_messages(msgs, dlq, settings.topic_dlq)
            # Raises on any DB error -> no commit -> the batch is re-consumed after restart.
            n_articles, n_signals = write_batch(conn, articles)
            if dlq.flush(10) > 0:
                raise RuntimeError("Failed to deliver dead letters to Kafka")
            consumer.commit(asynchronous=False)
            log.info(
                "Stored %d articles, %d signals from %d messages (%d invalid)",
                n_articles,
                n_signals,
                len(msgs),
                len(msgs) - len(articles),
            )
    finally:
        consumer.close()
        conn.close()
        log.info("Postgres writer stopped")


if __name__ == "__main__":
    main()
