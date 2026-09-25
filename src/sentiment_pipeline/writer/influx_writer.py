"""news.enriched -> InfluxDB

Measurements
------------
article_sentiment  tags: source, sentiment, llm_model
                   fields: score, confidence, llm_latency_ms, title, url, topics
topic_mention      tags: topic, source, sentiment
                   fields: score, count
ticker_signal      tags: ticker, event_type, sentiment, source
                   fields: score, confidence, is_new_info, feed_timestamp, title

Idempotency: InfluxDB overwrites points with the same measurement + tag set +
timestamp. The timestamp is `published_at` plus a deterministic sub-second offset
derived from the article id, so re-processing a message (at-least-once delivery)
overwrites the same point instead of double counting — without adding a
high-cardinality `id` tag.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from confluent_kafka import Producer
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.rest import ApiException
from pydantic import ValidationError

from sentiment_pipeline.common import (
    GracefulShutdown,
    delivery_report,
    make_consumer,
    make_producer,
    poll_batch,
    setup_logging,
)
from sentiment_pipeline.config import WriterSettings
from sentiment_pipeline.schemas import DeadLetter, EnrichedArticle

log = logging.getLogger("writer")

NS = 1_000_000_000


def point_time_ns(article: EnrichedArticle) -> int:
    base = int(article.published_at.timestamp()) * NS
    return base + int(article.id[:8], 16) % NS


def to_points(article: EnrichedArticle) -> list[Point]:
    ts = point_time_ns(article)
    points = [
        Point("article_sentiment")
        .tag("source", article.source)
        .tag("sentiment", article.sentiment)
        .tag("llm_model", article.llm_model)
        .field("score", float(article.score))
        .field("confidence", float(article.confidence))
        .field("llm_latency_ms", int(article.llm_latency_ms))
        .field("title", article.title)
        .field("url", article.url or "")
        .field("topics", ", ".join(article.topics))
        .time(ts, WritePrecision.NS)
    ]
    for topic in article.topics:
        points.append(
            Point("topic_mention")
            .tag("topic", topic)
            .tag("source", article.source)
            .tag("sentiment", article.sentiment)
            .field("score", float(article.score))
            .field("count", 1)
            .time(ts, WritePrecision.NS)
        )
    for sig in article.signals:
        # Same timestamp as the article: (ticker, event_type, sentiment, source) + time is
        # unique per article because signals are de-duplicated by ticker.
        points.append(
            Point("ticker_signal")
            .tag("ticker", sig.ticker)
            .tag("event_type", sig.event_type)
            .tag("sentiment", sig.sentiment)
            .tag("source", article.source)
            .field("score", float(sig.score))
            .field("confidence", float(sig.confidence))
            .field("is_new_info", sig.is_new_info)
            .field("feed_timestamp", article.timestamp_source == "feed")
            .field("title", article.title)
            .time(ts, WritePrecision.NS)
        )
    return points


def fingerprint(token: str) -> str:
    """Identify a token in logs without printing it in full."""
    if not token:
        return "<empty>"
    return f"len={len(token)} starts={token[:4]!r} ends={token[-4:]!r}"


def preflight(influx: InfluxDBClient, settings: WriterSettings) -> None:
    """Fail fast with an actionable message instead of a bare 401 traceback."""
    log.info(
        "InfluxDB config: url=%s org=%r bucket=%r token=%s",
        settings.influxdb_url,
        settings.influxdb_org,
        settings.influxdb_bucket,
        fingerprint(settings.influxdb_token),
    )
    if not influx.ping():
        raise SystemExit(f"InfluxDB is not reachable at {settings.influxdb_url}")

    try:
        influx.write_api(write_options=SYNCHRONOUS).write(
            bucket=settings.influxdb_bucket,
            record=Point("writer_startup").field("ok", 1),
        )
    except Exception as exc:
        status = getattr(exc, "status", None)
        if status == 401:
            log.error(
                "InfluxDB rejected the token (401). The token the writer sends (%s) is not the "
                "one InfluxDB was initialised with. Compare it with the line `token = ...` in "
                "`docker compose exec influxdb cat /etc/influxdb2/influx-configs`. If they "
                "differ, either copy that value into INFLUXDB_TOKEN in .env, or wipe the "
                "InfluxDB volume so it is created again from .env.",
                fingerprint(settings.influxdb_token),
            )
        elif status == 404:
            log.error(
                "InfluxDB has no org %r or no bucket %r. Check INFLUXDB_ORG / INFLUXDB_BUCKET "
                "against what the InfluxDB UI shows at %s.",
                settings.influxdb_org,
                settings.influxdb_bucket,
                settings.influxdb_url,
            )
        raise
    log.info("InfluxDB write check passed")


def main() -> None:
    settings = WriterSettings()
    setup_logging(settings.log_level)
    consumer = make_consumer(
        settings.kafka_bootstrap_servers, settings.consumer_group, [settings.topic_enriched]
    )
    dlq: Producer = make_producer(settings.kafka_bootstrap_servers, "influx-writer")
    influx = InfluxDBClient(
        url=settings.influxdb_url, token=settings.influxdb_token, org=settings.influxdb_org
    )
    write_api = influx.write_api(write_options=SYNCHRONOUS)
    shutdown = GracefulShutdown()
    preflight(influx, settings)
    log.info("Writer started -> %s bucket=%s", settings.influxdb_url, settings.influxdb_bucket)

    try:
        while shutdown.running:
            msgs = poll_batch(consumer, settings.writer_batch_size, 2.0)
            if not msgs:
                continue

            cutoff = datetime.now(UTC) - timedelta(days=settings.max_point_age_days)
            points: list[Point] = []
            too_old = 0
            for m in msgs:
                try:
                    article = EnrichedArticle.model_validate_json(m.value())
                    if article.published_at < cutoff:
                        # Older than the bucket's retention: InfluxDB would reject the whole
                        # request. Feeds do occasionally carry months-old items.
                        too_old += 1
                        continue
                    points.extend(to_points(article))
                except ValidationError as exc:
                    dl = DeadLetter(
                        stage="writer",
                        error=str(exc)[:1000],
                        source_topic=m.topic(),
                        partition=m.partition(),
                        offset=m.offset(),
                        payload=(m.value() or b"").decode("utf-8", "replace"),
                    )
                    dlq.produce(
                        settings.topic_dlq,
                        key=m.key(),
                        value=dl.model_dump_json().encode(),
                        on_delivery=delivery_report,
                    )

            if points:
                try:
                    # Raises on failure -> exits without committing -> Docker restarts it.
                    write_api.write(bucket=settings.influxdb_bucket, record=points)
                except ApiException as exc:
                    if exc.status != 422:
                        raise
                    # Partial write: the valid points were stored, the rest were rejected
                    # (usually retention). Retrying would loop forever, so log and move on.
                    log.warning("InfluxDB partial write: %s", (exc.body or b"")[:300])
            dlq.flush(10)
            consumer.commit(asynchronous=False)
            log.info(
                "Wrote %d points from %d messages (%d skipped as older than %d days)",
                len(points),
                len(msgs),
                too_old,
                settings.max_point_age_days,
            )
    finally:
        consumer.close()
        influx.close()
        log.info("Writer stopped")


if __name__ == "__main__":
    main()
