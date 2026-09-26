"""RSS -> Kafka producer.

Polls every feed in config/feeds.yaml, normalises entries into RawArticle,
drops already-seen articles and publishes the rest to `news.raw`
(key = source, so each source keeps its order within a partition).
"""

from __future__ import annotations

import calendar
import logging
import time
from datetime import UTC, datetime
from typing import Any

import feedparser
import httpx
import yaml

from sentiment_pipeline.common import (
    GracefulShutdown,
    delivery_report,
    make_producer,
    setup_logging,
)
from sentiment_pipeline.config import ProducerSettings
from sentiment_pipeline.producer.seen_store import SeenStore
from sentiment_pipeline.schemas import RawArticle, TimestampSource, article_id, utcnow

log = logging.getLogger("producer")


def load_feeds(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        feeds = yaml.safe_load(f)["feeds"]
    return [{"source": fd["source"], "url": fd["url"]} for fd in feeds]


def _published(entry: Any) -> tuple[datetime, TimestampSource]:
    """Publication time and where it came from.

    Entries without a date fall back to the ingestion time, flagged as "ingested" so that
    time-sensitive analysis (event studies) can exclude them instead of measuring price
    reactions from the wrong moment.
    """
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        return datetime.fromtimestamp(calendar.timegm(parsed), tz=UTC), "feed"
    return utcnow(), "ingested"


def parse_feed(source: str, content: bytes) -> list[RawArticle]:
    parsed = feedparser.parse(content)
    articles = []
    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        url = entry.get("link")
        published_at, ts_source = _published(entry)
        articles.append(
            RawArticle(
                id=article_id(url, title),
                source=source,
                title=title,
                summary=entry.get("summary"),
                url=url,
                published_at=published_at,
                timestamp_source=ts_source,
            )
        )
    return articles


def main() -> None:
    settings = ProducerSettings()
    setup_logging(settings.log_level)
    feeds = load_feeds(settings.feeds_file)
    seen = SeenStore(settings.state_file, settings.seen_cache_size)
    producer = make_producer(settings.kafka_bootstrap_servers, "rss-producer")
    shutdown = GracefulShutdown()
    http = httpx.Client(
        timeout=settings.http_timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "realtime-sentiment-pipeline/0.1 (+github)"},
    )
    log.info(
        "Producer started: %d feeds, polling every %ss", len(feeds), settings.poll_interval_seconds
    )

    while shutdown.running:
        cycle_start = time.monotonic()
        published = 0
        for feed in feeds:
            try:
                resp = http.get(feed["url"])
                resp.raise_for_status()
                articles = parse_feed(feed["source"], resp.content)
            except Exception as exc:  # one broken feed must not stop the others
                log.warning("Feed %s failed: %s", feed["source"], exc)
                continue

            for art in articles:
                if art.id in seen:
                    continue
                producer.produce(
                    settings.topic_raw,
                    key=art.source.encode(),
                    value=art.model_dump_json().encode(),
                    on_delivery=delivery_report,
                )
                seen.add(art.id)
                published += 1
            producer.poll(0)

        remaining = producer.flush(30)
        if remaining:
            log.error("%d messages were not delivered", remaining)
        else:
            seen.save()  # only persist ids once Kafka has acknowledged them
        log.info("Cycle done: %d new articles published", published)

        # Sleep in short steps so SIGTERM is handled quickly.
        while shutdown.running and time.monotonic() - cycle_start < settings.poll_interval_seconds:
            time.sleep(1)

    producer.flush(10)
    seen.save()
    http.close()
    log.info("Producer stopped")


if __name__ == "__main__":
    main()
