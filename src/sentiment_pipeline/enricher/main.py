"""news.raw -> LLM -> news.enriched

Consumes raw headlines in micro-batches (one LLM call per batch to cut cost and
latency), publishes enriched records and commits offsets only after the batch
has been fully produced (at-least-once). Items the LLM cannot classify are
retried individually and, if they still fail, sent to the dead-letter topic.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

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
from sentiment_pipeline.config import EnricherSettings
from sentiment_pipeline.llm import LLMClient, build_client
from sentiment_pipeline.schemas import DeadLetter, EnrichedArticle, RawArticle
from sentiment_pipeline.watchlist import Watchlist, load_watchlist

log = logging.getLogger("enricher")


def send_to_dlq(producer: Producer, topic: str, msg: Message, error: str) -> None:
    value = msg.value() or b""
    dl = DeadLetter(
        stage="enricher",
        error=error[:1000],
        source_topic=msg.topic(),
        partition=msg.partition(),
        offset=msg.offset(),
        payload=value.decode("utf-8", errors="replace"),
    )
    producer.produce(
        topic, key=msg.key(), value=dl.model_dump_json().encode(), on_delivery=delivery_report
    )


def enrich(
    client: LLMClient, articles: Sequence[RawArticle]
) -> tuple[list[EnrichedArticle], dict[int, str]]:
    """Classify a batch. Returns (enriched, {index: error}) — never raises for bad items."""
    enriched: dict[int, EnrichedArticle] = {}
    errors: dict[int, str] = {}

    def _run(indices: list[int]) -> None:
        batch = [articles[i] for i in indices]
        try:
            out = client.classify(batch)
        except Exception as exc:
            if len(indices) == 1:
                errors[indices[0]] = f"LLM error: {exc}"
                return
            log.warning("Batch of %d failed (%s); retrying one by one", len(indices), exc)
            for i in indices:
                _run([i])
            return

        missing = []
        for local_idx, global_idx in enumerate(indices):
            result = out.results.get(local_idx)
            if result is None:
                missing.append(global_idx)
                continue
            per_item_latency = out.latency_ms // max(len(indices), 1)
            enriched[global_idx] = EnrichedArticle.from_parts(
                articles[global_idx],
                result,
                provider=client.provider,
                model=client.model,
                latency_ms=per_item_latency,
            )
        if missing and len(indices) > 1:
            for i in missing:
                _run([i])
        elif missing:
            errors[missing[0]] = "LLM returned no valid result for this item"

    if articles:
        _run(list(range(len(articles))))
    return [enriched[i] for i in sorted(enriched)], errors


def restrict_to_universe(
    articles: Sequence[EnrichedArticle], universe: set[str] | None
) -> tuple[list[EnrichedArticle], list[str]]:
    """Keep only signals for tickers we track (and have prices for).

    The LLM is told the universe, but it can still return other or invented tickers
    ("TOKIO", exchange codes...). Returns the filtered articles and the dropped tickers.
    """
    if not universe:
        return list(articles), []
    out: list[EnrichedArticle] = []
    dropped: list[str] = []
    for art in articles:
        keep = [s for s in art.signals if s.ticker in universe]
        dropped.extend(s.ticker for s in art.signals if s.ticker not in universe)
        out.append(
            art if len(keep) == len(art.signals) else art.model_copy(update={"signals": keep})
        )
    return out, dropped


def _load_universe(path: str) -> Watchlist | None:
    if not path:
        return None
    if not Path(path).exists():
        log.warning("Watchlist %s not found: signals will not be filtered", path)
        return None
    return load_watchlist(path)


def main() -> None:
    settings = EnricherSettings()
    setup_logging(settings.log_level)
    watchlist = _load_universe(settings.watchlist_file)
    universe = set(watchlist.symbols) if watchlist else None
    client = build_client(settings, watchlist.prompt_hint() if watchlist else None)
    consumer = make_consumer(
        settings.kafka_bootstrap_servers,
        settings.consumer_group,
        [settings.topic_raw],
        max_poll_interval_ms=settings.max_poll_interval_ms,
    )
    producer = make_producer(settings.kafka_bootstrap_servers, "enricher")
    shutdown = GracefulShutdown()
    log.info(
        "Enricher started: provider=%s model=%s batch=%d universe=%s tickers",
        client.provider,
        client.model,
        settings.enricher_batch_size,
        len(universe) if universe else "all",
    )

    try:
        while shutdown.running:
            msgs = poll_batch(
                consumer, settings.enricher_batch_size, settings.enricher_batch_timeout_seconds
            )
            if not msgs:
                continue

            valid_msgs: list[Message] = []
            articles: list[RawArticle] = []
            for m in msgs:
                try:
                    articles.append(RawArticle.model_validate_json(m.value()))
                    valid_msgs.append(m)
                except ValidationError as exc:
                    send_to_dlq(producer, settings.topic_dlq, m, f"Invalid RawArticle: {exc}")

            enriched, errors = enrich(client, articles)
            enriched, dropped = restrict_to_universe(enriched, universe)
            for art in enriched:
                producer.produce(
                    settings.topic_enriched,
                    key=art.source.encode(),
                    value=art.model_dump_json().encode(),
                    on_delivery=delivery_report,
                )
            for idx, err in errors.items():
                send_to_dlq(producer, settings.topic_dlq, valid_msgs[idx], err)

            if producer.flush(30) > 0:
                # Don't commit: the batch will be re-consumed after restart.
                raise RuntimeError("Failed to deliver enriched batch to Kafka")
            consumer.commit(asynchronous=False)

            summary = ", ".join(f"{a.sentiment[:3]}:{a.score:+.2f}" for a in enriched[:5])
            tickers = ", ".join(f"{s.ticker}:{s.score:+.2f}" for a in enriched for s in a.signals)
            log.info(
                "Batch: %d in, %d enriched, %d DLQ [%s...] signals [%s]%s",
                len(msgs),
                len(enriched),
                len(msgs) - len(enriched),
                summary,
                tickers[:300],
                f" dropped off-watchlist {sorted(set(dropped))}" if dropped else "",
            )
    finally:
        consumer.close()
        producer.flush(10)
        client.close()
        log.info("Enricher stopped")


if __name__ == "__main__":
    main()
