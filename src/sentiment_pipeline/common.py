"""Shared helpers: logging, graceful shutdown and Kafka client factories."""

from __future__ import annotations

import logging
import signal
import socket
import sys
from typing import Any

from confluent_kafka import Consumer, KafkaError, Message, Producer

log = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )


class GracefulShutdown:
    """Flip `running` to False on SIGINT/SIGTERM so loops can finish their batch."""

    def __init__(self) -> None:
        self.running = True
        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

    def _stop(self, signum: int, _frame: Any) -> None:
        log.info("Received signal %s, shutting down after current batch...", signum)
        self.running = False


def make_producer(bootstrap: str, client_id: str) -> Producer:
    return Producer(
        {
            "bootstrap.servers": bootstrap,
            "client.id": f"{client_id}-{socket.gethostname()}",
            "enable.idempotence": True,  # no duplicates on broker retries
            "acks": "all",
            "compression.type": "zstd",
            "linger.ms": 50,
        }
    )


def make_consumer(
    bootstrap: str, group_id: str, topics: list[str], max_poll_interval_ms: int = 300_000
) -> Consumer:
    """`max_poll_interval_ms` must exceed the worst-case time to process one batch.

    A slow LLM that exhausts its retries can block far longer than the 5-minute default;
    the consumer would then be evicted mid-batch, triggering a rebalance and reprocessing.
    """
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "client.id": f"{group_id}-{socket.gethostname()}",
            "enable.auto.commit": False,  # commit only after the batch is fully processed
            "auto.offset.reset": "earliest",
            "partition.assignment.strategy": "cooperative-sticky",
            "max.poll.interval.ms": max_poll_interval_ms,
            "session.timeout.ms": 45_000,
        }
    )
    consumer.subscribe(topics)
    return consumer


def delivery_report(err: KafkaError | None, msg: Message) -> None:
    if err is not None:
        log.error("Delivery failed for key=%s: %s", msg.key(), err)


def poll_batch(consumer: Consumer, max_messages: int, timeout_s: float) -> list[Message]:
    """Return up to `max_messages` valid messages, waiting at most `timeout_s`."""
    msgs = consumer.consume(num_messages=max_messages, timeout=timeout_s)
    good: list[Message] = []
    for m in msgs:
        if m.error():
            if m.error().code() != KafkaError._PARTITION_EOF:
                log.warning("Consumer error: %s", m.error())
            continue
        good.append(m)
    return good
