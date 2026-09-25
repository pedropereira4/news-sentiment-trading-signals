"""Typed configuration loaded from environment variables (see .env.example)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class _Base(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"


class KafkaSettings(_Base):
    kafka_bootstrap_servers: str = "localhost:29092"
    topic_raw: str = "news.raw"
    topic_enriched: str = "news.enriched"
    topic_dlq: str = "news.dlq"


class ProducerSettings(KafkaSettings):
    feeds_file: str = "config/feeds.yaml"
    poll_interval_seconds: int = Field(default=120, ge=10)
    state_file: str = "state/seen_ids.json"
    seen_cache_size: int = 5000
    http_timeout_seconds: float = 15.0


class LLMSettings(_Base):
    llm_provider: Literal["ollama", "openrouter", "anthropic", "mock"] = "mock"
    llm_model: str = "llama3.2:3b"
    openrouter_api_key: str = ""
    anthropic_api_key: str = ""
    # From a container, the host machine is host.docker.internal (Docker Desktop);
    # running the service directly on the host, use http://localhost:11434
    ollama_base_url: str = "http://host.docker.internal:11434"
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 4


class EnricherSettings(KafkaSettings, LLMSettings):
    consumer_group: str = "sentiment-enricher"
    enricher_batch_size: int = Field(default=10, ge=1, le=50)
    enricher_batch_timeout_seconds: float = 5.0

    @property
    def max_poll_interval_ms(self) -> int:
        """Worst case for one batch: every LLM attempt times out, then each item is
        retried individually. Kafka must tolerate that, or it evicts us mid-batch.
        """
        worst_case_s = self.llm_timeout_seconds * self.llm_max_retries * 2 + 120
        return max(300_000, int(worst_case_s * 1000))


class WriterSettings(KafkaSettings):
    consumer_group: str = "influx-writer"
    influxdb_url: str = "http://localhost:8086"
    influxdb_org: str = "sentiment"
    influxdb_bucket: str = "news_sentiment"
    influxdb_token: str = ""
    writer_batch_size: int = 200
    # Must stay below the bucket's retention (30d in docker-compose.yml): InfluxDB rejects
    # the whole request when any point falls outside it.
    max_point_age_days: int = 25

    @field_validator(
        "influxdb_token", "influxdb_org", "influxdb_bucket", "influxdb_url", mode="before"
    )
    @classmethod
    def _clean(cls, v: str) -> str:
        """Tolerate values quoted or padded in .env (a classic source of 401s)."""
        return v.strip().strip("\"'").strip() if isinstance(v, str) else v
