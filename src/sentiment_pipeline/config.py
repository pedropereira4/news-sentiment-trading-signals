"""Typed configuration loaded from environment variables (see .env.example)."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field, field_validator
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


class FinnhubSettings(KafkaSettings):
    finnhub_api_key: str = ""
    watchlist_file: str = "config/watchlist.yaml"
    # Free tier allows ~60 requests/min: 40 tickers every 2 min is ~20/min.
    finnhub_poll_interval_seconds: int = Field(default=120, ge=30)
    finnhub_min_request_interval_seconds: float = Field(default=1.1, ge=0)
    # Finnhub filters by calendar day; on each cycle we ask for [today - lookback, today].
    # 7 days means the machine can be off for most of a week without missing news
    # (already-seen stories are skipped, so a longer window costs no extra LLM calls).
    finnhub_lookback_days: int = Field(default=7, ge=0, le=30)
    finnhub_state_file: str = "state/finnhub_seen_ids.json"
    finnhub_seen_cache_size: int = 20_000
    http_timeout_seconds: float = 15.0

    @field_validator("finnhub_api_key", mode="before")
    @classmethod
    def _clean_key(cls, v: str) -> str:
        return v.strip().strip("\"'").strip() if isinstance(v, str) else v


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
    # Hard cap on the reply. Without it, small models in JSON mode can loop on whitespace
    # until the request times out (seen with llama3.2:3b: 3 x 120s lost on one batch).
    llm_max_output_tokens: int = Field(default=4096, ge=256)


class EnricherSettings(KafkaSettings, LLMSettings):
    consumer_group: str = "sentiment-enricher"
    # Signals are kept only for these tickers (we only have prices for them). Empty = no filter.
    watchlist_file: str = "config/watchlist.yaml"
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


class PostgresSettings(_Base):
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "sentiment"
    postgres_user: str = "sentiment"
    postgres_password: str = ""

    @field_validator("postgres_password", "postgres_user", "postgres_db", mode="before")
    @classmethod
    def _clean(cls, v: str) -> str:
        return v.strip().strip("\"'").strip() if isinstance(v, str) else v

    @property
    def postgres_conninfo(self) -> str:
        from psycopg.conninfo import make_conninfo

        return make_conninfo(
            host=self.postgres_host,
            port=self.postgres_port,
            dbname=self.postgres_db,
            user=self.postgres_user,
            password=self.postgres_password,
            application_name="sentiment-pipeline",
        )


class PostgresWriterSettings(KafkaSettings, PostgresSettings):
    consumer_group: str = "postgres-writer"
    postgres_writer_batch_size: int = Field(default=200, ge=1)
    watchlist_file: str = "config/watchlist.yaml"


class PriceIngestorSettings(PostgresSettings):
    # Accepts the names used in .env here and the official Alpaca SDK names.
    alpaca_api_key: str = Field(
        default="", validation_alias=AliasChoices("ALPACA_API_KEY", "APCA_API_KEY_ID")
    )
    alpaca_secret_key: str = Field(
        default="", validation_alias=AliasChoices("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY")
    )
    # "sip" = all US exchanges (free plan: ~15 min delay); "iex" = one exchange, real time.
    alpaca_data_feed: Literal["sip", "iex"] = "sip"
    price_poll_interval_seconds: int = Field(default=900, ge=60)
    price_backfill_days: int = Field(default=10, ge=1, le=365)
    # SIP on the free plan is only served after ~15 minutes: stay behind that edge.
    price_delay_minutes: int = Field(default=16, ge=0)
    watchlist_file: str = "config/watchlist.yaml"

    @field_validator("alpaca_api_key", "alpaca_secret_key", mode="before")
    @classmethod
    def _clean_keys(cls, v: str) -> str:
        return v.strip().strip("\"'").strip() if isinstance(v, str) else v
