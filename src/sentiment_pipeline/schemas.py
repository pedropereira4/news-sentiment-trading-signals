"""Message contracts for every Kafka topic.

news.raw       -> RawArticle
news.enriched  -> EnrichedArticle
news.dlq       -> DeadLetter

Schema v2 adds company-level signals: an article can move several stocks in different
directions ("Apple gains share from Samsung" is good for AAPL, bad for Samsung), so besides
the article-level tone the LLM returns one `TickerSignal` per company affected.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Sentiment = Literal["positive", "neutral", "negative"]

EventType = Literal[
    "earnings",  # quarterly/annual results
    "guidance",  # outlook raised/cut, pre-announcements
    "m&a",  # mergers, acquisitions, divestitures
    "analyst_rating",  # upgrades, downgrades, price targets
    "product",  # launches, contracts, partnerships
    "regulatory_legal",  # approvals (e.g. FDA), lawsuits, fines, investigations
    "management",  # CEO/CFO changes, board
    "capital",  # buybacks, dividends, offerings, debt
    "macro",  # rates, inflation, policy affecting the company
    "commentary",  # opinion, recaps, "stocks to watch" — usually no new information
    "other",
]
EVENT_TYPES: tuple[str, ...] = EventType.__args__  # type: ignore[attr-defined]

# How much we trust `published_at`: "feed" = timestamp given by the source,
# "ingested" = the source had none, so we used the ingestion time. Event studies must only
# use "feed" timestamps, otherwise returns are measured from the wrong moment.
TimestampSource = Literal["feed", "ingested"]

SCHEMA_VERSION = 2

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{0,5}([.\-][A-Z0-9]{1,2})?$")  # AAPL, BRK.B, RDS-A


def utcnow() -> datetime:
    return datetime.now(UTC)


def article_id(url: str | None, title: str) -> str:
    """Stable id used for de-duplication and idempotent writes."""
    basis = (url or "").strip() or title.strip().lower()
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


def normalise_ticker(value: Any) -> str | None:
    """'$aapl ' -> 'AAPL'. Returns None when the value is not a plausible US ticker."""
    if not isinstance(value, str):
        return None
    t = value.strip().lstrip("$").upper()
    return t if _TICKER_RE.match(t) else None


class RawArticle(BaseModel):
    schema_version: int = SCHEMA_VERSION
    id: str
    source: str
    title: str = Field(min_length=1)
    summary: str | None = None
    url: str | None = None
    # Tickers the *source* associates with the article (e.g. Finnhub company news).
    # Empty for general feeds: the LLM then identifies the companies itself.
    tickers: list[str] = Field(default_factory=list)
    published_at: datetime
    timestamp_source: TimestampSource = "feed"
    ingested_at: datetime = Field(default_factory=utcnow)

    @field_validator("tickers", mode="before")
    @classmethod
    def _normalise_tickers(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        out: list[str] = []
        for item in v:
            t = normalise_ticker(item)
            if t and t not in out:
                out.append(t)
        return out


class TickerSignal(BaseModel):
    """The LLM's read of what one article means for one company."""

    ticker: str
    sentiment: Sentiment
    score: float = Field(ge=-1.0, le=1.0, description="-1 very negative .. +1 very positive")
    confidence: float = Field(ge=0.0, le=1.0)
    event_type: EventType = "other"
    is_new_info: bool = Field(
        description="True if the article reports a new fact/event, False for recaps/opinion"
    )

    @field_validator("ticker", mode="before")
    @classmethod
    def _check_ticker(cls, v: Any) -> str:
        t = normalise_ticker(v)
        if t is None:
            raise ValueError(f"not a valid ticker: {v!r}")
        return t

    @field_validator("sentiment", mode="before")
    @classmethod
    def _normalise_sentiment(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("event_type", mode="before")
    @classmethod
    def _normalise_event_type(cls, v: Any) -> str:
        # "M&A" -> "m&a", "Analyst Rating" -> "analyst_rating"; an unexpected label
        # becomes "other" instead of throwing away an otherwise valid signal.
        t = v.strip().lower().replace(" ", "_").replace("-", "_") if isinstance(v, str) else ""
        return t if t in EVENT_TYPES else "other"


class LLMResult(BaseModel):
    """What the LLM returns for a single headline."""

    index: int = Field(ge=0)
    sentiment: Sentiment
    score: float = Field(ge=-1.0, le=1.0, description="-1 very negative .. +1 very positive")
    confidence: float = Field(ge=0.0, le=1.0)
    topics: list[str] = Field(default_factory=list)
    signals: list[TickerSignal] = Field(default_factory=list)

    @field_validator("sentiment", mode="before")
    @classmethod
    def _normalise_sentiment(cls, v: Any) -> Any:
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("topics", mode="before")
    @classmethod
    def _normalise_topics(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        seen: list[str] = []
        for t in v:
            t = " ".join(str(t).strip().lower().split())
            if t and t not in seen:
                seen.append(t[:40])
        return seen[:5]

    @field_validator("signals", mode="before")
    @classmethod
    def _keep_valid_signals(cls, v: Any) -> list[TickerSignal]:
        """Drop malformed signals (bad ticker, score out of range...) and duplicate tickers,
        keeping the article-level result: one bad signal must not discard the whole item."""
        if not isinstance(v, list):
            return []
        out: dict[str, TickerSignal] = {}
        for item in v:
            try:
                s = TickerSignal.model_validate(item)
            except ValueError:
                continue
            out.setdefault(s.ticker, s)
        return list(out.values())[:10]


class EnrichedArticle(RawArticle):
    sentiment: Sentiment
    score: float
    confidence: float
    topics: list[str]
    signals: list[TickerSignal] = Field(default_factory=list)
    llm_provider: str
    llm_model: str
    llm_latency_ms: int
    enriched_at: datetime = Field(default_factory=utcnow)

    @classmethod
    def from_parts(
        cls,
        raw: RawArticle,
        result: LLMResult,
        *,
        provider: str,
        model: str,
        latency_ms: int,
    ) -> EnrichedArticle:
        return cls(
            **raw.model_dump(exclude={"schema_version"}),
            sentiment=result.sentiment,
            score=result.score,
            confidence=result.confidence,
            topics=result.topics,
            signals=result.signals,
            llm_provider=provider,
            llm_model=model,
            llm_latency_ms=latency_ms,
        )


class DeadLetter(BaseModel):
    stage: Literal["enricher", "writer"]
    error: str
    source_topic: str
    partition: int | None = None
    offset: int | None = None
    payload: str
    failed_at: datetime = Field(default_factory=utcnow)
