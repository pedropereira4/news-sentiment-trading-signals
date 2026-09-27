"""The registered study protocol (config/study.yaml), loaded and validated."""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class Horizon(BaseModel):
    """'5min' / '1h' = wall-clock offsets; '1d' / '5d' = same clock time N trading days later."""

    label: str
    minutes: int = 0
    trading_days: int = 0

    @classmethod
    def parse(cls, label: str) -> Horizon:
        m = re.fullmatch(r"(\d+)\s*(min|h|d)", label.strip())
        if not m:
            raise ValueError(f"unknown horizon {label!r} (use e.g. 5min, 1h, 1d)")
        n, unit = int(m.group(1)), m.group(2)
        if unit == "min":
            return cls(label=label, minutes=n)
        if unit == "h":
            return cls(label=label, minutes=60 * n)
        return cls(label=label, trading_days=n)

    @property
    def offset(self) -> timedelta:
        return timedelta(minutes=self.minutes)


class StrongSignal(BaseModel):
    min_abs_score: float
    is_new_info: bool = True
    exclude_event_types: list[str] = Field(default_factory=list)


class DataSpec(BaseModel):
    collection_start: date
    watchlist: str
    news_source: str
    llm_model: str
    benchmark: str
    price_feed: str
    timestamps: str


class StoppingRule(BaseModel):
    min_small_cap_signals: int
    max_end_date: date


class Analysis(BaseModel):
    horizons: list[Horizon]
    primary_horizon: str
    strong_signal: StrongSignal

    @field_validator("horizons", mode="before")
    @classmethod
    def _parse(cls, v):
        return [Horizon.parse(h) if isinstance(h, str) else h for h in v]


class Protocol(BaseModel):
    registered_on: date
    hypothesis: str
    data: DataSpec
    stopping_rule: StoppingRule
    analysis: Analysis
    amendments: list = Field(default_factory=list)

    @property
    def primary(self) -> Horizon:
        return next(h for h in self.analysis.horizons if h.label == self.analysis.primary_horizon)


def load_protocol(path: str | Path = "config/study.yaml") -> Protocol:
    return Protocol.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
