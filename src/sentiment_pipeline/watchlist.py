"""The stock universe (config/watchlist.yaml), shared by the producers and the enricher."""

from __future__ import annotations

from typing import Any, Literal

import yaml
from pydantic import BaseModel, field_validator

from sentiment_pipeline.schemas import normalise_ticker


class WatchlistEntry(BaseModel):
    ticker: str
    name: str | None = None  # company name: helps the LLM map "Nvidia" to NVDA
    group: Literal["large_cap", "small_mid_cap"]
    sector: str

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, v: Any) -> str:
        t = normalise_ticker(v)
        if t is None:
            raise ValueError(f"not a valid ticker: {v!r}")
        return t


class Watchlist(BaseModel):
    benchmark: str = "SPY"
    tickers: list[WatchlistEntry]

    @field_validator("tickers")
    @classmethod
    def _unique(cls, v: list[WatchlistEntry]) -> list[WatchlistEntry]:
        seen = [e.ticker for e in v]
        dupes = sorted({t for t in seen if seen.count(t) > 1})
        if dupes:
            raise ValueError(f"duplicate tickers in watchlist: {dupes}")
        return v

    @property
    def symbols(self) -> list[str]:
        return [e.ticker for e in self.tickers]

    def prompt_hint(self) -> str:
        """'AAPL (Apple), MSFT (Microsoft), ...' for the LLM prompt."""
        return ", ".join(f"{e.ticker} ({e.name})" if e.name else e.ticker for e in self.tickers)


def load_watchlist(path: str) -> Watchlist:
    with open(path, encoding="utf-8") as f:
        return Watchlist.model_validate(yaml.safe_load(f))
