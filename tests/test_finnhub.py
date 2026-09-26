from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from sentiment_pipeline.producer.finnhub_producer import (
    FinnhubAuthError,
    FinnhubClient,
    Watchlist,
    load_watchlist,
    merge_articles,
    parse_company_news,
)

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = {"AAPL", "MSFT", "NVDA"}


def news_item(**overrides):
    item = {
        "category": "company",
        "datetime": 1790330400,  # 2026-09-25 10:00:00 UTC
        "headline": "Apple and Microsoft sign cloud deal",
        "id": 123,
        "image": "",
        "related": "AAPL",
        "source": "Reuters",
        "summary": "The two companies agreed...",
        "url": "https://finnhub.io/api/news?id=abc",
    }
    item.update(overrides)
    return item


# ---------------------------------------------------------------- parsing
def test_parse_company_news_maps_fields_and_timestamps():
    [art] = parse_company_news([news_item()], "AAPL", UNIVERSE)
    assert art.source == "finnhub" and art.publisher == "Reuters"
    assert art.tickers == ["AAPL"]
    assert art.published_at == datetime(2026, 9, 25, 10, 0, tzinfo=UTC)
    assert art.timestamp_source == "feed"


def test_parse_company_news_skips_empty_headlines_and_flags_missing_dates():
    items = [news_item(headline="  "), news_item(datetime=0, url="https://x/2")]
    [art] = parse_company_news(items, "AAPL", UNIVERSE)
    assert art.timestamp_source == "ingested"


def test_related_tickers_are_kept_only_if_in_the_watchlist():
    [art] = parse_company_news([news_item(related="AAPL,MSFT,SSNLF, bad!")], "AAPL", UNIVERSE)
    assert art.tickers == ["AAPL", "MSFT"]


def test_same_story_for_two_tickers_becomes_one_article():
    batch = {}
    merge_articles(batch, parse_company_news([news_item(related="AAPL")], "AAPL", UNIVERSE))
    merge_articles(batch, parse_company_news([news_item(related="MSFT")], "MSFT", UNIVERSE))
    [art] = batch.values()
    assert art.tickers == ["AAPL", "MSFT"]


# ---------------------------------------------------------------- client
def make_client(handler) -> FinnhubClient:
    return FinnhubClient(
        "secret-key",
        min_interval_s=0,
        backoff_initial_s=0,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_client_request_shape_keeps_the_key_out_of_the_url():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json=[news_item()])

    items = make_client(handler).company_news("AAPL", date(2026, 9, 24), date(2026, 9, 25))
    req = seen["request"]
    assert req.url.path == "/api/v1/company-news"
    assert dict(req.url.params) == {"symbol": "AAPL", "from": "2026-09-24", "to": "2026-09-25"}
    assert req.headers["X-Finnhub-Token"] == "secret-key"
    assert "secret-key" not in str(req.url)
    assert items[0]["headline"].startswith("Apple")


def test_client_retries_rate_limits_then_succeeds():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429) if len(calls) < 3 else httpx.Response(200, json=[])

    assert make_client(handler).company_news("AAPL", date(2026, 9, 25), date(2026, 9, 25)) == []
    assert len(calls) == 3


def test_client_does_not_retry_an_invalid_key():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": "Invalid API key"})

    with pytest.raises(FinnhubAuthError):
        make_client(handler).company_news("AAPL", date(2026, 9, 25), date(2026, 9, 25))
    assert len(calls) == 1


def test_client_rejects_error_payloads():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "You don't have access to this resource."})

    with pytest.raises(ValueError, match="access"):
        make_client(handler).company_news("AAPL", date(2026, 9, 25), date(2026, 9, 25))


# ---------------------------------------------------------------- watchlist
def test_repo_watchlist_matches_the_study_design():
    wl = load_watchlist(str(ROOT / "config" / "watchlist.yaml"))
    groups = [e.group for e in wl.tickers]
    assert groups.count("large_cap") == 10 and groups.count("small_mid_cap") == 30
    # Same sectors on both sides, so group effects are not sector effects.
    sectors = {g: {e.sector for e in wl.tickers if e.group == g} for g in set(groups)}
    assert sectors["large_cap"] == sectors["small_mid_cap"]


def test_watchlist_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        Watchlist.model_validate(
            {
                "tickers": [
                    {"ticker": "aapl", "group": "large_cap", "sector": "tech"},
                    {"ticker": "AAPL", "group": "large_cap", "sector": "tech"},
                ]
            }
        )
