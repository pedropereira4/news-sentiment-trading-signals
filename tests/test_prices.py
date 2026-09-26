"""Price ingestion: Alpaca client, request planning and storage."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from sentiment_pipeline.market.alpaca import (
    AlpacaAuthError,
    AlpacaBarsClient,
    Bar,
    parse_bars,
)
from sentiment_pipeline.market.price_ingestor import plan_requests

NOW = datetime(2026, 9, 28, 15, 0, 30, tzinfo=UTC)


def bar_json(minute: int, close: float = 100.0) -> dict:
    return {
        "t": f"2026-09-28T13:{minute:02d}:00Z",
        "o": close,
        "h": close + 1,
        "l": close - 1,
        "c": close,
        "v": 1200,
        "n": 35,
        "vw": close + 0.1,
    }


def make_client(handler, feed: str = "sip") -> AlpacaBarsClient:
    return AlpacaBarsClient(
        "key",
        "secret",
        feed=feed,
        backoff_initial_s=0,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )


# ---------------------------------------------------------------- parsing & client
def test_parse_bars_maps_every_field():
    [bar] = parse_bars({"bars": {"AAPL": [bar_json(30, 210.5)]}}, feed="sip")
    assert bar == Bar(
        ticker="AAPL",
        ts=datetime(2026, 9, 28, 13, 30, tzinfo=UTC),
        open=210.5,
        high=211.5,
        low=209.5,
        close=210.5,
        volume=1200,
        trade_count=35,
        vwap=210.6,
        feed="sip",
    )
    assert parse_bars({"bars": None}, feed="sip") == []


def test_client_follows_pagination_and_sends_auth_headers():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if "page_token" not in request.url.params:
            return httpx.Response(
                200, json={"bars": {"AAPL": [bar_json(30)]}, "next_page_token": "p2"}
            )
        return httpx.Response(200, json={"bars": {"SPY": [bar_json(30)]}, "next_page_token": None})

    client = make_client(handler)
    start, end = NOW - timedelta(hours=2), NOW
    pages = list(client.bars(["AAPL", "SPY"], start, end))
    assert [[b.ticker for b in p] for p in pages] == [["AAPL"], ["SPY"]]
    first = calls[0]
    assert first.url.path == "/v2/stocks/bars"
    assert first.url.params["symbols"] == "AAPL,SPY"
    assert first.url.params["timeframe"] == "1Min" and first.url.params["feed"] == "sip"
    assert first.url.params["start"].endswith("Z")
    assert first.headers["APCA-API-KEY-ID"] == "key"
    assert calls[1].url.params["page_token"] == "p2"


def test_client_falls_back_to_iex_when_sip_is_not_permitted():
    feeds = []

    def handler(request: httpx.Request) -> httpx.Response:
        feeds.append(request.url.params["feed"])
        if request.url.params["feed"] == "sip":
            return httpx.Response(403, json={"message": "subscription does not permit SIP"})
        return httpx.Response(200, json={"bars": {"AAPL": [bar_json(30)]}})

    client = make_client(handler)
    [page] = list(client.bars(["AAPL"], NOW - timedelta(hours=1), NOW))
    assert feeds == ["sip", "iex"] and client.feed == "iex" and page[0].feed == "iex"


def test_client_retries_rate_limits_but_not_bad_keys():
    calls = []

    def flaky(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(429) if len(calls) < 3 else httpx.Response(200, json={"bars": {}})

    assert list(make_client(flaky).bars(["AAPL"], NOW - timedelta(hours=1), NOW)) == [[]]
    assert len(calls) == 3

    calls.clear()

    def unauthorized(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401)

    with pytest.raises(AlpacaAuthError):
        list(make_client(unauthorized).bars(["AAPL"], NOW - timedelta(hours=1), NOW))
    assert len(calls) == 1


def test_empty_or_inverted_window_makes_no_request():
    def handler(request):  # pragma: no cover - must not be called
        raise AssertionError("no request expected")

    client = make_client(handler)
    assert list(client.bars([], NOW - timedelta(hours=1), NOW)) == []
    assert list(client.bars(["AAPL"], NOW, NOW - timedelta(hours=1))) == []


# ---------------------------------------------------------------- planning
def test_plan_resumes_after_last_bar_and_backfills_new_symbols_separately():
    last = {
        "AAPL": datetime(2026, 9, 28, 14, 10, tzinfo=UTC),  # the oldest: sets the start
        "SPY": datetime(2026, 9, 28, 14, 30, tzinfo=UTC),
    }
    plan = plan_requests(["AAPL", "SPY", "IRDM"], last, NOW, backfill_days=10, delay_minutes=16)
    end = datetime(2026, 9, 28, 14, 44, tzinfo=UTC)  # now - 16 min, truncated to the minute
    assert plan == [
        (["AAPL", "SPY"], datetime(2026, 9, 28, 14, 11, tzinfo=UTC), end),
        (["IRDM"], end - timedelta(days=10), end),
    ]


def test_plan_is_empty_when_everything_is_up_to_date():
    end = datetime(2026, 9, 28, 14, 44, tzinfo=UTC)
    assert plan_requests(["AAPL"], {"AAPL": end}, NOW, backfill_days=10, delay_minutes=16) == []


# ---------------------------------------------------------------- storage (real Postgres)
PG_DSN = os.environ.get("PG_TEST_DSN")
needs_pg = pytest.mark.skipif(not PG_DSN, reason="set PG_TEST_DSN to run Postgres tests")


@needs_pg
def test_bars_persist_through_the_service_connection_and_upserts_are_idempotent():
    """Uses the same connect() as the services, then reads back from a second connection:
    catches writes that look fine inside a transaction but are never committed."""
    import uuid

    import psycopg

    from sentiment_pipeline.storage.postgres import ensure_schema, last_bar_times, upsert_bars
    from sentiment_pipeline.writer.postgres_writer import connect

    schema = f"test_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(PG_DSN, autocommit=True) as admin:
        admin.execute(f"CREATE SCHEMA {schema}")
    try:
        conn = connect(PG_DSN + f"?options=-csearch_path%3D{schema}")
        ensure_schema(conn)
        assert last_bar_times(conn) == {}  # plain SELECT first, like the ingestor does
        bars = parse_bars({"bars": {"AAPL": [bar_json(30), bar_json(31)]}}, feed="sip")
        assert upsert_bars(conn, bars) == 2
        assert upsert_bars(conn, parse_bars({"bars": {"AAPL": [bar_json(31, 105)]}}, "sip")) == 1
        conn.close()

        with psycopg.connect(PG_DSN, autocommit=True) as reader:
            reader.execute(f"SET search_path TO {schema}")
            rows = reader.execute("SELECT ts, close FROM price_bars ORDER BY ts").fetchall()
            assert [c for _, c in rows] == [100.0, 105.0]
            last = dict(reader.execute("SELECT ticker, max(ts) FROM price_bars GROUP BY 1"))
            assert last == {"AAPL": datetime(2026, 9, 28, 13, 31, tzinfo=UTC)}
    finally:
        with psycopg.connect(PG_DSN, autocommit=True) as admin:
            admin.execute(f"DROP SCHEMA {schema} CASCADE")
