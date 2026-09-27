"""Event-study data layer against a real Postgres (runs when PG_TEST_DSN is set)."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from sentiment_pipeline.analysis.data import load_bars, load_signals, write_event_returns
from sentiment_pipeline.analysis.protocol import load_protocol
from sentiment_pipeline.market.alpaca import Bar
from sentiment_pipeline.schemas import EnrichedArticle, LLMResult, RawArticle, TickerSignal
from sentiment_pipeline.storage.postgres import (
    ensure_schema,
    sync_tickers,
    upsert_bars,
    write_batch,
)
from sentiment_pipeline.watchlist import load_watchlist
from sentiment_pipeline.writer.postgres_writer import connect

ROOT = Path(__file__).resolve().parents[1]
PG_DSN = os.environ.get("PG_TEST_DSN")
pytestmark = pytest.mark.skipif(not PG_DSN, reason="set PG_TEST_DSN to run Postgres tests")
PROTOCOL = load_protocol(ROOT / "config" / "study.yaml")
MODEL = PROTOCOL.data.llm_model


@pytest.fixture
def conn():
    import psycopg

    schema = f"test_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(PG_DSN, autocommit=True) as admin:
        admin.execute(f"CREATE SCHEMA {schema}")
    c = connect(PG_DSN + f"?options=-csearch_path%3D{schema}")
    ensure_schema(c)
    sync_tickers(c, load_watchlist(str(ROOT / "config" / "watchlist.yaml")))
    try:
        yield c
    finally:
        c.close()
        with psycopg.connect(PG_DSN, autocommit=True) as admin:
            admin.execute(f"DROP SCHEMA {schema} CASCADE")


def article(i, ticker, *, source="finnhub", model=MODEL, ts_source="feed", day=29):
    raw = RawArticle(
        id=f"a{i}",
        source=source,
        title=f"t{i}",
        tickers=[ticker],
        published_at=datetime(2026, 9, day, 15, tzinfo=UTC),
        timestamp_source=ts_source,
    )
    sig = TickerSignal(
        ticker=ticker, sentiment="positive", score=0.7, confidence=0.9,
        event_type="earnings", is_new_info=True,
    )  # fmt: skip
    res = LLMResult(index=0, sentiment="positive", score=0.7, confidence=0.9, signals=[sig])
    return EnrichedArticle.from_parts(raw, res, provider="openrouter", model=model, latency_ms=1)


def test_load_signals_applies_the_protocol_and_counts_exclusions(conn):
    write_batch(
        conn,
        [
            article(1, "NVDA"),
            article(2, "IRDM"),
            article(3, "NVDA", source="bbc_business"),
            article(4, "NVDA", ts_source="ingested"),
            article(5, "NVDA", model="llama3.2:3b"),
            article(6, "IONQ"),  # not in the watchlist any more
            article(7, "NVDA", day=25),  # before collection_start
        ],
    )
    df, excluded = load_signals(conn, PROTOCOL, PROTOCOL.data.collection_start, date(2026, 10, 30))
    assert sorted(df["article_id"]) == ["a1", "a2"]
    assert dict(zip(df["ticker"], df["cap_group"], strict=True)) == {
        "NVDA": "large_cap",
        "IRDM": "small_mid_cap",
    }
    assert excluded == {
        "not from the registered news source": 1,
        "no source timestamp": 1,
        "different LLM model": 1,
        "ticker no longer in the watchlist": 1,
    }
    pilot, _ = load_signals(conn, PROTOCOL, None, date(2026, 10, 30))
    assert "a7" in set(pilot["article_id"])  # the pilot ignores collection_start


def test_bars_round_trip_and_results_are_rewritten_each_run(conn):
    t = datetime(2026, 9, 29, 14, 30, tzinfo=UTC)
    upsert_bars(conn, [Bar("SPY", t, 1, 1, 1, 500.0, 10, 1, 1, "sip")])
    bars = load_bars(conn, ["SPY"], t.replace(hour=0), t.replace(hour=23))
    assert list(bars["close"]) == [500.0] and str(bars["ts"].dt.tz) == "UTC"

    results = pd.DataFrame(
        [
            {"article_id": "a1", "ticker": "NVDA", "horizon": "1d", "status": "ok", "t0": t,
             "t1": t, "p0": 1.0, "p1": 1.1, "spy_p0": 1.0, "spy_p1": 1.0, "ret": 0.1,
             "spy_ret": 0.0, "abnormal_ret": 0.1, "traded_in_window": True},
            {"article_id": "a1", "ticker": "NVDA", "horizon": "5d", "status": "pending",
             "t0": t, "t1": None, "p0": None, "p1": None, "spy_p0": None, "spy_p1": None,
             "ret": None, "spy_ret": None, "abnormal_ret": None, "traded_in_window": None},
        ]
    )  # fmt: skip
    assert write_event_returns(conn, results, "pilot") == 2
    assert write_event_returns(conn, results.iloc[:1], "registered") == 1
    rows = conn.execute(
        "SELECT horizon, status, abnormal_ret, run_label FROM event_returns"
    ).fetchall()
    assert rows == [("1d", "ok", 0.1, "registered")]


def test_dashboard_queries_run_and_count_what_the_analysis_counts(conn):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_study_dashboard", ROOT / "scripts" / "build_study_dashboard.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    write_batch(
        conn,
        [
            article(1, "NVDA"),
            article(2, "IRDM"),
            article(3, "IRDM", day=30),
            article(4, "IRDM", source="bbc_business"),
            article(5, "IRDM", model="llama3.2:3b"),
            article(6, "IRDM", day=25),  # before collection_start
        ],
    )
    upsert_bars(conn, [Bar("SPY", datetime.now(UTC), 1, 1, 1, 500.0, 10, 1, 1, "sip")])
    dashboard = builder.build(PROTOCOL)
    results = {}
    for title, sql in builder.all_sql(dashboard):
        sql = sql.replace("$__timeFilter(published_at)", "published_at > now() - interval '1 year'")
        results[title] = conn.execute(sql).fetchall()  # every panel query must run

    analysis, _ = load_signals(conn, PROTOCOL, PROTOCOL.data.collection_start, date(2026, 10, 30))
    n_small = int((analysis["cap_group"] == "small_mid_cap").sum())
    assert results["Small-cap signals (stopping rule)"] == [(n_small,)] == [(2,)]
    assert results["Large-cap signals"] == [(1,)]
    assert dict(results["Signals excluded from the study"]) == {
        "eligible": 3,
        "not from the registered news source": 1,
        "different LLM model": 1,
    }
    per_ticker = {row[0]: row[3] for row in results["Signals per ticker (collection period)"]}
    assert per_ticker["IRDM"] == 2 and per_ticker["NVDA"] == 1 and per_ticker["AAPL"] == 0
    assert len(per_ticker) == 40  # every watchlist ticker, with or without signals
    coverage = {row[0]: row[1:3] for row in results["Price data coverage (last 7 days)"]}
    assert coverage["SPY"] == ("Benchmark", 1) and coverage["IRDM"] == ("Small caps", 0)
