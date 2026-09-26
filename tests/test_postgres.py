"""PostgreSQL storage tests.

Row mapping is tested everywhere. The integration tests need a real database and run
only when PG_TEST_DSN is set (CI starts a Postgres service container for them); each
test gets its own throwaway schema.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sentiment_pipeline.schemas import EnrichedArticle, LLMResult, RawArticle, TickerSignal
from sentiment_pipeline.storage.postgres import (
    article_row,
    ensure_schema,
    latest_per_id,
    schema_sql,
    signal_rows,
    sync_tickers,
    write_batch,
)
from sentiment_pipeline.watchlist import load_watchlist

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = datetime(2026, 9, 25, 14, 30, tzinfo=UTC)


def signal(ticker: str, score: float = 0.6, event: str = "earnings") -> TickerSignal:
    return TickerSignal(
        ticker=ticker,
        sentiment="positive" if score > 0.1 else "negative" if score < -0.1 else "neutral",
        score=score,
        confidence=0.8,
        event_type=event,
        is_new_info=True,
    )


def enriched(article_id: str = "a1", signals=(), model: str = "gemini", score: float = 0.5):
    raw = RawArticle(
        id=article_id,
        source="finnhub",
        publisher="Reuters",
        title=f"Headline {article_id}",
        url=f"https://example.com/{article_id}",
        tickers=["NVDA"],
        published_at=PUBLISHED,
    )
    result = LLMResult(
        index=0,
        sentiment="positive",
        score=score,
        confidence=0.9,
        topics=["earnings"],
        signals=list(signals),
    )
    return EnrichedArticle.from_parts(
        raw, result, provider="openrouter", model=model, latency_ms=40
    )


# ---------------------------------------------------------------- row mapping (no database)
def test_article_row_matches_the_insert_columns():
    row = article_row(enriched())
    assert len(row) == 19
    assert row[0] == "a1" and row[6] == ["NVDA"] and row[7] == PUBLISHED


def test_signal_rows_carry_timestamp_and_model():
    rows = signal_rows(enriched(signals=[signal("NVDA"), signal("AMD", -0.4)]))
    assert [r[1] for r in rows] == ["NVDA", "AMD"]
    assert all(r[2] == PUBLISHED and r[3] is True and r[9] == "gemini" for r in rows)


def test_duplicate_articles_in_a_batch_keep_the_last_version():
    batch = latest_per_id([enriched(score=0.1), enriched("a2"), enriched(score=0.9)])
    assert [(a.id, a.score) for a in batch] == [("a1", 0.9), ("a2", 0.5)]


def test_schema_file_ships_with_the_package():
    sql = schema_sql()
    assert "CREATE TABLE IF NOT EXISTS ticker_signals" in sql
    assert "CREATE OR REPLACE VIEW v_signals" in sql


# ---------------------------------------------------------------- integration (real Postgres)
PG_DSN = os.environ.get("PG_TEST_DSN")
needs_pg = pytest.mark.skipif(not PG_DSN, reason="set PG_TEST_DSN to run Postgres tests")


@pytest.fixture
def conn():
    import psycopg

    schema = f"test_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(PG_DSN, autocommit=True) as c:
        c.execute(f"CREATE SCHEMA {schema}")
        c.execute(f"SET search_path TO {schema}")
        try:
            ensure_schema(c)
            yield c
        finally:
            c.execute(f"DROP SCHEMA {schema} CASCADE")


@needs_pg
def test_schema_bootstrap_is_idempotent(conn):
    ensure_schema(conn)  # second run must not fail
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
        )
    }
    assert {"tickers", "articles", "ticker_signals", "v_signals"} <= tables


@needs_pg
def test_write_batch_stores_articles_and_signals(conn):
    n_art, n_sig = write_batch(conn, [enriched(signals=[signal("NVDA"), signal("AMD", -0.4)])])
    assert (n_art, n_sig) == (1, 2)
    row = conn.execute("SELECT publisher, source_tickers, topics FROM articles").fetchone()
    assert row == ("Reuters", ["NVDA"], ["earnings"])
    sigs = conn.execute("SELECT ticker, score FROM ticker_signals ORDER BY ticker").fetchall()
    assert sigs == [("AMD", pytest.approx(-0.4)), ("NVDA", pytest.approx(0.6))]


@needs_pg
def test_redelivery_and_reenrichment_overwrite_instead_of_duplicating(conn):
    write_batch(conn, [enriched(signals=[signal("NVDA"), signal("AMD")])])
    write_batch(conn, [enriched(signals=[signal("NVDA"), signal("AMD")])])  # redelivered
    # Re-enriched by another model, which no longer sees AMD in the story.
    write_batch(conn, [enriched(signals=[signal("NVDA", 0.2)], model="other-model")])
    assert conn.execute("SELECT count(*) FROM articles").fetchone()[0] == 1
    assert conn.execute("SELECT ticker, score, llm_model FROM ticker_signals").fetchall() == [
        ("NVDA", pytest.approx(0.2), "other-model")
    ]


@needs_pg
def test_failed_batch_leaves_nothing_behind(conn):
    import psycopg

    bad = enriched("bad", signals=[signal("NVDA")])
    bad = bad.model_copy(update={"score": 5.0})  # violates the CHECK constraint
    with pytest.raises(psycopg.errors.CheckViolation):
        write_batch(conn, [enriched("good", signals=[signal("AMD")]), bad])
    assert conn.execute("SELECT count(*) FROM articles").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM ticker_signals").fetchone()[0] == 0


@needs_pg
def test_view_joins_signals_with_watchlist_groups(conn):
    sync_tickers(conn, load_watchlist(str(ROOT / "config" / "watchlist.yaml")))
    write_batch(conn, [enriched(signals=[signal("NVDA"), signal("IRDM", -0.5, "product")])])
    rows = conn.execute(
        "SELECT ticker, cap_group, sector, publisher FROM v_signals ORDER BY ticker"
    ).fetchall()
    assert rows == [
        ("IRDM", "small_mid_cap", "communication", "Reuters"),
        ("NVDA", "large_cap", "technology", "Reuters"),
    ]


@needs_pg
def test_tickers_removed_from_the_watchlist_drop_out_of_group_analysis(conn):
    from sentiment_pipeline.watchlist import Watchlist

    old = Watchlist.model_validate(
        {
            "tickers": [
                {"ticker": "NVDA", "group": "large_cap", "sector": "technology"},
                {"ticker": "IONQ", "group": "small_mid_cap", "sector": "technology"},
            ]
        }
    )
    sync_tickers(conn, old)
    write_batch(conn, [enriched(signals=[signal("NVDA"), signal("IONQ")])])
    sync_tickers(conn, load_watchlist(str(ROOT / "config" / "watchlist.yaml")))  # IONQ removed

    assert conn.execute("SELECT count(*) FROM tickers WHERE ticker = 'IONQ'").fetchone()[0] == 0
    groups = dict(conn.execute("SELECT ticker, cap_group FROM v_signals").fetchall())
    assert groups == {"NVDA": "large_cap", "IONQ": None}  # history kept, but ungrouped
