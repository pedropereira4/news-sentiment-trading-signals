"""Reading signals and prices for the event study, and storing its results."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd
import psycopg

from sentiment_pipeline.analysis.prices import ET
from sentiment_pipeline.analysis.protocol import Protocol

_SIGNALS_SQL = """
SELECT s.article_id, s.ticker, s.published_at, s.score, s.sentiment, s.confidence,
       s.event_type, s.is_new_info, s.feed_timestamp, s.llm_model,
       a.source, t.cap_group, t.sector
FROM ticker_signals s
JOIN articles a ON a.id = s.article_id
LEFT JOIN tickers t ON t.ticker = s.ticker
WHERE s.published_at >= %s AND s.published_at < %s
ORDER BY s.published_at
"""


def _day_start(d: date) -> datetime:
    return datetime.combine(d, time(0), tzinfo=ET)


def _frame(cur: psycopg.Cursor) -> pd.DataFrame:
    cols = [c.name for c in cur.description]
    return pd.DataFrame(cur.fetchall(), columns=cols)


def load_signals(
    conn: psycopg.Connection, protocol: Protocol, since: date | None, until: date
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Signals eligible under the protocol, plus how many were excluded and why."""
    start = _day_start(since) if since else datetime(2000, 1, 1, tzinfo=ET)
    with conn.cursor() as cur:
        cur.execute(_SIGNALS_SQL, (start, _day_start(until + timedelta(days=1))))
        df = _frame(cur)
    excluded: dict[str, int] = {}
    rules = [
        ("not from the registered news source", df["source"] != protocol.data.news_source),
        ("no source timestamp", ~df["feed_timestamp"].astype(bool)),
        ("different LLM model", df["llm_model"] != protocol.data.llm_model),
        ("ticker no longer in the watchlist", df["cap_group"].isna()),
    ]
    keep = pd.Series(True, index=df.index)
    for reason, drop in rules:
        drop = drop & keep
        excluded[reason] = int(drop.sum())
        keep &= ~drop
    out = df[keep].reset_index(drop=True)
    if not out.empty:
        out["published_at"] = pd.to_datetime(out["published_at"], utc=True)
    return out, excluded


def load_bars(
    conn: psycopg.Connection, tickers: list[str], start: datetime, end: datetime
) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ticker, ts, close FROM price_bars "
            "WHERE ticker = ANY(%s) AND ts >= %s AND ts < %s ORDER BY ticker, ts",
            (tickers, start, end),
        )
        df = _frame(cur)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _py(value: Any) -> Any:
    """NaN/NaT -> NULL and numpy scalars -> Python ones (psycopg does not adapt numpy.bool_)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return value.item() if hasattr(value, "item") and not isinstance(value, datetime) else value


def write_event_returns(conn: psycopg.Connection, results: pd.DataFrame, run_label: str) -> int:
    cols = [
        "article_id", "ticker", "horizon", "status", "t0", "t1", "p0", "p1", "spy_p0",
        "spy_p1", "ret", "spy_ret", "abnormal_ret", "traded_in_window",
    ]  # fmt: skip
    rows: list[tuple[Any, ...]] = [
        tuple(_py(v) for v in r) + (run_label,)
        for r in results[cols].itertuples(index=False, name=None)
    ]
    sql = (
        f"INSERT INTO event_returns ({', '.join(cols)}, run_label) "
        f"VALUES ({', '.join(['%s'] * (len(cols) + 1))})"
    )
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM event_returns")  # each run is a full recomputation
        if rows:
            cur.executemany(sql, rows)
    return len(rows)
