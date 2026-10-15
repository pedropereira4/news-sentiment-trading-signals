"""Draw the evaluation sample and write the blind file used for labelling.

The sample comes from the same signals the event study analyses (collection period, protocol
filters). It is stratified by group, because the study compares large and small caps and the
LLM may be better on one than the other, and shuffled, so labelling can stop at any point and
what has been labelled is still a random sample of both groups.

The blind file holds what a reader saw at publication time and nothing the LLM produced.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import psycopg

from sentiment_pipeline.analysis.data import load_signals
from sentiment_pipeline.analysis.protocol import Protocol

GROUP_ORDER = ("large_cap", "small_mid_cap")
SUMMARY_CHARS = 280  # the enricher shows the LLM the same amount of summary

_TEXT_SQL = """
SELECT a.id AS article_id, a.title, a.summary, a.publisher, a.published_at
FROM articles a WHERE a.id = ANY(%s)
"""


def draw_sample(signals: pd.DataFrame, n_per_group: int, seed: int) -> pd.DataFrame:
    """`n_per_group` random signals per cap group, in a random order, numbered from 1."""
    rng = np.random.default_rng(seed)
    parts = []
    for group in GROUP_ORDER:  # fixed order and sorting: same seed, same sample
        g = signals[signals["cap_group"] == group].sort_values(
            ["published_at", "article_id", "ticker"]
        )
        idx = rng.choice(len(g), size=min(n_per_group, len(g)), replace=False)
        parts.append(g.iloc[np.sort(idx)])
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    sample.insert(0, "item", range(1, len(sample) + 1))
    return sample[["item", "article_id", "ticker", "cap_group"]]


def clean_summary(summary: str | None) -> str:
    text = re.sub(r"<[^>]+>", "", summary or "").strip()
    return text[:SUMMARY_CHARS]


def model_text(title: str, summary: str | None) -> str:
    """The text both models classify: headline plus the start of the summary."""
    s = clean_summary(summary)
    return f"{title.strip()} — {s}" if s else title.strip()


def build_blind_sample(
    conn: psycopg.Connection, protocol: Protocol, n_per_group: int, seed: int
) -> pd.DataFrame:
    if protocol.closing is None:
        raise SystemExit("The collection is not closed yet: sample after `closing` is recorded.")
    signals, _ = load_signals(
        conn, protocol, protocol.data.collection_start, protocol.closing.collection_end
    )
    sample = draw_sample(signals, n_per_group, seed)
    with conn.cursor() as cur:
        cur.execute(_TEXT_SQL, (list(sample["article_id"].unique()),))
        cols = [c.name for c in cur.description]
        texts = pd.DataFrame(cur.fetchall(), columns=cols)
        cur.execute("SELECT ticker, name FROM tickers")
        names = dict(cur.fetchall())
    out = sample.merge(texts, on="article_id", how="left")
    out["company"] = out["ticker"].map(names).fillna("")
    out["summary"] = out["summary"].map(clean_summary)
    return out.sort_values("item")[
        ["item", "article_id", "ticker", "company", "cap_group", "published_at", "publisher",
         "title", "summary"]
    ].reset_index(drop=True)  # fmt: skip
