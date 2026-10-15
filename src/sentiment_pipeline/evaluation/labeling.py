"""Terminal tool for blind labelling. Resumable: every answer is saved immediately."""

from __future__ import annotations

import csv
import textwrap
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

LABELS = {"1": "negative", "2": "neutral", "3": "positive", "0": "not_about"}
LABEL_FIELDS = ["item", "article_id", "ticker", "label", "labelled_at"]

GUIDE = """
For each item, judge THE STOCK OF THE COMPANY SHOWN, as an investor reading the news at the
time it was published (do not look up what the price did afterwards):

  3 positive   good news for this company's stock (new contract, beat, upgrade, approval...)
  2 neutral    no clear impact: recaps, opinion, lists, mixed or minor news
  1 negative   bad news for this company's stock (miss, downgrade, lawsuit, guidance cut...)
  0 not about  the article is not really about this company (only a passing mention)

  s skip (decide later)   q quit (progress is saved)
"""


def read_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LABEL_FIELDS)
    return pd.read_csv(path, dtype={"article_id": str, "ticker": str})


def format_item(row, position: int, total: int) -> str:
    group = "large cap" if row.cap_group == "large_cap" else "small cap"
    when = str(row.published_at)[:16]
    lines = [
        "",
        f"[{position}/{total}]  {row.ticker} - {row.company} ({group})"
        f"  ·  {row.publisher}  ·  {when}",
        "",
        *textwrap.wrap(f"Headline: {row.title}", 96),
    ]
    if isinstance(row.summary, str) and row.summary:
        lines += ["", *textwrap.wrap(f"Summary:  {row.summary}", 96)]
    return "\n".join(lines)


def run_labelling(
    sample: pd.DataFrame,
    labels_path: Path,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> int:
    """Ask for every item not labelled yet; returns how many labels were added."""
    done = set(read_labels(labels_path)["item"].astype(int))
    todo = sample[~sample["item"].isin(done)]
    show(GUIDE)
    show(f"{len(done)} of {len(sample)} labelled so far.")
    new_file = not labels_path.exists()
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with labels_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        if new_file:
            writer.writeheader()
        for row in todo.itertuples(index=False):
            show(format_item(row, len(done) + added + 1, len(sample)))
            while True:
                key = ask(f"\nFor {row.ticker}: 1 neg · 2 neutral · 3 pos · 0 not about · s · q > ")
                key = key.strip().lower()
                if key in LABELS or key in {"s", "q"}:
                    break
                show("Please answer 1, 2, 3, 0, s or q.")
            if key == "q":
                break
            if key == "s":
                continue
            now = datetime.now(UTC).isoformat(timespec="seconds")
            writer.writerow({
                "item": row.item, "article_id": row.article_id, "ticker": row.ticker,
                "label": LABELS[key], "labelled_at": now,
            })  # fmt: skip
            f.flush()
            added += 1
    show(f"\nSaved. {len(done) + added} of {len(sample)} labelled.")
    return added
