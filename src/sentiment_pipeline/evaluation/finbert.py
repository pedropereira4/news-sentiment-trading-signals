"""FinBERT baseline: a finance-tuned BERT that labels the text, not a particular company.

Runs locally on CPU (needs the `eval` extra: transformers + torch). Predictions are cached in
a CSV so scoring can be re-run without the model.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

MODEL = "ProsusAI/finbert"
Predict = Callable[[Sequence[str]], list[str]]


def load_finbert(model: str = MODEL) -> Predict:
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise SystemExit('FinBERT needs the eval extra: pip install -e ".[eval]"') from exc
    clf = pipeline("text-classification", model=model, truncation=True, max_length=512)

    def predict(texts: Sequence[str]) -> list[str]:
        return [out["label"].lower() for out in clf(list(texts), batch_size=16)]

    return predict


def run_finbert(texts: pd.Series, predict: Predict, cache: Path) -> pd.DataFrame:
    """Labels for every item (index = item), computing only what is not cached yet."""
    cached = pd.read_csv(cache, index_col="item") if cache.exists() else pd.DataFrame()
    todo = texts[~texts.index.isin(cached.index)]
    if len(todo):
        start = time.perf_counter()
        labels = predict(list(todo))
        per_item_ms = (time.perf_counter() - start) * 1000 / len(todo)
        new = pd.DataFrame({"label": labels, "latency_ms": per_item_ms}, index=todo.index)
        new.index.name = "item"
        cached = pd.concat([cached, new]).sort_index()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(cache)
    return cached.loc[texts.index]
